"""Relay only immutable selected assets over SSH, without sharing SSH keys.

Run on the local machine after the destination SSH alias and cloned repo exist.
Both SSH connections use local authentication. Never starts a GPU process.
"""
import argparse
import json
import pathlib
import shlex
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

PRODUCER = r'''
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path('/home/ubuntu/DTLR').resolve()
assets=ASSET_PATHS
class Reader:
    def __init__(self,handle): self.handle=handle; self.digest=hashlib.sha256()
    def read(self,size=-1):
        data=self.handle.read(size); self.digest.update(data); return data
records={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for name in assets:
        item=root/name
        paths=sorted(item.rglob('*')) if item.is_dir() else [item]
        for path in paths:
            if path.is_symlink(): raise RuntimeError('Unexpected symlink: '+str(path))
            if not path.is_file(): continue
            if not path.resolve().is_relative_to(root): raise RuntimeError('Path outside source root')
            relative=path.relative_to(root).as_posix()
            before=path.stat()
            info=tarfile.TarInfo(relative); info.size=before.st_size; info.mode=0o644
            with path.open('rb') as handle:
                reader=Reader(handle); archive.addfile(info,reader)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError('Source changed during transfer: '+relative)
            records[relative]={'sha256':reader.digest.hexdigest(),'bytes':before.st_size}
    data=json.dumps(records,sort_keys=True).encode()
    info=tarfile.TarInfo('_cdtr_transfer_manifest.json'); info.size=len(data)
    archive.addfile(info,io.BytesIO(data))
'''

CONSUMER = r'''
import hashlib,json,os,pathlib,sys,tarfile,tempfile
root=pathlib.Path(sys.argv[1]).resolve()
allowed=pathlib.Path('/root/autodl-tmp').resolve()
if not root.is_relative_to(allowed) or root==allowed or not (root/'models/dino').is_dir():
    raise RuntimeError('Destination must be a cloned project under /root/autodl-tmp')
records={}; expected=None; reused=0
def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''): h.update(block)
    return h.hexdigest()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
    for member in archive:
        if member.name=='_cdtr_transfer_manifest.json':
            expected=json.load(archive.extractfile(member)); continue
        path=(root/member.name).resolve()
        if not path.is_relative_to(root) or not member.isfile():
            raise RuntimeError('Unexpected transfer entry: '+member.name)
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary=tempfile.mkstemp(prefix=path.name+'.migration-part-',dir=path.parent)
        digest=hashlib.sha256(); size=0
        with os.fdopen(fd,'wb') as output, archive.extractfile(member) as source:
            for block in iter(lambda:source.read(1048576),b''):
                output.write(block); digest.update(block); size+=len(block)
        actual=digest.hexdigest()
        if size!=member.size: raise RuntimeError('Incomplete file: '+member.name)
        if path.exists():
            if path.stat().st_size!=size or file_hash(path)!=actual:
                raise RuntimeError('Refusing to overwrite different destination file: '+str(path))
            pathlib.Path(temporary).unlink(); reused+=1
        else: os.rename(temporary,path)
        records[member.name]={'sha256':actual,'bytes':size}
        if len(records)%10000==0: print('Verified files:',len(records),flush=True)
if expected is None or records!=expected: raise RuntimeError('End-to-end manifest mismatch')
report=root/'migration_reports/assets_sha256.json'
report.parent.mkdir(parents=True,exist_ok=True)
report.write_text(json.dumps({'status':'PASS','files':records,'reused':reused},indent=2))
print('TRANSFER PASS',len(records),'files;',sum(x['bytes'] for x in records.values()),'bytes',flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="3090-Codex", help="Local SSH alias for the source")
    parser.add_argument("--destination", required=True, help="Local SSH alias for the AutoDL instance")
    parser.add_argument("--target-root", required=True, help="Existing clone, e.g. /root/autodl-tmp/SAQT")
    cli = parser.parse_args()
    for value in [cli.source, cli.destination]:
        if not value or value.startswith("-") or any(c.isspace() for c in value):
            parser.error("Provide one SSH alias or user@host without whitespace")
    if not cli.target_root.startswith("/root/autodl-tmp/"):
        parser.error("Destination must be under /root/autodl-tmp/")
    assets = json.loads((ROOT / "migration/assets.json").read_text())["assets"]
    paths = [entry["path"] for entry in assets if entry["kind"] != "verify-only"]
    producer = PRODUCER.replace("ASSET_PATHS", repr(paths))
    source_command = "nice -n 15 python3 -c " + shlex.quote(producer)
    destination_command = "python3 -c " + shlex.quote(CONSUMER) + " " + shlex.quote(cli.target_root)
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15"]
    source = subprocess.Popen(["ssh", *options, cli.source, source_command], stdout=subprocess.PIPE)
    try:
        destination = subprocess.Popen(["ssh", *options, cli.destination, destination_command], stdin=source.stdout)
        source.stdout.close()
        destination_status = destination.wait()
        source_status = source.wait()
        if destination_status or source_status:
            raise SystemExit(f"Transfer failed: source={source_status}, destination={destination_status}")
    finally:
        if source.poll() is None:
            source.terminate()
            source.wait()


if __name__ == "__main__":
    main()
