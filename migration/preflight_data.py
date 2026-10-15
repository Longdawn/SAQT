"""Verify original annotations and dataset construction before any target training."""
import argparse
import hashlib
import json
import os
import pathlib
import pickle
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="migration_reports/data_preflight.json")
    cli = parser.parse_args()
    os.chdir(ROOT)
    expected = json.loads((ROOT / "migration/source_data_fingerprints.json").read_text())
    hashes = expected["label_and_annotation_sha256"]
    for name, digest in hashes.items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Missing or different original annotation: {name}")
    assets = json.loads((ROOT / "migration/assets.json").read_text())["assets"]
    for item in assets:
        if item["kind"] in {"file", "verify-only"}:
            path = ROOT / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise RuntimeError(f"Missing or different asset: {item['path']}")
    from datasets import build_dataset
    from util.slconfig import SLConfig

    cfg = SLConfig.fromfile("config/MTHV2_stage1_query_budget.py")._cfg_dict.to_dict()
    cfg.update(dataset_file="mth_combo", fix_size=False, eval=False, distributed=False)
    args = argparse.Namespace(**cfg)
    counts = {}
    for split in ["train", "valid"]:
        dataset = build_dataset(split, args)
        assert len(dataset) == expected[split + "_lines"]
        assert len(dataset.charset) == expected["charset_size"]
        for child in dataset.datasets:
            assert child.use_char_boxes, f"Dummy boxes: {child.raw_root_path}"
            # Verify every image exists, without decoding every image on the new machine.
            for sample in child.samples:
                image_path = pathlib.Path(child.images_path) / (str(sample["id"]) + "." + child.im_extension)
                assert image_path.is_file(), str(image_path)
        counts[split] = len(dataset)
    charset_path = ROOT / "logs/mth1000_mth1200_stage1_0512-1755/charset.pkl"
    with charset_path.open("rb") as handle:
        assert len(pickle.load(handle)) == 6700
    result = {"status": "PASS", "counts": counts, "charset": expected["charset_size"],
              "stage1_steps_per_epoch_batch2": counts["train"] // 2,
              "original_annotation_hashes_verified": len(hashes),
              "scope": "Data integrity and dataset construction; no GPU training or cross-device CER claim"}
    output = ROOT / cli.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
