import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/paper-drafts/tools/verify_latex_package.py"
SPEC = importlib.util.spec_from_file_location("verify_latex_package", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyLatexPackageTest(unittest.TestCase):
    def test_matching_zip_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sections").mkdir()
            (root / "main.tex").write_text("main", encoding="utf-8")
            (root / "sections/a.tex").write_text("section", encoding="utf-8")
            archive_path = root / "paper.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(root / "main.tex", "main.tex")
                archive.write(root / "sections/a.tex", "sections/a.tex")
            self.assertEqual(MODULE.zip_sync_failures(archive_path, root), [])

    def test_stale_zip_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "main.tex"
            source.write_text("current", encoding="utf-8")
            archive_path = root / "paper.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("main.tex", "stale")
            failures = MODULE.zip_sync_failures(archive_path, root)
            self.assertEqual(
                failures,
                ["zip content differs from current source: main.tex"],
            )


if __name__ == "__main__":
    unittest.main()
