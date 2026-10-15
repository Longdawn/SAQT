import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/paper-drafts/tools/verify_figure_assets.py"
SPEC = importlib.util.spec_from_file_location("verify_figure_assets", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyFigureAssetsTest(unittest.TestCase):
    def test_valid_editable_vector_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                '<text x="1" y="5">CER</text></svg>',
                encoding="utf-8",
            )
            (root / "sample.pdf").write_bytes(b"%PDF-1.4\n" + b"0" * 1200)
            (root / "sample.csv").write_text("x,y\n1,2\n", encoding="utf-8")
            failures = MODULE.inspect_figure(
                root, MODULE.FigureSpec("sample", "sample.csv")
            )
            self.assertEqual(failures, [])

    def test_data_svg_rejects_path_only_text_and_embedded_raster(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.svg"
            path.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                '<path d="M0 0L1 1"/><image href="data:image/png;base64,AA=="/>'
                "</svg>",
                encoding="utf-8",
            )
            failures = MODULE.inspect_svg(path, allow_embedded_images=False)
            self.assertTrue(any("no editable <text>" in item for item in failures))
            self.assertTrue(any("embedded raster" in item for item in failures))


if __name__ == "__main__":
    unittest.main()

