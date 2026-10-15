import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/paper-drafts/tools/verify_anonymous_submission.py"
SPEC = importlib.util.spec_from_file_location(
    "verify_anonymous_submission", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerifyAnonymousSubmissionTest(unittest.TestCase):
    def test_identity_bearing_form_reference_is_rejected(self):
        text = '<a href="forms/aas_author_commitment_2025_official.pdf">form</a>'
        matches = MODULE.find_forbidden_patterns(text)
        self.assertIn("identity-bearing form reference", {name for name, _ in matches})

    def test_normal_anonymous_text_has_no_form_match(self):
        matches = MODULE.find_forbidden_patterns(
            "SAQT performs cropped vertical single-column recognition."
        )
        self.assertNotIn(
            "identity-bearing form reference", {name for name, _ in matches}
        )

    def test_stale_pdf_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "paper.md"
            html = root / "paper.html"
            pdf = root / "paper.pdf"
            for path in (source, html, pdf):
                path.write_text(path.name, encoding="utf-8")
            os.utime(source, ns=(100, 100))
            os.utime(pdf, ns=(200, 200))
            os.utime(html, ns=(300, 300))
            self.assertEqual(
                MODULE.artifact_freshness_failures(pdf, html, source),
                ["anonymous PDF is older than the generated HTML"],
            )

    def test_fresh_build_order_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "paper.md"
            html = root / "paper.html"
            pdf = root / "paper.pdf"
            for path in (source, html, pdf):
                path.write_text(path.name, encoding="utf-8")
            os.utime(source, ns=(100, 100))
            os.utime(html, ns=(200, 200))
            os.utime(pdf, ns=(300, 300))
            self.assertEqual(
                MODULE.artifact_freshness_failures(pdf, html, source), []
            )


if __name__ == "__main__":
    unittest.main()
