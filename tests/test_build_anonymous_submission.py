import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/paper-drafts/tools/build_anonymous_submission.py"
SPEC = importlib.util.spec_from_file_location("build_anonymous_submission", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

INTEGRATION_PATH = ROOT / "docs/paper-drafts/tools/integrate_final_experiment_results.py"
INTEGRATION_SPEC = importlib.util.spec_from_file_location(
    "integrate_final_experiment_results_for_anonymous_test", INTEGRATION_PATH
)
INTEGRATION_MODULE = importlib.util.module_from_spec(INTEGRATION_SPEC)
INTEGRATION_SPEC.loader.exec_module(INTEGRATION_MODULE)


class BuildAnonymousSubmissionTest(unittest.TestCase):
    def test_optional_figure4_is_included_only_when_pdf_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            paper_root = Path(directory)
            figures = paper_root / "figures"
            figures.mkdir()
            with mock.patch.object(MODULE, "PAPER_ROOT", paper_root):
                self.assertNotIn(
                    "figure4_hdrc_low_resource", MODULE.available_figure_stems()
                )
                (figures / "figure4_hdrc_low_resource.pdf").write_bytes(b"pdf")
                self.assertIn(
                    "figure4_hdrc_low_resource", MODULE.available_figure_stems()
                )

    def test_generated_qclc_method_math_renders_without_literal_dollars(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(MODULE, "ASSET_ROOT", Path(directory)):
                rendered = MODULE.markdown_to_html(
                    INTEGRATION_MODULE.render_qclc_method(), {}
                )
        self.assertNotIn("$", rendered)
        self.assertGreaterEqual(rendered.count('<img class="'), 7)


if __name__ == "__main__":
    unittest.main()
