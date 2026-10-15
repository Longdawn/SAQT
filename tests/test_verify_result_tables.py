import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/paper-drafts/tools/verify_result_tables.py"
SPEC = importlib.util.spec_from_file_location("verify_result_tables", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VerifyResultTablesTest(unittest.TestCase):
    def test_current_paper_contains_required_result_evidence(self):
        text = MODULE.PAPER_RESULTS.read_text(encoding="utf-8")
        self.assertEqual(MODULE.paper_structure_failures(text), [])

    def test_missing_structure_pretraining_section_is_rejected(self):
        text = "\n".join(
            fragment
            for fragment in MODULE.REQUIRED_PAPER_FRAGMENTS
            if "结构查询预训练的下界对照" not in fragment
        )
        failures = MODULE.paper_structure_failures(text)
        self.assertTrue(any("结构查询预训练的下界对照" in item for item in failures))

    def test_qclc_result_requires_reproducible_method(self):
        result_text = "\n".join(MODULE.REQUIRED_PAPER_FRAGMENTS) + "\n" + MODULE.QCLC_RESULT_HEADING
        failures = MODULE.paper_structure_failures(result_text, method_text="")
        self.assertTrue(any("QCLC result present" in item for item in failures))

        method_text = "\n".join(MODULE.REQUIRED_QCLC_METHOD_FRAGMENTS)
        self.assertEqual(
            MODULE.paper_structure_failures(result_text, method_text=method_text),
            [],
        )

    def test_final_results_do_not_require_seed_prose_in_abstract_or_conclusion(self):
        results = MODULE.FINAL_RESULTS_MARKER
        self.assertEqual(MODULE.cross_section_failures(results, "", ""), [])

    def test_current_integrated_paper_has_consistent_final_blocks(self):
        results = MODULE.PAPER_RESULTS.read_text(encoding="utf-8")
        abstract = MODULE.PAPER_ABSTRACT.read_text(encoding="utf-8")
        conclusion = MODULE.PAPER_CONCLUSION.read_text(encoding="utf-8")
        self.assertIn(MODULE.FINAL_RESULTS_MARKER, results)
        self.assertEqual(
            MODULE.cross_section_failures(results, abstract, conclusion), []
        )

    def test_low_resource_budget_disclosure_is_required(self):
        failures = MODULE.protocol_disclosure_failures("")
        self.assertEqual(
            len(failures), len(MODULE.REQUIRED_LOW_RESOURCE_PROTOCOL_FRAGMENTS)
        )
        complete = "\n".join(MODULE.REQUIRED_LOW_RESOURCE_PROTOCOL_FRAGMENTS)
        self.assertEqual(MODULE.protocol_disclosure_failures(complete), [])

    def test_current_experimental_setup_discloses_sample_cap(self):
        text = MODULE.PAPER_EXPERIMENTS.read_text(encoding="utf-8")
        self.assertEqual(MODULE.protocol_disclosure_failures(text), [])


if __name__ == "__main__":
    unittest.main()
