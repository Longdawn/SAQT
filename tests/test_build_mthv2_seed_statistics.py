import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/paper-drafts/tools/build_mthv2_seed_statistics.py"
SPEC = importlib.util.spec_from_file_location("build_mthv2_seed_statistics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildMthv2SeedStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "allowed_config_differences": ["seed", "output_dir", "config_file"],
            "equivalent_model_names": ["dino", "saqt"],
        }
        self.expected = {
            "dataset_file": "mth_combo",
            "resume": "fixed-head.pth",
            "batch_size": 2,
            "lr": 1e-5,
            "ctc_count_loss_coef": 0.01,
            "mth1000_length_balance": False,
        }
        self.config = {"seed": 43, **self.expected}

    def test_protocol_accepts_only_seed_difference(self):
        MODULE.validate_run_config({"seed": 43}, self.config, self.expected)
        MODULE.validate_run_config(
            {"seed": 44}, {"seed": 44, **self.expected}, self.expected
        )

    def test_protocol_rejects_wrong_head_or_loss(self):
        wrong_head = dict(self.config, resume="another-head.pth")
        with self.assertRaises(MODULE.IncompleteEvidence):
            MODULE.validate_run_config({"seed": 43}, wrong_head, self.expected)
        wrong_loss = dict(self.config, ctc_count_loss_coef=0.0)
        with self.assertRaises(MODULE.IncompleteEvidence):
            MODULE.validate_run_config({"seed": 43}, wrong_loss, self.expected)

    def test_calibration_match_includes_ratio_bias(self):
        best = {
            "blank_bias": -2.0,
            "nonblank_bias": 0.8,
            "ratio_nonblank_bias": 0.0,
        }
        self.assertTrue(MODULE.calibration_matches(dict(best), best))
        candidate = dict(best, ratio_nonblank_bias=0.4)
        self.assertFalse(MODULE.calibration_matches(candidate, best))

    def test_full_config_comparison_allows_only_seed_and_paper_aliases(self):
        reference = dict(
            self.config,
            seed=42,
            output_dir="logs/seed42",
            config_file="config/legacy.py",
            modelname="dino",
        )
        repeat = dict(
            self.config,
            seed=43,
            output_dir="logs/seed43",
            config_file="config/paper.py",
            modelname="saqt",
        )
        MODULE.validate_only_seed_difference(
            {"seed": 43}, repeat, reference, self.manifest
        )

    def test_full_config_comparison_rejects_unlisted_difference(self):
        reference = dict(self.config, seed=42, hidden_dim=256, modelname="dino")
        repeat = dict(self.config, seed=43, hidden_dim=384, modelname="saqt")
        with self.assertRaises(MODULE.IncompleteEvidence):
            MODULE.validate_only_seed_difference(
                {"seed": 43}, repeat, reference, self.manifest
            )


if __name__ == "__main__":
    unittest.main()
