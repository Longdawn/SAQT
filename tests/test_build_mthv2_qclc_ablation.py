import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/paper-drafts/tools/build_mthv2_qclc_ablation.py"
SPEC = importlib.util.spec_from_file_location("build_mthv2_qclc_ablation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def result(checkpoint, bias=0.0):
    return {
        "samples": 5,
        "dataset": "mth_combo",
        "split": "test",
        "checkpoint": checkpoint,
        "micro_cer": 0.1,
        "micro_ar": 0.9,
        "micro_cr": 0.91,
        "blank_bias": bias,
        "nonblank_bias": 0.4 if bias else 0.0,
        "ratio_nonblank_bias": 0.0,
        "empty_pred_rate": 0.02,
        "pred_gt_len_ratio": 0.98,
        "by_gt_len_bin": {
            key: {"cer_micro": 0.1} for key in MODULE.BUCKETS
        },
    }


class BuildQclcAblationTest(unittest.TestCase):
    def test_status_requires_positive_gate_and_complete_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status"
            path.write_text(
                "BEST_VALID epoch=2 cer=0.11 baseline=0.12 now\n"
                "DONE_TEST_BIAS now\nDONE_MATCHED_BASELINE_TEST_BIAS now\nEXIT:0 now\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.validate_qclc_status(path, 0.12), 0.11)
            with self.assertRaises(MODULE.IncompleteEvidence):
                MODULE.validate_qclc_status(path, 0.10)

    def test_full_configs_may_differ_only_by_module_and_output_fields(self):
        manifest = {
            "allowed_config_differences": ["config_file", "output_dir"],
            "qclc_only_fields": ["use_ctc_length_calibration", "ctc_length_init"],
            "equivalent_model_names": ["dino", "saqt"],
        }
        baseline = {
            "config_file": "base.py",
            "output_dir": "base",
            "modelname": "dino",
            "lr": 1e-5,
        }
        qclc_config = {
            "config_file": "qclc.py",
            "output_dir": "qclc",
            "modelname": "saqt",
            "lr": 1e-5,
            "use_ctc_length_calibration": True,
            "ctc_length_init": 8.0,
        }
        MODULE.validate_matched_configs(baseline, qclc_config, manifest)
        qclc_config["lr"] = 2e-5
        with self.assertRaises(MODULE.IncompleteEvidence):
            MODULE.validate_matched_configs(baseline, qclc_config, manifest)

    def test_collect_requires_validation_selected_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = MODULE.ROOT
            MODULE.ROOT = Path(tmp)
            try:
                run_dir = MODULE.ROOT / "logs/run"
                run_dir.mkdir(parents=True)
                config = {
                    "seed": 42,
                    "resume": "head.pth",
                    "dataset_file": "mth_combo",
                    "batch_size": 2,
                    "lr": 1e-5,
                    "epochs": 4,
                    "max_iterations": 10000,
                    "max_optimizer_steps": 16000,
                }
                (run_dir / "config_args_all.json").write_text(json.dumps(config))
                clean = result("logs/run/checkpoint_best_regular.pth")
                calibrated = result("logs/run/checkpoint_best_regular.pth", -2.0)
                (run_dir / "clean.json").write_text(json.dumps(clean))
                (run_dir / "sweep.json").write_text(
                    json.dumps([{
                        "samples": 7,
                        "cer_micro": 0.09,
                        "blank_bias": -2.0,
                        "nonblank_bias": 0.4,
                        "ratio_nonblank_bias": 0.0,
                    }])
                )
                (run_dir / "bias.json").write_text(json.dumps(calibrated))
                manifest = {
                    "expected_test_samples": 5,
                    "expected_valid_samples": 7,
                    "shared_config": config,
                }
                run = {
                    "variant": "matched_baseline",
                    "run_dir": "logs/run",
                    "clean_file": "clean.json",
                    "sweep_file": "sweep.json",
                    "bias_glob": "bias.json",
                }
                row = MODULE.collect(run, manifest)
                self.assertEqual(row["blank_bias"], -2.0)
                self.assertEqual(row["calibrated_len_1_cer"], 0.1)
                calibrated["blank_bias"] = -1.6
                (run_dir / "bias.json").write_text(json.dumps(calibrated))
                with self.assertRaises(MODULE.IncompleteEvidence):
                    MODULE.collect(run, manifest)
            finally:
                MODULE.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
