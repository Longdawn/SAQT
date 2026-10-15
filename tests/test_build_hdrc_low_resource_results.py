import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/paper-drafts/tools/build_hdrc_low_resource_results.py"
SPEC = importlib.util.spec_from_file_location("build_hdrc_low_resource_results", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildHdrcLowResourceResultsTest(unittest.TestCase):
    def lineage_fixture(self):
        manifest = {
            "stage1_checkpoint": "logs/stage1/checkpoint.pth",
            "source_charset": "logs/stage1/charset.pkl",
            "head_protocol": {
                "num_classes": 6727,
                "batch_size": 2,
                "lr": 1e-4,
                "epochs": 3,
                "eval_epoch": 1,
                "max_iterations": 10000,
            },
            "full_protocol": {
                "num_classes": 6727,
                "batch_size": 2,
                "lr": 1e-5,
                "epochs": 12,
                "eval_epoch": 1,
                "max_iterations": 10000,
            },
        }
        common = {
            "new_class_embedding": True,
            "smart_mapping": True,
            "path_old_charset": manifest["source_charset"],
        }
        head = dict(
            common,
            resume=manifest["stage1_checkpoint"],
            resume_finetuning=False,
            **manifest["head_protocol"],
        )
        full = dict(
            common,
            resume="logs/head/checkpoint_best_regular.pth",
            resume_finetuning=True,
            **manifest["full_protocol"],
        )
        return manifest, head, full

    def test_plot_style_keeps_svg_text_editable(self):
        MODULE.configure_plot_style()
        self.assertEqual(MODULE.plt.rcParams["svg.fonttype"], "none")
        self.assertEqual(MODULE.plt.rcParams["pdf.fonttype"], 42)

    def test_low_resource_plot_uses_zero_baseline_and_sample_counts(self):
        rows = [
            {
                "fraction": fraction,
                "selected_train_samples": samples,
                "clean_cer": clean,
                "calibrated_cer": calibrated,
            }
            for fraction, samples, clean, calibrated in (
                (0.10, 2429, 0.16, 0.14),
                (0.25, 6071, 0.13, 0.11),
                (0.50, 12143, 0.10, 0.09),
                (1.00, 24285, 0.08, 0.07),
            )
        ]
        figure = MODULE.make_low_resource_figure(rows)
        axis = figure.axes[0]
        self.assertEqual(axis.get_ylim()[0], 0.0)
        self.assertAlmostEqual(figure.get_size_inches()[0], 6.3)
        self.assertEqual([line.get_linestyle() for line in axis.lines], ["-", "--"])
        self.assertNotEqual(axis.lines[0].get_color(), axis.lines[1].get_color())
        tick_text = " ".join(item.get_text() for item in axis.get_xticklabels())
        self.assertIn("10%\n(n=2,429)", tick_text)
        self.assertIn("100%\n(n=24,285)", tick_text)
        MODULE.plt.close(figure)

    def test_nested_fraction_indices_extend_anchor(self):
        ten = set(MODULE.select_fraction_indices(100, 0.10, 17, 0.10))
        twenty_five = set(MODULE.select_fraction_indices(100, 0.25, 17, 0.10))
        fifty = set(MODULE.select_fraction_indices(100, 0.50, 17, 0.10))
        self.assertLess(ten, twenty_five)
        self.assertLess(twenty_five, fifty)

    def test_protocol_requires_anchor_for_larger_fractions(self):
        condition = {"fraction": 0.25}
        config = {
            "seed": 42,
            "dataset_file": "mth1000",
            "mth1000_root": "hdrc_dtlr",
            "mth1000_train_fraction": 0.25,
            "mth1000_train_fraction_seed": 20260718,
        }
        with self.assertRaises(MODULE.IncompleteEvidence):
            MODULE.validate_run_protocol(
                condition, config, 42, 20260718, 0.10
            )
        config["mth1000_train_fraction_anchor"] = 0.10
        MODULE.validate_run_protocol(condition, config, 42, 20260718, 0.10)

    def test_anchor_and_unanchored_ten_percent_match(self):
        anchored = MODULE.select_fraction_indices(101, 0.10, 23, 0.10)
        unanchored = MODULE.select_fraction_indices(101, 0.10, 23, None)
        self.assertEqual(anchored, unanchored)
        samples = [{"id": f"line-{index}"} for index in range(101)]
        self.assertEqual(
            MODULE.selected_id_fingerprint(samples, anchored),
            MODULE.selected_id_fingerprint(samples, unanchored),
        )

    def test_full_data_run_does_not_require_fraction_metadata(self):
        MODULE.validate_run_protocol(
            {"fraction": 1.0},
            {"seed": 42, "dataset_file": "mth1000", "mth1000_root": "hdrc_dtlr"},
            42,
            20260718,
            0.10,
        )

    def test_training_lineage_accepts_matching_head_and_full_protocols(self):
        manifest, head, full = self.lineage_fixture()
        MODULE.validate_training_lineage(
            {"fraction": 0.25}, full, head, manifest
        )

    def test_training_lineage_rejects_wrong_stage1_source(self):
        manifest, head, full = self.lineage_fixture()
        head["resume"] = "logs/unrelated/checkpoint.pth"
        with self.assertRaises(MODULE.IncompleteEvidence):
            MODULE.validate_training_lineage(
                {"fraction": 0.25}, full, head, manifest
            )

    def test_training_lineage_rejects_changed_full_budget(self):
        manifest, head, full = self.lineage_fixture()
        full["epochs"] = 8
        with self.assertRaises(MODULE.IncompleteEvidence):
            MODULE.validate_training_lineage(
                {"fraction": 0.25}, full, head, manifest
            )


if __name__ == "__main__":
    unittest.main()
