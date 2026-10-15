import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs/paper-drafts/tools/integrate_final_experiment_results.py"
SPEC = importlib.util.spec_from_file_location("integrate_final_experiment_results", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
integrate_text = MODULE.integrate_text
integrate_abstract_text = MODULE.integrate_abstract_text
integrate_conclusion_text = MODULE.integrate_conclusion_text
integrate_qclc_method_text = MODULE.integrate_qclc_method_text
remove_qclc_method_text = MODULE.remove_qclc_method_text
remove_generated_block = MODULE.remove_generated_block
render = MODULE.render
render_abstract_evidence = MODULE.render_abstract_evidence
render_conclusion_evidence = MODULE.render_conclusion_evidence
render_qclc = MODULE.render_qclc
render_qclc_method = MODULE.render_qclc_method
validate_qclc = MODULE.validate_qclc
validate = MODULE.validate


def seed(seed_value, offset=0.0):
    return {
        "seed": str(seed_value),
        "clean_cer": str(0.04 + offset),
        "clean_ar": str(0.96 - offset),
        "clean_cr": str(0.965 - offset),
        "calibrated_cer": str(0.035 + offset),
        "calibrated_ar": str(0.965 - offset),
        "calibrated_cr": str(0.97 - offset),
        "calibrated_empty": str(0.01 + offset),
        "calibrated_pred_gt_ratio": str(0.995 - offset),
    }


def fraction(value, cer):
    return {
        "fraction": str(value),
        "seed": "42",
        "selected_train_samples": str(int(24285 * value)),
        "max_train_samples_per_epoch": str(min(int(24285 * value), 10000)),
        "clean_cer": str(cer + 0.01),
        "clean_ar": str(0.99 - cer),
        "clean_cr": str(0.995 - cer),
        "blank_bias": "-2.0",
        "nonblank_bias": "1.0",
        "calibrated_cer": str(cer),
        "calibrated_ar": str(1.0 - cer),
        "calibrated_cr": str(1.0 - cer / 2),
        "calibrated_empty": "0.01",
        "calibrated_pred_gt_ratio": "0.99",
    }


def qclc(variant, offset=0.0):
    row = {
        "variant": variant,
        "validation_cer": str(0.11 + offset),
        "clean_cer": str(0.04 + offset),
        "clean_ar": str(0.96 - offset),
        "clean_cr": str(0.965 - offset),
        "blank_bias": "-2.0",
        "nonblank_bias": "0.8",
        "calibrated_cer": str(0.035 + offset),
        "calibrated_ar": str(0.965 - offset),
        "calibrated_cr": str(0.97 - offset),
        "calibrated_empty": str(0.01 + offset),
        "calibrated_pred_gt_ratio": str(0.995 - offset),
    }
    for bucket in ("1", "2", "3-5", "6-10", "11+"):
        row[f"calibrated_len_{bucket}_cer"] = str(0.1 + offset)
    return row


class IntegrateFinalExperimentResultsTest(unittest.TestCase):
    def test_low_resource_rows_require_verified_per_epoch_cap(self):
        seed_rows = [seed(42), seed(43), seed(44)]
        low_rows = [
            fraction(0.1, 0.12),
            fraction(0.25, 0.10),
            fraction(0.5, 0.08),
            fraction(1.0, 0.06),
        ]
        validate(seed_rows, low_rows)
        low_rows[2]["max_train_samples_per_epoch"] = "12143"
        with self.assertRaises(MODULE.IncompleteEvidence):
            validate(seed_rows, low_rows)

    def test_render_and_integrate_are_idempotent(self):
        generated = render(
            [seed(42), seed(43, 0.001), seed(44, -0.001)],
            [fraction(0.1, 0.12), fraction(0.25, 0.10), fraction(0.5, 0.08), fraction(1.0, 0.06)],
        )
        source = "# Results\n\n## 5.8 局限性\n\nText.\n"
        once = integrate_text(source, generated)
        twice = integrate_text(once, generated)
        self.assertEqual(once, twice)
        self.assertIn("## 5.8 重复实验", once)
        self.assertIn("## 5.9 目标域低标注迁移", once)
        self.assertIn("## 5.10 局限性", once)
        self.assertIn("各条件单次训练，seed=42，无误差棒", once)
        self.assertIn("不估计跨次不确定性", once)
        self.assertIn("| Target labels | Label-pool lines | Max lines/epoch |", once)
        self.assertIn("| 100% | 24285 | 10000 |", once)
        self.assertEqual(once.count("GENERATED_FINAL_EXPERIMENTS_START"), 1)

    def test_obsolete_generated_prose_block_is_removed_idempotently(self):
        source = (
            "## 摘要\n\nText.\n\n"
            f"{MODULE.ABSTRACT_ZH_START}\n\nObsolete.\n\n"
            f"{MODULE.ABSTRACT_ZH_END}\n\n**关键词：** x\n"
        )
        once = remove_generated_block(
            source, MODULE.ABSTRACT_ZH_START, MODULE.ABSTRACT_ZH_END
        )
        twice = remove_generated_block(
            once, MODULE.ABSTRACT_ZH_START, MODULE.ABSTRACT_ZH_END
        )
        self.assertEqual(once, twice)
        self.assertNotIn("Obsolete", once)
        self.assertIn("**关键词：** x", once)

    def test_qclc_render_and_dynamic_limitations_heading(self):
        qclc_rows = validate_qclc(
            [qclc("qclc", -0.002), qclc("matched_baseline")]
        )
        self.assertEqual(qclc_rows[0]["variant"], "matched_baseline")
        generated = render(
            [seed(42), seed(43, 0.001), seed(44, -0.001)],
            [fraction(0.1, 0.12), fraction(0.25, 0.10), fraction(0.5, 0.08), fraction(1.0, 0.06)],
            qclc_rows,
        )
        source = "# Results\n\n## 5.8 局限性\n\nText.\n"
        once = integrate_text(source, generated, has_qclc=True)
        twice = integrate_text(once, generated, has_qclc=True)
        self.assertEqual(once, twice)
        self.assertIn("## 5.10 查询条件长度校准消融", once)
        self.assertIn("## 5.11 局限性", once)
        self.assertNotIn("## 5.10 局限性", once)
        self.assertIn("不使用测试集决定是否保留该行", render_qclc(qclc_rows))

        method = "# Method\n\nExisting text.\n"
        method_once = integrate_qclc_method_text(method, render_qclc_method())
        method_twice = integrate_qclc_method_text(method_once, render_qclc_method())
        self.assertEqual(method_once, method_twice)
        self.assertEqual(method_once.count("GENERATED_QCLC_METHOD_START"), 1)
        self.assertIn("## 3.5 查询条件长度校准", method_once)
        self.assertIn("不接收真实字符框、测试标签或外部文本", method_once)
        self.assertIn("在 CTC 训练与推理时", method_once)
        self.assertIn("\\(\\gamma=0.25\\)", method_once)
        self.assertNotIn("$\\gamma", method_once)

        cleaned = remove_qclc_method_text(method_once)
        self.assertNotIn("QCLC", cleaned)
        self.assertEqual(remove_qclc_method_text(cleaned), cleaned)


if __name__ == "__main__":
    unittest.main()
