import importlib
import unittest

import torch

from models.dino.dino import SetCriterion, apply_length_adaptive_ctc_bias


class CtcLengthCalibrationTest(unittest.TestCase):
    def test_zero_scale_is_identity(self):
        probabilities = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]])
        adjusted, bias = apply_length_adaptive_ctc_bias(
            probabilities,
            torch.tensor([1.0]),
            bias_scale=0.0,
        )
        torch.testing.assert_close(adjusted, probabilities)
        torch.testing.assert_close(bias, torch.zeros_like(bias))

    def test_longer_prediction_increases_nonblank_mass(self):
        probabilities = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]])
        current_nonblank = probabilities[:, :, 1:].sum()
        adjusted, bias = apply_length_adaptive_ctc_bias(
            probabilities,
            torch.log1p(torch.tensor([3.0])),
            bias_scale=1.0,
            max_bias=1.0,
        )
        self.assertGreater(float(bias.item()), 0.0)
        self.assertGreater(float(adjusted[:, :, 1:].sum()), float(current_nonblank))
        torch.testing.assert_close(
            adjusted.sum(dim=-1),
            torch.ones_like(adjusted[:, :, 0]),
        )

    def test_shorter_prediction_increases_blank_mass(self):
        probabilities = torch.tensor([[[0.4, 0.6], [0.3, 0.7]]])
        adjusted, bias = apply_length_adaptive_ctc_bias(
            probabilities,
            torch.log1p(torch.tensor([0.25])),
            bias_scale=1.0,
            max_bias=1.0,
        )
        self.assertLess(float(bias.item()), 0.0)
        self.assertGreater(
            float(adjusted[:, :, 0].sum()),
            float(probabilities[:, :, 0].sum()),
        )

    def test_bias_is_bounded(self):
        probabilities = torch.tensor([[[0.9, 0.1]]])
        _, bias = apply_length_adaptive_ctc_bias(
            probabilities,
            torch.tensor([100.0]),
            bias_scale=10.0,
            max_bias=0.2,
        )
        torch.testing.assert_close(bias, torch.tensor([0.2]))

    def test_probe_config_is_isolated(self):
        config = importlib.import_module("config.MTHV2_dtlr_qclc_probe")
        baseline = importlib.import_module("config.MTHV2_dtlr")
        self.assertTrue(config.use_ctc_length_calibration)
        self.assertFalse(hasattr(baseline, "use_ctc_length_calibration"))
        self.assertEqual(config.ctc_length_loss_coef, 0.01)

    def test_ctc_criterion_reports_length_loss_and_stats(self):
        criterion = SetCriterion(
            num_classes=2,
            matcher=None,
            weight_dict={"loss_CTC": 1.0, "loss_ctc_length": 0.01},
            focal_alpha=0.25,
            losses=[],
            CTC=True,
            use_ctc_length_calibration=True,
            ctc_length_bias_scale=0.25,
            ctc_length_bias_max=0.5,
        )
        outputs = {
            "pred_logits": torch.tensor([[[3.0, -3.0], [-3.0, 3.0]]]),
            "pred_boxes": torch.tensor(
                [[[0.5, 0.1, 0.1, 0.1], [0.5, 0.9, 0.1, 0.1]]]
            ),
            "pred_log_length": torch.log1p(torch.tensor([1.0])),
        }
        targets = [{"labels": torch.tensor([0]), "direction": torch.tensor(1)}]
        losses, probabilities, _ = criterion.loss_CTC(
            outputs,
            targets,
            indices=None,
            num_boxes=None,
            return_preds=True,
        )
        self.assertIn("loss_ctc_length", losses)
        self.assertIn("ctc_predicted_length", losses)
        self.assertIn("ctc_adaptive_bias", losses)
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[:, :, 0]),
        )


if __name__ == "__main__":
    unittest.main()
