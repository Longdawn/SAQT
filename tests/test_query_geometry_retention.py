import unittest

import torch

from engine import compute_query_geometry_retention_loss


class QueryGeometryRetentionLossTest(unittest.TestCase):
    def _outputs(self, boxes, logits=None):
        if logits is None:
            logits = torch.tensor([[[5.0], [4.0], [3.0]]])
        return {
            "pred_boxes": torch.tensor([boxes], dtype=torch.float32),
            "pred_logits": logits,
        }

    def test_identical_geometry_has_zero_loss(self):
        outputs = self._outputs(
            [[0.5, 0.1, 0.2, 0.1], [0.5, 0.4, 0.2, 0.1], [0.5, 0.8, 0.2, 0.1]]
        )
        loss, stats = compute_query_geometry_retention_loss(
            outputs,
            outputs,
            [{"labels": torch.tensor([1, 2])}],
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats["selected_queries"], 2.0)

    def test_vertical_drift_is_penalized(self):
        teacher = self._outputs(
            [[0.5, 0.1, 0.2, 0.1], [0.5, 0.4, 0.2, 0.1], [0.5, 0.8, 0.2, 0.1]]
        )
        student = self._outputs(
            [[0.5, 0.3, 0.2, 0.2], [0.5, 0.4, 0.2, 0.1], [0.5, 0.8, 0.2, 0.1]]
        )
        loss, _ = compute_query_geometry_retention_loss(
            student,
            teacher,
            [{"labels": torch.tensor([1])}],
        )
        self.assertGreater(loss.item(), 0.0)

    def test_horizontal_drift_is_not_penalized(self):
        teacher = self._outputs(
            [[0.2, 0.1, 0.1, 0.1], [0.5, 0.4, 0.2, 0.1], [0.5, 0.8, 0.2, 0.1]]
        )
        student = self._outputs(
            [[0.9, 0.1, 0.8, 0.1], [0.5, 0.4, 0.2, 0.1], [0.5, 0.8, 0.2, 0.1]]
        )
        loss, _ = compute_query_geometry_retention_loss(
            student,
            teacher,
            [{"labels": torch.tensor([1])}],
        )
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
