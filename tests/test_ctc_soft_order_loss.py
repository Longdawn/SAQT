import unittest

import torch

from models.dino.dino import (
    SetCriterion,
    compute_transcript_guided_soft_order_loss,
    soft_sort_permutation,
)


class CTCSoftOrderLossTest(unittest.TestCase):
    def test_soft_sort_approaches_hard_ascending_permutation(self):
        scores = torch.tensor([[0.8, 0.1, 0.5]])
        permutation = soft_sort_permutation(scores, temperature=1e-3)
        self.assertEqual(permutation.argmax(-1).tolist(), [[1, 2, 0]])
        self.assertTrue(torch.allclose(permutation.sum(-1), torch.ones(1, 3)))

    def test_ordered_queries_have_lower_loss_than_reversed_queries(self):
        probabilities = torch.tensor(
            [[[1e-4, 0.9998, 1e-4], [1e-4, 1e-4, 0.9998]]],
            dtype=torch.float32,
        )
        target = {"labels": torch.tensor([0, 1]), "direction": torch.tensor(1)}
        ordered_boxes = torch.tensor([[[0.5, 0.2, 0.1, 0.1], [0.5, 0.8, 0.1, 0.1]]])
        reversed_boxes = ordered_boxes.flip(1)
        ordered_loss, _ = compute_transcript_guided_soft_order_loss(
            probabilities, ordered_boxes, [target], temperature=0.01
        )
        reversed_loss, _ = compute_transcript_guided_soft_order_loss(
            probabilities, reversed_boxes, [target], temperature=0.01
        )
        self.assertLess(ordered_loss.item(), reversed_loss.item())

    def test_gradient_reaches_boxes_but_not_character_probabilities(self):
        logits = torch.randn(1, 5, 4, requires_grad=True)
        probabilities = logits.softmax(-1)
        boxes = torch.rand(1, 5, 4, requires_grad=True)
        target = {"labels": torch.tensor([0, 2]), "direction": torch.tensor(1)}
        loss, stats = compute_transcript_guided_soft_order_loss(
            probabilities, boxes, [target], temperature=0.05
        )
        loss.backward()
        self.assertIsNone(logits.grad)
        self.assertIsNotNone(boxes.grad)
        self.assertGreater(boxes.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(stats["permutation_entropy"]))

    def test_rejects_nonpositive_temperature(self):
        with self.assertRaises(ValueError):
            soft_sort_permutation(torch.ones(1, 3), temperature=0.0)

    def test_criterion_wires_soft_order_loss_without_changing_main_ctc(self):
        criterion = SetCriterion(
            num_classes=3,
            matcher=None,
            weight_dict={"loss_CTC": 1.0, "loss_ctc_soft_order": 0.01},
            focal_alpha=0.25,
            losses=[],
            CTC=True,
            ctc_soft_order_temperature=0.05,
        )
        boxes = torch.rand(1, 6, 4, requires_grad=True)
        outputs = {
            "pred_logits": torch.randn(1, 6, 3, requires_grad=True),
            "pred_boxes": boxes,
        }
        targets = [{"labels": torch.tensor([0, 1]), "direction": torch.tensor(1)}]
        losses = criterion.loss_CTC(outputs, targets, None, None)
        self.assertIn("loss_ctc_soft_order", losses)
        self.assertIn("loss_CTC", losses)
        losses["loss_ctc_soft_order"].backward()
        self.assertGreater(boxes.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
