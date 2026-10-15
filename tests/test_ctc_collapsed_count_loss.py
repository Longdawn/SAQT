import unittest

import torch

from models.dino.dino import compute_ctc_collapsed_count_loss


class CTCCollapsedCountLossTest(unittest.TestCase):
    @staticmethod
    def one_hot(tokens, classes=4):
        return torch.nn.functional.one_hot(
            torch.tensor(tokens), num_classes=classes
        ).float().unsqueeze(0)

    def test_matches_greedy_ctc_collapse_for_one_hot_path(self):
        probabilities = self.one_hot([0, 1, 1, 0, 1, 2, 2])
        loss, expected = compute_ctc_collapsed_count_loss(
            probabilities, torch.tensor([3])
        )
        self.assertAlmostEqual(expected.item(), 3.0)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_adjacent_repeat_is_not_counted_twice(self):
        probabilities = self.one_hot([1, 1, 1])
        _, expected = compute_ctc_collapsed_count_loss(
            probabilities, torch.tensor([1])
        )
        self.assertAlmostEqual(expected.item(), 1.0)

    def test_blank_resets_repeated_character(self):
        probabilities = self.one_hot([1, 0, 1])
        _, expected = compute_ctc_collapsed_count_loss(
            probabilities, torch.tensor([2])
        )
        self.assertAlmostEqual(expected.item(), 2.0)

    def test_probability_path_is_differentiable(self):
        logits = torch.randn(2, 5, 4, requires_grad=True)
        probabilities = logits.softmax(dim=-1)
        loss, expected = compute_ctc_collapsed_count_loss(
            probabilities, torch.tensor([1, 3]), short_weight=3.0
        )
        loss.backward()
        self.assertEqual(tuple(expected.shape), (2,))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_rejects_invalid_probability_shape(self):
        with self.assertRaises(ValueError):
            compute_ctc_collapsed_count_loss(
                torch.ones(2, 4), torch.tensor([1, 1])
            )


if __name__ == "__main__":
    unittest.main()
