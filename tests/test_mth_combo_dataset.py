import types
import unittest
from unittest import mock

from datasets import MTHCombo as mth_combo_module
from datasets.MTH1000 import sample_id_fingerprint, select_train_fraction_indices


class _FakeMTH1000:
    def __init__(self, mode, transform, args=None):
        self.root = args.mth1000_root
        self.samples = [
            {"id": f"{self.root}-0", "text": self.root[:1]},
            {"id": f"{self.root}-1", "text": self.root[:2]},
        ]
        self.charset = sorted(set("".join(sample["text"] for sample in self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MTHComboDatasetTest(unittest.TestCase):
    def test_exposes_flattened_samples_without_changing_index_routing(self):
        args = types.SimpleNamespace(
            mth_combo_roots=["alpha", "beta"],
            mth_combo_raw_roots=["raw_alpha", "raw_beta"],
        )

        with mock.patch.object(mth_combo_module, "MTH1000", _FakeMTH1000):
            dataset = mth_combo_module.MTHCombo("train", transform=None, args=args)

        self.assertEqual(len(dataset), 4)
        self.assertEqual([sample["id"] for sample in dataset.samples], [
            "alpha-0",
            "alpha-1",
            "beta-0",
            "beta-1",
        ])
        self.assertEqual(dataset[0]["id"], "alpha-0")
        self.assertEqual(dataset[2]["id"], "beta-0")
        self.assertEqual(dataset[3]["id"], "beta-1")

    def test_fraction_indices_are_stable_and_source_ordered(self):
        first = select_train_fraction_indices(20, 0.25, 17)
        second = select_train_fraction_indices(20, 0.25, 17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(first, sorted(first))
        self.assertNotEqual(first, select_train_fraction_indices(20, 0.25, 18))

    def test_fraction_validation_and_fingerprint(self):
        with self.assertRaises(ValueError):
            select_train_fraction_indices(10, 0.0, 42)
        with self.assertRaises(ValueError):
            select_train_fraction_indices(10, 1.1, 42)

        samples = [{"id": "a"}, {"id": "b"}]
        self.assertEqual(sample_id_fingerprint(samples), sample_id_fingerprint(samples))
        self.assertNotEqual(
            sample_id_fingerprint(samples), sample_id_fingerprint(list(reversed(samples)))
        )

    def test_anchor_fraction_preserves_existing_subset_and_nests_larger_subsets(self):
        num_samples = 101
        seed = 20260718
        original_ten = select_train_fraction_indices(num_samples, 0.10, seed)
        anchored_ten = select_train_fraction_indices(num_samples, 0.10, seed, 0.10)
        anchored_twenty_five = select_train_fraction_indices(
            num_samples, 0.25, seed, 0.10
        )
        anchored_fifty = select_train_fraction_indices(num_samples, 0.50, seed, 0.10)

        self.assertEqual(original_ten, anchored_ten)
        self.assertTrue(set(anchored_ten) < set(anchored_twenty_five))
        self.assertTrue(set(anchored_twenty_five) < set(anchored_fifty))
        self.assertEqual(anchored_twenty_five, sorted(anchored_twenty_five))

        with self.assertRaises(ValueError):
            select_train_fraction_indices(num_samples, 0.25, seed, 0.0)


if __name__ == "__main__":
    unittest.main()
