import copy
from bisect import bisect_right

from torch.utils.data import Dataset

from .MTH1000 import MTH1000, make_coco_transforms


class MTHCombo(Dataset):
    """Concatenate multiple MTH-style processed roots with one shared charset."""

    def __init__(self, mode, transform, args=None):
        roots = getattr(args, "mth_combo_roots", None)
        raw_roots = getattr(args, "mth_combo_raw_roots", None)
        if not roots or not raw_roots:
            raise ValueError("mth_combo_roots and mth_combo_raw_roots must be set")
        if len(roots) != len(raw_roots):
            raise ValueError("mth_combo_roots and mth_combo_raw_roots must have equal length")

        self.datasets = []
        base_fraction_seed = int(getattr(args, "mth1000_train_fraction_seed", 42))
        for dataset_index, (root, raw_root) in enumerate(zip(roots, raw_roots)):
            child_args = copy.copy(args)
            child_args.mth1000_root = root
            child_args.mth1000_raw_root = raw_root
            child_args.mth1000_train_fraction_seed = base_fraction_seed + dataset_index
            self.datasets.append(MTH1000(mode, transform, args=child_args))

        self.samples = []
        for dataset in self.datasets:
            self.samples.extend(getattr(dataset, "samples", []))

        charset = set()
        for dataset in self.datasets:
            charset.update(dataset.charset)
        self.charset = sorted(charset)
        for dataset in self.datasets:
            dataset.charset = self.charset

        fraction_manifests = [
            dataset.train_fraction_manifest
            for dataset in self.datasets
            if getattr(dataset, "train_fraction_manifest", None) is not None
        ]
        self.train_fraction_manifest = (
            {"strategy": "per_child_deterministic", "children": fraction_manifests}
            if fraction_manifests
            else None
        )

        self.cumulative_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

        names = ", ".join(f"{root}:{len(dataset)}" for root, dataset in zip(roots, self.datasets))
        print(
            f"[MTHCombo] split={mode} datasets={names} total={len(self)} charset={len(self.charset)}"
        )

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx
        if dataset_idx > 0:
            sample_idx -= self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


def build_mth_combo(image_set, args):
    transforms = make_coco_transforms(image_set, args=args)
    return MTHCombo(image_set, transforms, args=args)
