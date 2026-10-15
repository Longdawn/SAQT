if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(sys.path[0]))

import json
import hashlib
import os
import pickle
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets.transforms as T

current_dir = os.path.dirname(os.path.abspath(__file__))

if "dataset" not in current_dir:
    current_dir = os.path.join(current_dir, "dataset")
else:
    current_dir = os.path.join(current_dir, ".")

with open(os.path.join(current_dir, "config.json"), "r") as f:
    datasets_path = json.load(f)

datasets_path = datasets_path["datasets_path"]


def select_train_fraction_indices(num_samples, fraction, seed, anchor_fraction=None):
    """Return a stable subset, optionally extending a fixed anchor subset."""
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"mth1000_train_fraction must be in (0, 1], got {fraction}")
    if num_samples <= 0 or fraction == 1.0:
        return list(range(num_samples))

    subset_size = max(1, int(num_samples * fraction + 0.5))
    generator = random.Random(int(seed))
    if anchor_fraction is None:
        return sorted(generator.sample(range(num_samples), subset_size))

    anchor_fraction = float(anchor_fraction)
    if not 0.0 < anchor_fraction <= 1.0:
        raise ValueError(
            "mth1000_train_fraction_anchor must be in (0, 1], "
            f"got {anchor_fraction}"
        )
    anchor_size = max(1, int(num_samples * anchor_fraction + 0.5))
    if subset_size <= anchor_size:
        return sorted(generator.sample(range(num_samples), subset_size))

    # Preserve the already established anchor subset exactly, then use one
    # deterministic ordering of its complement for all larger fractions.
    anchor_indices = generator.sample(range(num_samples), anchor_size)
    anchor_set = set(anchor_indices)
    remaining = [index for index in range(num_samples) if index not in anchor_set]
    extension_generator = random.Random(int(seed) + 1)
    extension_generator.shuffle(remaining)
    selected = anchor_indices + remaining[: subset_size - anchor_size]
    return sorted(selected)


def sample_id_fingerprint(samples):
    identities = [str(sample.get("id", index)) for index, sample in enumerate(samples)]
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()


class MTH1000(Dataset):
    def __init__(self, mode, transform=transforms.ToTensor(), target_transform=None, args=None):
        if mode == "val":
            mode = "valid"

        self.mode = mode
        self._transforms = transform
        self.target_transform = target_transform

        root_name = getattr(args, "mth1000_root", "mth1000_dtlr")
        labels_name = getattr(args, "mth1000_labels_file", "labels.pkl")
        images_subdir = getattr(args, "mth1000_images_subdir", "lines")
        self.im_extension = getattr(args, "mth1000_image_ext", "jpg")

        self.root_path = os.path.join(datasets_path, root_name)
        labels_path = os.path.join(self.root_path, labels_name)
        self.images_path = os.path.join(self.root_path, images_subdir)
        self.use_char_boxes = getattr(args, "mth1000_use_char_boxes", False)
        raw_root_name = getattr(args, "mth1000_raw_root", "MTH1000")
        self.raw_root_path = os.path.join(datasets_path, raw_root_name)
        self.textline_path = os.path.join(self.raw_root_path, "label_textline")
        self.char_path = os.path.join(self.raw_root_path, "label_char")
        self._textline_cache = {}
        self._char_cache = {}
        self._char_box_stats = defaultdict(int)
        self._char_box_fallback_examples = []

        self.data = pickle.load(open(labels_path, "rb"))

        # Build charset from all splits so new_class_embedding can resize properly.
        charset = set()
        for split in ["train", "valid", "test"]:
            for sample in self.data["ground_truth"].get(split, []):
                charset.update(list(sample["text"]))
        self.charset = sorted(list(charset))
        self.direction_source = getattr(args, "direction_source", "label")
        self.forced_direction = getattr(args, "forced_direction", "auto")
        self.filter_direction = getattr(args, "mth1000_filter_direction", "all")
        self.samples = self.data["ground_truth"][self.mode]
        if self.filter_direction in ["vertical", "horizontal"]:
            target_dir = 1 if self.filter_direction == "vertical" else 0
            filtered_samples = []
            for sample in self.samples:
                dir_label = self._infer_direction(sample, image=None)
                if dir_label == target_dir:
                    filtered_samples.append(sample)
            self.samples = filtered_samples
            print(
                f"[MTH1000] split={self.mode} filter_direction={self.filter_direction} "
                f"{len(filtered_samples)}/{len(self.data['ground_truth'][self.mode])}"
            )

        self.train_fraction_manifest = None
        if self.mode == "train":
            train_fraction = float(getattr(args, "mth1000_train_fraction", 1.0))
            subset_seed = int(getattr(args, "mth1000_train_fraction_seed", 42))
            anchor_fraction = getattr(args, "mth1000_train_fraction_anchor", None)
            source_size = len(self.samples)
            selected_indices = select_train_fraction_indices(
                source_size, train_fraction, subset_seed, anchor_fraction
            )
            if len(selected_indices) != source_size:
                self.samples = [self.samples[index] for index in selected_indices]
            self.train_fraction_manifest = {
                "root": root_name,
                "fraction": train_fraction,
                "seed": subset_seed,
                "anchor_fraction": (
                    float(anchor_fraction) if anchor_fraction is not None else None
                ),
                "source_size": source_size,
                "selected_size": len(self.samples),
                "selected_id_sha256": sample_id_fingerprint(self.samples),
            }
            if train_fraction < 1.0:
                print(
                    f"[MTH1000] split=train fraction={train_fraction:.4f} "
                    f"seed={subset_seed} selected={len(self.samples)}/{source_size} "
                    f"sha256={self.train_fraction_manifest['selected_id_sha256'][:12]}"
                )

        if self.use_char_boxes:
            has_raw_annotations = os.path.isdir(self.textline_path) and os.path.isdir(self.char_path)
            self.use_char_boxes = has_raw_annotations
            if self.use_char_boxes:
                print(
                    f"[MTH1000] split={self.mode} using raw char boxes from {self.raw_root_path}"
                )
            else:
                print(
                    f"[MTH1000] split={self.mode} raw char boxes unavailable under {self.raw_root_path}; "
                    "falling back to dummy boxes"
                )

    def __len__(self):
        return len(self.samples)

    def convert_str_to_tensor(self, text):
        labels = [self.charset.index(c) for c in list(text)]
        return torch.tensor(labels, dtype=torch.int64)

    def _infer_direction(self, sample, image=None):
        direction = sample.get("direction", None)
        if isinstance(direction, str):
            direction = direction.strip().lower()
            if direction in ["vertical", "v", "1"]:
                return 1
            if direction in ["horizontal", "h", "0"]:
                return 0
        if isinstance(direction, bool):
            return int(direction)
        if isinstance(direction, int):
            return 1 if direction == 1 else 0

        # Keep the old geometry heuristic as a fallback when labels do not provide direction.
        if self.direction_source in ["label", "rule"]:
            if image is not None:
                return 1 if image.size[1] > image.size[0] else 0
            im_id = sample.get("id")
            if im_id is not None:
                path_image = os.path.join(self.images_path, f"{im_id}.{self.im_extension}")
                if os.path.exists(path_image):
                    with Image.open(path_image) as im:
                        return 1 if im.size[1] > im.size[0] else 0
        return 0

    def _normalize_direction(self, sample, image=None):
        if isinstance(self.forced_direction, str):
            forced = self.forced_direction.strip().lower()
            if forced in ["vertical", "v", "1"]:
                return 1
            if forced in ["horizontal", "h", "0"]:
                return 0

        return self._infer_direction(sample, image=image)

    @staticmethod
    def _parse_line_sample_id(sample_id):
        if "_" not in sample_id:
            return None, None
        page_id, line_idx = sample_id.rsplit("_", 1)
        if not line_idx.isdigit():
            return None, None
        return page_id, int(line_idx)

    def _load_textlines(self, page_id):
        if page_id not in self._textline_cache:
            path = os.path.join(self.textline_path, f"{page_id}.txt")
            if not os.path.exists(path):
                self._textline_cache[page_id] = None
            else:
                with open(path, "r", encoding="utf-8") as f:
                    self._textline_cache[page_id] = f.read().splitlines()
        return self._textline_cache[page_id]

    def _load_char_boxes(self, page_id):
        if page_id not in self._char_cache:
            path = os.path.join(self.char_path, f"{page_id}.txt")
            if not os.path.exists(path):
                self._char_cache[page_id] = None
            else:
                entries = []
                with open(path, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        parts = raw_line.strip().split()
                        if len(parts) < 5:
                            continue
                        ch = parts[0]
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        entries.append(
                            {
                                "char": ch,
                                "box": [x1, y1, x2, y2],
                                "cx": (x1 + x2) / 2.0,
                                "cy": (y1 + y2) / 2.0,
                            }
                        )
                self._char_cache[page_id] = entries if entries else None
        return self._char_cache[page_id]

    def _maybe_real_char_boxes(self, sample, image, direction):
        if not self.use_char_boxes:
            return None

        sample_id = sample.get("id")
        page_id, line_idx = self._parse_line_sample_id(sample_id)
        if page_id is None:
            self._char_box_stats["bad_sample_id"] += 1
            return None

        textline_lines = self._load_textlines(page_id)
        char_entries = self._load_char_boxes(page_id)
        if textline_lines is None or char_entries is None or line_idx >= len(textline_lines):
            self._char_box_stats["missing_raw_annotation"] += 1
            return None

        parts = textline_lines[line_idx].split(",")
        if len(parts) < 9:
            self._char_box_stats["bad_textline_format"] += 1
            return None

        raw_text = parts[0]
        if raw_text != sample["text"]:
            self._char_box_stats["text_mismatch"] += 1
            if len(self._char_box_fallback_examples) < 5:
                self._char_box_fallback_examples.append((sample_id, "text_mismatch", sample["text"], raw_text))
            return None

        coords = list(map(float, parts[1:9]))
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        selected = [
            entry
            for entry in char_entries
            if xmin <= entry["cx"] <= xmax and ymin <= entry["cy"] <= ymax
        ]
        if not selected:
            self._char_box_stats["empty_selection"] += 1
            return None

        if direction == 1:
            selected.sort(key=lambda entry: entry["cy"])
        else:
            selected.sort(key=lambda entry: entry["cx"])

        selected_text = "".join(entry["char"] for entry in selected)
        if selected_text != sample["text"]:
            self._char_box_stats["sequence_mismatch"] += 1
            if len(self._char_box_fallback_examples) < 5:
                self._char_box_fallback_examples.append(
                    (sample_id, "sequence_mismatch", sample["text"], selected_text)
                )
            return None

        width, height = image.size
        boxes = []
        for entry in selected:
            x1, y1, x2, y2 = entry["box"]
            x1 -= xmin
            x2 -= xmin
            y1 -= ymin
            y2 -= ymin
            x1 = min(max(x1, 0.0), float(width))
            x2 = min(max(x2, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            y2 = min(max(y2, 0.0), float(height))
            boxes.append([x1, y1, x2, y2])

        self._char_box_stats["used_real_boxes"] += 1
        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        example = self.samples[idx]
        text = example["text"]
        im_id = example["id"]

        path_image = os.path.join(self.images_path, f"{im_id}.{self.im_extension}")
        image = Image.open(path_image).convert("RGB")

        labels = {}
        labels["labels"] = self.convert_str_to_tensor(text)
        labels["orig_size"] = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        labels["size"] = torch.tensor([image.size[1], image.size[0]], dtype=torch.int64)
        labels["img_idx"] = torch.tensor([idx], dtype=torch.int64)
        labels["idx"] = torch.tensor([idx], dtype=torch.int64)
        direction = self._normalize_direction(example, image)
        labels["direction"] = torch.tensor([direction], dtype=torch.int64)

        labels["boxes"] = self._maybe_real_char_boxes(example, image, direction)
        if labels["boxes"] is None:
            dummy_boxes = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            labels["boxes"] = dummy_boxes.repeat(labels["labels"].shape[0], 1)
            self._char_box_stats["used_dummy_boxes"] += 1

        image, labels = self._transforms(image, labels)
        return image, labels


class VerticalMSRResize:
    """Aspect-ratio bucketed resize for vertical text-line crops.

    The existing DTLR resize maps the short side to a large fixed value. That is
    fine for long vertical columns because max_size clamps the height, but it
    over-enlarges near-square one/two-character crops. This transform keeps long
    columns close to the original scale policy while using smaller short-side
    sizes for short crops.
    """

    def __init__(self, buckets, max_size=1333, random_choice=False):
        self.buckets = buckets
        self.max_size = max_size
        self.random_choice = random_choice

    @staticmethod
    def _normalize_bucket(bucket):
        if len(bucket) == 2:
            max_ratio, sizes = bucket
            return max_ratio, sizes, None
        if len(bucket) == 3:
            return bucket
        raise ValueError(f"MSR bucket must be (max_ratio, sizes) or (max_ratio, sizes, max_size), got {bucket}")

    def _choose_size(self, sizes):
        if isinstance(sizes, (list, tuple)):
            if self.random_choice:
                return random.choice(list(sizes))
            return max(sizes)
        return int(sizes)

    def _pick_resize(self, ratio):
        for bucket in self.buckets:
            max_ratio, sizes, bucket_max_size = self._normalize_bucket(bucket)
            if ratio <= float(max_ratio):
                return self._choose_size(sizes), bucket_max_size or self.max_size
        _, sizes, bucket_max_size = self._normalize_bucket(self.buckets[-1])
        return self._choose_size(sizes), bucket_max_size or self.max_size

    def __call__(self, img, target=None):
        w, h = img.size
        ratio = float(h) / max(float(w), 1.0)
        size, max_size = self._pick_resize(ratio)
        return T.resize(img, target, size, max_size)


class LengthAwareVerticalMSRResize(VerticalMSRResize):
    """Vertical MSR with optional GT-length rules for diagnosis.

    This is not deployment-safe when rules depend on labels, but it is useful to
    estimate whether length-aware scaling can fix short-line failure modes.
    Rules use the tuple format:
      (min_len, max_len, min_ratio_exclusive, max_ratio_inclusive, sizes)
    """

    def __init__(self, buckets, length_rules, max_size=1333, random_choice=False):
        super().__init__(buckets, max_size=max_size, random_choice=random_choice)
        self.length_rules = length_rules

    def _choose(self, sizes):
        if isinstance(sizes, (list, tuple)):
            return random.choice(list(sizes)) if self.random_choice else max(sizes)
        return int(sizes)

    def __call__(self, img, target=None):
        w, h = img.size
        ratio = float(h) / max(float(w), 1.0)
        gt_len = None
        if target is not None and "labels" in target:
            gt_len = int(target["labels"].shape[0])

        if gt_len is not None:
            for min_len, max_len, min_ratio, max_ratio, sizes in self.length_rules:
                if int(min_len) <= gt_len <= int(max_len) and float(min_ratio) < ratio <= float(max_ratio):
                    return T.resize(img, target, self._choose(sizes), self.max_size)

        size, max_size = self._pick_resize(ratio)
        return T.resize(img, target, size, max_size)


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333

    scales = getattr(args, "data_aug_scales", scales)
    max_size = getattr(args, "data_aug_max_size", max_size)
    use_msr = getattr(args, "mth1000_use_msr", False)
    use_length_msr = getattr(args, "mth1000_use_length_msr", False)
    train_msr_buckets = getattr(
        args,
        "mth1000_msr_train_buckets",
        [
            (1.5, [256, 320, 384]),
            (3.0, [384, 448, 512]),
            (6.0, [544, 608, 672]),
            (1.0e9, [704, 768, 800]),
        ],
    )
    eval_msr_buckets = getattr(
        args,
        "mth1000_msr_eval_buckets",
        [
            (1.5, 320),
            (3.0, 480),
            (6.0, 640),
            (1.0e9, 800),
        ],
    )
    train_length_msr_rules = getattr(args, "mth1000_train_length_msr_rules", [])
    eval_length_msr_rules = getattr(args, "mth1000_eval_length_msr_rules", [])

    if image_set == "train":
        if use_msr:
            if use_length_msr:
                return T.Compose([
                    LengthAwareVerticalMSRResize(
                        train_msr_buckets,
                        train_length_msr_rules,
                        max_size=max_size,
                        random_choice=True,
                    ),
                    normalize,
                ])
            return T.Compose([
                VerticalMSRResize(train_msr_buckets, max_size=max_size, random_choice=True),
                normalize,
            ])

        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([T.RandomResize(scales, max_size=max_size)]),
            ),
            normalize,
        ])

    if image_set in ["val", "eval_debug", "train_reg", "test", "valid"]:
        if use_msr:
            if use_length_msr:
                return T.Compose([
                    LengthAwareVerticalMSRResize(
                        eval_msr_buckets,
                        eval_length_msr_rules,
                        max_size=max_size,
                        random_choice=False,
                    ),
                    normalize,
                ])
            return T.Compose([
                VerticalMSRResize(eval_msr_buckets, max_size=max_size, random_choice=False),
                normalize,
            ])

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f"unknown {image_set}")


def build_mth1000(image_set, args):
    transforms = make_coco_transforms(image_set, args=args)
    return MTH1000(image_set, transforms, args=args)
