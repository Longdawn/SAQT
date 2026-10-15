import argparse
import json
import pickle
from pathlib import Path


SPLIT_TO_FILE = {
    "train": "textrecog_train.json",
    "valid": "textrecog_val.json",
    "test": "textrecog_test.json",
}


PRESETS = {
    "mthv2": [
        "tkhmth2200_mth1000_dtlr",
        "tkhmth2200_mth1200_dtlr",
        "tkhmth2200_tkh_dtlr",
    ],
    "hdrc": [
        "hdrc_dtlr",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        "Export DTLR labels.pkl recognition datasets to MMOCR OCRDataset JSON"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root that contains the processed DTLR dataset directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/mmocr_exports"),
        help="Directory where exported MMOCR annotation folders are written.",
    )
    parser.add_argument(
        "--preset",
        choices=["mthv2", "hdrc", "all"],
        default="all",
        help="Dataset preset to export.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Custom export name. Must be used together with --roots.",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help="Custom processed dataset roots under --data-root.",
    )
    parser.add_argument(
        "--labels-file",
        default="labels.pkl",
        help="Label pickle name inside each processed dataset root.",
    )
    parser.add_argument(
        "--images-subdir",
        default="lines",
        help="Image subdirectory inside each processed dataset root.",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        help="Expected line image extension.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip samples whose image file is missing instead of failing.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="JSON indentation. Omit for compact files.",
    )
    return parser.parse_args()


def load_labels(path):
    with path.open("rb") as handle:
        labels = pickle.load(handle)
    if "ground_truth" not in labels:
        raise KeyError(f"{path} does not contain a 'ground_truth' key")
    return labels["ground_truth"]


def mmocr_record(data_root, image_path, text):
    rel_image_path = image_path.relative_to(data_root).as_posix()
    return {
        "img_path": rel_image_path,
        "instances": [
            {
                "text": text,
            }
        ],
    }


def export_dataset(name, roots, args):
    data_root = args.data_root.resolve()
    output_dir = args.output_root.resolve() / name
    output_dir.mkdir(parents=True, exist_ok=True)

    split_records = {split: [] for split in SPLIT_TO_FILE}
    split_sources = {split: {} for split in SPLIT_TO_FILE}
    charset = set()
    missing = []

    for root_name in roots:
        root = data_root / root_name
        labels_path = root / args.labels_file
        images_dir = root / args.images_subdir
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {images_dir}")

        ground_truth = load_labels(labels_path)
        for split in SPLIT_TO_FILE:
            samples = ground_truth.get(split, [])
            kept = 0
            for sample in samples:
                sample_id = sample["id"]
                text = sample["text"]
                image_path = images_dir / f"{sample_id}.{args.image_ext}"
                if not image_path.is_file():
                    missing.append(image_path.as_posix())
                    if args.skip_missing_images:
                        continue
                    raise FileNotFoundError(f"Missing image: {image_path}")
                split_records[split].append(mmocr_record(data_root, image_path, text))
                charset.update(text)
                kept += 1
            split_sources[split][root_name] = kept

    metainfo = {
        "dataset_type": "TextRecogDataset",
        "task_name": "textrecog",
    }
    for split, filename in SPLIT_TO_FILE.items():
        payload = {
            "metainfo": metainfo,
            "data_list": split_records[split],
        }
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=args.indent)
            handle.write("\n")

    all_records = [
        record
        for split in SPLIT_TO_FILE
        for record in split_records[split]
    ]
    all_filename = "textrecog_all.json"
    with (output_dir / all_filename).open("w", encoding="utf-8") as handle:
        json.dump(
            {"metainfo": metainfo, "data_list": all_records},
            handle,
            ensure_ascii=False,
            indent=args.indent,
        )
        handle.write("\n")

    charset_path = output_dir / "charset.txt"
    with charset_path.open("w", encoding="utf-8") as handle:
        for char in sorted(charset):
            handle.write(char + "\n")

    summary = {
        "dataset": name,
        "data_root": data_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "roots": roots,
        "annotation_files": {
            split: (output_dir / filename).as_posix()
            for split, filename in SPLIT_TO_FILE.items()
        },
        "all_annotation_file": (output_dir / all_filename).as_posix(),
        "all_count": len(all_records),
        "counts": {split: len(records) for split, records in split_records.items()},
        "source_counts": split_sources,
        "charset_file": charset_path.as_posix(),
        "charset_size": len(charset),
        "missing_images": missing,
        "mmocr_dataset_config_hint": {
            "type": "OCRDataset",
            "data_root": data_root.as_posix(),
            "ann_file": f"{output_dir.relative_to(data_root).as_posix()}/textrecog_train.json"
            if output_dir.is_relative_to(data_root)
            else "use an absolute ann_file or place output-root under data-root",
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def selected_presets(preset):
    if preset == "all":
        return PRESETS.items()
    return [(preset, PRESETS[preset])]


def selected_exports(args):
    if (args.name is None) != (args.roots is None):
        raise ValueError("--name and --roots must be provided together")
    if args.name is not None:
        return [(args.name, args.roots)]
    return selected_presets(args.preset)


def main():
    args = parse_args()
    summaries = []
    for name, roots in selected_exports(args):
        summaries.append(export_dataset(name, roots, args))
    for summary in summaries:
        print(
            f"{summary['dataset']}: counts={summary['counts']} "
            f"charset={summary['charset_size']} output={summary['output_dir']}"
        )


if __name__ == "__main__":
    main()
