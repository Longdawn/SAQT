"""SAQT config for the internal CuratedHistoricalColumns-v1 dataset."""

from config.MTH1000_dtlr import *


modelname = "saqt"

mth1000_root = "curated_historical_v1_dtlr"
mth1000_raw_root = "curated_sources_raw"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = False

forced_direction = "vertical"
mth1000_filter_direction = "vertical"
