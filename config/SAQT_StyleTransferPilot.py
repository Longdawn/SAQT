"""SAQT config for the page-disjoint cross-style transfer pilot."""

from config.MTH1000_dtlr import *


modelname = "saqt"

mth1000_root = "style_stratified_transfer_pilot_v2_candidate/zhanguoce_song"
mth1000_raw_root = "curated_sources_raw"
mth1000_labels_file = "labels.pkl"
mth1000_images_subdir = "lines"
mth1000_image_ext = "jpg"
mth1000_use_char_boxes = False

forced_direction = "vertical"
mth1000_filter_direction = "vertical"

# Exactly 50 of the fixed 200 training columns. Validation and test are full.
mth1000_train_fraction = 0.25
mth1000_train_fraction_seed = 20260803
mth1000_train_fraction_anchor = 0.25
