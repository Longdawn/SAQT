"""Matched MTHv2 probe for greedy-CTC collapse-aware length consistency."""

from config.MTHV2_dtlr import *


ctc_collapsed_count_loss_coef = 0.01
ctc_collapsed_count_loss_short_weight = 3.0
