"""Matched probe for target-domain query geometry retention."""

from config.MTHV2_dtlr import *


query_geometry_retention_coef = 0.05
query_geometry_retention_confidence_power = 1.0
query_geometry_teacher_checkpoint = (
    "logs/mthv2_qbudgetstage1pre_mthv2_head_0603/checkpoint_best_regular.pth"
)
