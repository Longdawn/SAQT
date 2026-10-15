"""SAQT HDRC low-resource transfer protocol.

The training fraction applies only to the HDRC training split. Validation and
test remain complete, and the charset is still built from the fixed dataset
protocol so classifier dimensions are identical across fractions. Override
``mth1000_train_fraction`` for the 25% and 50% conditions.
"""

from config.SAQT_HDRC import *


mth1000_train_fraction = 0.10
mth1000_train_fraction_seed = 20260718
mth1000_train_fraction_anchor = 0.10
