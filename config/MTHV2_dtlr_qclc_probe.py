"""MTHv2 probe for query-conditioned length calibration (QCLC).

The branch predicts log line length from pooled localization-query features
and uses the prediction to make a bounded, sample-level blank/nonblank CTC
adjustment. It is experiment-only and leaves the trusted MTHv2 config intact.
"""

from config.MTHV2_dtlr import *


use_ctc_length_calibration = True
ctc_length_init = 8.0
ctc_length_loss_coef = 0.01
ctc_length_bias_scale = 0.25
ctc_length_bias_max = 0.5
