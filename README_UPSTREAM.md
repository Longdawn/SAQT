# SAQT: Structure-Aware Query Learning for Vertical Historical Chinese Text Recognition

This repository contains the implementation and experiment artifacts for **SAQT**, a structure-aware query learning framework for cropped vertical ancient Chinese text recognition. The project focuses on single-column / line-level recognition: given a cropped vertical text image, the model predicts the corresponding character sequence.

SAQT uses character-localization supervision to learn structure-aware query representations, then converts vertically ordered query outputs into a CTC recognition sequence. For cross-dataset adaptation, the training pipeline supports charset-aware classifier initialization, CTC head reconstruction, full-model recognition finetuning, and development-set-fixed decoding calibration.

## Scope

- Single-column / line-level vertical ancient text recognition.
- Character boxes are used as training-time localization supervision when available.
- Inference outputs a text sequence, not page regions or detected character boxes.
- Experiments are reported with CER, AR, CR, empty prediction rate, predicted/ground-truth length ratio, and length-bucket metrics when available.

This repository is not a page-level layout analysis, reading-order prediction, or document parsing system.

## Main Entry Points

- `main_synthetic.py`: character-localization / query pretraining.
- `finetuning.py`: classifier-head reconstruction and full CTC recognition finetuning.
- `evaluation.py`: standalone evaluation entry point.
- `engine.py`: shared training and evaluation loops.

## SAQT Model And Configs

- `models/saqt/`: paper-facing SAQT model registration layer.
- `models/dino/`: compatibility implementation layer for the query-based detector backbone and existing checkpoints.
- `config/SAQT_MTHV2.py`: paper-facing MTHv2 recognition config.
- `config/SAQT_HDRC.py`: paper-facing HDRC recognition config.
- `config/MTHV2_stage1_query_budget.py`: query-budget localization pretraining config.

The legacy `*_dtlr.py` configs remain for checkpoint compatibility and historical reproducibility. New paper-facing runs should prefer `SAQT_*` config names.

## Data

Processed line-level datasets are expected under `data/`, including:

- `data/tkhmth2200_mth1000_dtlr/`
- `data/tkhmth2200_mth1200_dtlr/`
- `data/tkhmth2200_tkh_dtlr/`
- `data/hdrc_dtlr/`

The `_dtlr` suffix is kept as a data-format compatibility name for existing preprocessing outputs.

## Results And Logs

- `logs/paper_results_summary.md`: consolidated paper metrics.
- `logs/PAPER_RUNS.md`: paper-ready run manifest and archived-run policy.
- `docs/paper-drafts/`: manuscript drafts, figures, and LaTeX package.
- `debug_vis/`: box/query diagnostics and visualizations.

Do not judge a run from training loss alone. Recognition results should include dataset-level CER/AR/CR and short-text behavior when available.

## Minimal Verification

```bash
python - <<'PY'
from util.slconfig import SLConfig
import models

cfg = SLConfig.fromfile("config/SAQT_MTHV2.py")
print(cfg.modelname)
PY
```

For full training and evaluation commands, see `docs/EXPERIMENT_POSTPROCESS.md` and `logs/PAPER_RUNS.md`.

## Acknowledgement

SAQT builds on query-based detection and localization ideas, including DINO/Deformable-DETR components and the public DTLR codebase. The paper-facing contribution in this repository is the adaptation of structure-aware query learning to vertical ancient Chinese line recognition, including query-to-CTC recognition, charset-aware classifier adaptation, and the experiment protocol for MTHv2/HDRC.
