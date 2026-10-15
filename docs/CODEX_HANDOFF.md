# Codex Handoff

Last updated: 2026-07-30

## Recovery Snapshot (2026-08-02)

- The matched HDRC 10% random-head control completed with `EXIT:0`, using the
  exact same 2,429-line subset, source checkpoint, seed and optimization budget
  as the charset-aware condition.
- Random-head calibrated test CER/AR/CR are `25.80/74.20/75.42`; matched
  charset-aware results are `9.75/90.25/91.48`. The 16.05-point CER reduction
  is now the authoritative classifier-transfer ablation.
- Random-head artifacts are in
  `logs/saqt_hdrc_transfer10random0801_full_seed42_0718/`; the matched mapped
  artifacts are in `logs/saqt_hdrc_transfer10_full_seed42_0718/`.
- No training process or tmux session remains. The paper-facing table was
  replaced with this strict control and the anonymous manuscript was rebuilt:
  33 result rows pass with zero warnings/failures, and PDF QA reports 27 A4
  pages, 34 references, 53 assets and zero failures.
- A subsequent implementation audit clarified that the control transfers
  random pretrained source-classifier rows rather than freshly initializing
  the target head. The manuscript now uses the accurate term
  `character-agnostic random-row transfer`. It also reports 82,030 aligned
  source training lines with 827,113 real boxes and 3,266 shared MTHv2/HDRC
  characters covering 96.56% of HDRC test character occurrences. The rebuilt
  PDF is 28 pages and passes the same automated checks.
- Active method probe: tmux `saqt_query_geometry_retention_0802` runs a matched
  MTHv2 query-geometry-retention experiment on GPU0. It anchors top-T teacher
  query vertical centers/heights during CTC adaptation to close the current
  non-differentiable sorting/geometry-drift gap. Output is
  `logs/mthv2_query_geometry_retention_resume_16000_0802`; matched baseline
  validation CER is `0.11879202286282202`. Keep only on a positive matched
  validation result.
- The probe subsequently improved validation CER from `0.118792` to `0.114455`
  at the matched global-step endpoint with nearly unchanged blank ratio. The
  full 12-epoch run and automatic postprocessing are active in tmux
  `saqt_query_geometry_retention_full_0802`; output is
  `logs/mthv2_query_geometry_retention_full_0802`.

## Recovery Snapshot (2026-07-30)

- HDRC 50% completed before the 2026-07-18 host interruption and passed unified
  postprocessing. The strict low-resource evidence is now `4/4`. Its
  validation-selected `blank=-2.0, nonblank=0.8` calibrated test CER/AR/CR are
  `6.3410/93.6590/94.5572`.
- MTHv2 seed 44 completed epochs 0--9; epoch 7 remains the protected best at
  development CER `0.09624162552586628`. The run stopped during incomplete
  epoch 10, and the host later rebooted. No seed-44 test result should be used
  until unified postprocessing completes.
- Exact output-directory resume now restores epoch/global step,
  optimizer/scheduler, and historical best CER in `finetuning.py`; focused
  tests are in `tests/test_finetuning_resume.py`.
- Active tmux: `saqt_mthv2_seed44_recovery_0730`, running on the single
  currently visible RTX 3090. It resumed at epoch 10/global step 65000 and will
  automatically run unified postprocessing after epoch 11.
- The former QCLC and final-evidence watcher tmux sessions were lost at reboot.
  Relaunch them only after seed 44 reaches strict `3/3` readiness; do not start
  a competing GPU job while seed 44 is active.
- Seed 44 subsequently completed with `EXIT:0`; strict seed evidence is now
  `3/3`. Calibrated three-seed CER/AR/CR are
  `3.50+/-0.25 / 96.50+/-0.25 / 96.77+/-0.23`.
- The complete HDRC low-resource evidence is `4/4`; Figure 4 and its CSV were
  generated and pass the Figure 1--4 asset gate.
- Active tmux is now `saqt_mthv2_qclc_queue_0730` on GPU0. It runs the fixed
  development-only QCLC gate and evaluates test only if the gate is positive.
- QCLC passed that gate with development CER `0.11595983771087572` versus
  `0.11879202286282202`; unified QCLC and matched-baseline test collection is
  active.
- A current evidence-only draft excluding unfinished QCLC numbers is available
  at `docs/paper-drafts/paper_draft.md` and
  `docs/paper-drafts/submission/saqt_aas_anonymous_draft.pdf`. The PDF is 28
  A4 pages with 35 references and 54 assets; automated and selected-page visual
  checks pass.
- QCLC later completed with `EXIT:0`. Its calibrated test CER is `3.97%`
  versus matched baseline `3.95%`; empty prediction rate improves
  `0.57% -> 0.18%`, so the manuscript treats it as a diagnostic tradeoff rather
  than a core gain.
- The complete current initial draft now includes Method 3.5, the matched QCLC
  tables, three-seed statistics, and HDRC low-resource Figure 4. The anonymous
  PDF is 29 A4 pages with 35 references and 61 assets; automated QA and
  selected-page visual checks pass. No experiment process is running.

## Current Authoritative Snapshot (2026-07-18)

The active goal is an anonymous, evidence-audited submission to 《自动化学报》.
Older June experiment descriptions below are provenance, not current process
state. On resume, read `experiments/notes/goal_status.md`, then inspect the tmux
sessions and status files listed here before launching anything.

Active/queued experiments:

- MTHv2 main-model seed 43 completed and passed strict unified-result checks.
  Its clean test CER/AR/CR are `4.1471/95.8529/96.0297`; validation-selected
  `blank=-2.0, nonblank=0.8` gives calibrated test CER/AR/CR
  `3.7412/96.2588/96.5440`.
- MTHv2 seed 44 started at `13:29:35` in tmux
  `saqt_mthv2_seed44_queue_0718` on GPU1, under the matched 12-epoch protocol.
- HDRC 10% nested low-resource transfer completed with `EXIT:0`. Its calibrated
  test CER/AR/CR are `9.7462/90.2538/91.4827` at validation-selected
  `blank=-2.0, nonblank=1.0`; see
  `logs/saqt_hdrc_transfer10_full_seed42_0718/micro_test_bias_bm20_nb10_0718.json`.
- HDRC 25% completed and passed strict unified-result checks. Its clean test
  CER/AR/CR are `8.4905/91.5095/91.7687`; validation-selected
  `blank=-2.0, nonblank=1.0` gives calibrated test CER/AR/CR
  `6.9354/93.0646/94.2265`.
- HDRC 50% started automatically at `12:48:00` in tmux
  `saqt_hdrc_low_resource_queue_0718` on GPU0. Its independent classifier head
  completed with best development CER `0.19417441044302478` at `epoch=2,
  global_step=15000`; the matched 12-epoch full run started from that protected
  checkpoint at `14:14:22`.
- QCLC matched probe: waiting tmux `saqt_mthv2_qclc_queue_0718`; it starts on
  GPU1 only after the seed-44 pipeline reports `EXIT:0`. The probe stops before
  test postprocessing unless validation CER beats `0.11879202286282202`. A
  positive branch now also postprocesses the already-trained same-head,
  same-seed, same-budget baseline `logs/mthv2_base_resume_1000_0607`; test
  results are reported without a second acceptance gate.

Current intermediate validation:

- MTHv2 seed 43: all 12 epochs completed; epoch-7 CER
  `0.10030692971795412` remains the saved best (`epoch=7,
  global_step=55000`). The strict seed builder now accepts seeds 42 and 43
  (`2/3` ready), while seed 44 is running.
- MTHv2 seed 44: epoch-0 development CER is `0.12208094240183898`; this is an
  intermediate validation result. Epoch 1 improved it to
  `0.10987788140880596`, and epoch 2 improved it to
  `0.10709067322003646`, and epoch 3 improved it to
  `0.10432417147997171`; the protected checkpoint is `epoch=3,
  global_step=35000`, and unified test evidence is not yet present.
- HDRC 50% full finetuning: epoch-0 development CER is
  `0.09264636472628718`; the protected checkpoint is `epoch=0,
  global_step=20000`. This is already below the 25% condition's best
  development CER, but remains intermediate evidence until the fixed run and
  unified postprocessing finish. Epochs 1--3 regressed to
  `0.2516628219491267/0.21231068454181926/0.22554991326911883`; the protected
  best remains epoch 0.
- HDRC 25% independently reconstructed head: epoch-0/1/2 CER
  `0.2587038576998038/0.22920193771657965/0.21438858156384388`. The full run
  has verified `resume`, `lr=1e-5`, `epochs=12`, fraction/seed, and nested-anchor
  settings in `config_args_all.json`. Full epoch-0/1/2/3/4 validation CER is
  `0.12145243720268993/0.10928225991968375/0.10614284204433325/0.10314356406401663/0.09852336950933273`;
  epoch 4 at `global_step=24280` remained the protected best through epoch 11.
  Unified postprocessing completed with `EXIT:0`, and the strict HDRC builder
  accepts 10%, 25%, and 100% (`3/4` ready) while 50% is running.
- HDRC 10% independently reconstructed head: epoch-2 CER
  `0.2563417660518842`.
- HDRC 10% full finetuning completed the fixed 12 epochs. Epoch-0--4 CER
  `0.17032209517700092/0.16614229078634163/0.15332695182697897/0.15522997114050782/0.1605014377142521`;
  epoch 5--11 were
  `0.32454301673147534/0.33619754565510424/0.2921901262956864/0.3601330345462829/0.331627247856814/0.37062222019942087/0.3630817910127929`.
  Epoch 2 remains the protected saved best (`global_step=7284`). Unified
  postprocessing completed at `2026-07-18T07:59:29+08:00`. Clean test
  CER/AR/CR are `12.8832/87.1168/87.2822`; calibrated empty rate is `1.0352%`,
  prediction/GT length ratio is `0.99084`, and calibrated bucket CER for
  `1/2/3-5/6-10/11+` is `13.0573/12.2307/8.6687/11.9613/9.2189`.

New reproducibility/collection artifacts:

- Nested HDRC protocol and fingerprints:
  `experiments/notes/hdrc_low_resource_protocol_0718.md` and
  `hdrc_low_resource_manifest_0718.json`. The protocol explicitly records
  that CTC `max_iterations=10000` is a per-epoch processed-sample cap, not an
  optimizer-step count: 10%/25% traverse their pools each epoch, whereas
  50%/100% stop at 10,000 samples and reshuffle next epoch.
- Parameterized low-resource pipeline:
  `experiments/scripts/run_hdrc_low_resource_fraction.sh`.
- Three-seed manifest and pipeline:
  `experiments/notes/mthv2_seed_manifest_0718.json` and
  `experiments/scripts/run_mthv2_seed_repeat.sh`. The strict statistics builder
  normalizes only the permitted seed/output/config-entry and `dino/saqt` alias
  differences, then requires the remaining full configuration snapshots to be
  identical. Current seed-42 and seed-43 configs pass this comparison.
- Strict result builders:
  `docs/paper-drafts/tools/build_hdrc_low_resource_results.py` and
  `build_mthv2_seed_statistics.py`. They intentionally return incomplete until
  every required JSON exists and matches the validation-selected bias.
  The HDRC builder also follows each full run back to its independently trained
  head configuration and verifies the common stage-1 source, charset mapping,
  adaptation flags, learning rates, epoch counts, batch size, and sample
  budget. The completed 10% and existing 100% runs pass this lineage gate.
  The HDRC CSV and final Table 10 include a verified
  `max_train_samples_per_epoch=min(label-pool lines, 10000)` column; the final
  integrator rejects a missing or inconsistent per-epoch cap.
- Matched QCLC evidence contract:
  `experiments/notes/mthv2_qclc_manifest_0718.json` and
  `docs/paper-drafts/tools/build_mthv2_qclc_ablation.py`. The builder validates
  the development-set gate, complete clean/calibrated artifacts, length
  buckets, calibration selection, checkpoint lineage, declared module fields,
  and full-config equality outside the QCLC fields. A positive result is
  integrated automatically; an incomplete positive branch stops the build.
- Final result integration:
  `docs/paper-drafts/tools/integrate_final_experiment_results.py`. It modifies
  the result section, bilingual abstract, and conclusion only after both
  strict builders have produced complete CSV files; incomplete inputs leave
  the manuscript unchanged. A development-positive QCLC branch additionally
  inserts a reproducible Method 3.5 with the implemented pooling, length loss,
  bounded adaptive bias, supervision boundary, and fixed hyperparameters.
  `verify_result_tables.py` rejects a QCLC result table if that method block is
  absent. Once the generated seed/low-resource result block exists, the same
  verifier also requires synchronized Chinese/English abstract and conclusion
  evidence blocks, preventing a partial final freeze.
- Figure asset gate: `docs/paper-drafts/tools/verify_figure_assets.py`. The
  final watcher runs it with `--require-figure4` after low-resource generation;
  it checks SVG/PDF/CSV completeness, editable SVG text, and vector-only data
  figures. Figure 4 uses a zero CER baseline, actual sample-count tick labels,
  an approximately 16 cm vector canvas, grayscale-safe redundant line/marker
  encodings, and an explicit single-seed/no-error-bar caption.
- Split audit: `tools/audit_recognition_splits.py` and
  `experiments/notes/dataset_split_audit_0718.{json,md}`. The corrected HDRC
  source-stem resolver confirms `936/113/123` pages and zero line/page overlap.
- Anonymous manuscript:
  `docs/paper-drafts/submission/saqt_aas_anonymous_draft.pdf`. Current QA is 25
  A4 pages, 35 sequential references, 53 assets, zero machine-check failures;
  see `verify_anonymous_submission.py` and `submission/README.md`.
  The Chinese results section again contains the verified MTHv2
  no-localization lower-bound table; the PDF verifier now rejects stale HTML
  or PDF artifacts by checking manuscript-to-HTML-to-PDF modification order.
- Identity-bearing submission template:
  `docs/paper-drafts/submission/author_materials_template.md`. It is excluded
  from the anonymous builder and collects author order, bilingual affiliations,
  contributions, funding, declarations, signatures, and institution-seal
  status. The official one-page 2025 author-commitment PDF has also been
  downloaded to
  `docs/paper-drafts/submission/forms/aas_author_commitment_2025_official.pdf`;
  `forms/README.md` records the official source, 2026-07-18 access date, and
  SHA-256. Authors must recheck the source on submission day, complete all
  signatures, and obtain the required institutional seal.
- Submission reproducibility index:
  `docs/paper-drafts/submission/REPRODUCIBILITY.md`. It links the authoritative
  split evidence, configs, runners, manifests, evaluation tools, baseline
  evidence, strict builders, and verifiers without copying provisional values.
- AI-use disclosure: `docs/paper-drafts/110_ai_use_disclosure_draft.md` records
  the current machine-observed environment as Codex CLI `0.144.5`, model
  `gpt-5.6-sol`. Authors still need to confirm whether earlier sessions used
  other versions.
- Citation audit: `docs/paper-drafts/112_reference_metadata_audit_0718.md`.
  The nine previously identifier-free records were checked against primary
  sources; all 35 cited records now have DOI, stable URL, or arXiv provenance,
  and the three BibTeX copies are byte-identical.
- The two Overleaf ZIP copies are rebuilt from the current `latex/` tree and
  have SHA-256
  `8aae8ea0e12d636d03bff45407ef03a98641ae9a7cd862c5af62e80ccf61f7ab`.
  `verify_latex_package.py` now hashes every archived file against the current
  source tree instead of only comparing the two ZIP copies to each other.

Remaining hard gates:

1. Finish seeds 43/44 and report mean plus sample standard deviation.
2. Finish HDRC 10/25/50/100 clean and validation-fixed calibrated curves.
3. Rebuild result tables, Figure 4, abstract/results/conclusion, and anonymous
   PDF from frozen numbers.
4. Obtain author-confirmed AI tool/model disclosure and the author metadata,
   signatures, and institutional stamp required outside the anonymous PDF.

The CPU final-evidence watcher also waits for the queued QCLC gate. An explicit
negative result permits automatic finalization. A validation-positive result
must pass the matched QCLC builder and is then inserted into the result section
before rebuilding the final PDF; incomplete matched evidence stops with
`STOP_QCLC_POSITIVE_INCOMPLETE_MATCHED_EVIDENCE`. The watcher was reloaded at
`2026-07-18T09:07:13+08:00` without restarting GPU work.

## Project Goal

This repository is used for SAQT vertical ancient Chinese text recognition experiments. The practical target is single-column / line-level recognition for vertical ancient text images. It is not a page-level layout-analysis project and should not be framed as detecting page regions, reading order, or full-document layout.

The paper story currently being shaped is: SAQT uses structure-aware query/localization pretraining, then adapts the query representation for vertical ancient text recognition with CTC finetuning, development-set-fixed decode calibration, and cross-dataset evaluation on MTHv2-combo and HDRC. Historical MTH1000, MTH1200, and TKH runs remain useful for provenance. The writing should present SAQT as the paper-facing method while acknowledging query-based detection/localization influences where appropriate.

## Current Code Structure

- `main_synthetic.py`: stage-1 detection / localization pretraining.
- `finetuning.py`: CTC head reconstruction and full recognition finetuning.
- `evaluation.py`: standalone evaluation entry point.
- `engine.py`: shared training and evaluation loops.
- `models/saqt/`: paper-facing SAQT model registration layer.
- `models/dino/`: compatibility implementation layer for the query transformer, criterion, and deformable attention modules used by existing checkpoints.
- `datasets/`: dataset builders and transforms.
- `datasets/MTH1000.py`: MTH1000-style processed single-column line dataset.
- `datasets/MTHCombo.py`: combined MTH1000/MTH1200/TKH dataset.
- `util/ctc_decoding.py`: CTC greedy decode and decode-time calibration.
- `tools/analyze_ctc_errors.py`: CER/AR/CR, empty prediction, length-bucket analysis.
- `tools/eval_ctc_micro_batch.py`: batch CTC micro CER/AR/CR evaluator for valid/test clean or fixed bias.
- `tools/sweep_ctc_decode_bias.py`: blank / nonblank decode calibration sweeps.
- `tools/format_ctc_result_row.py`: formats completed CTC JSON results into Markdown/CSV paper table rows.
- `tools/diagnose_boxes.py`: stage-1 query/box diagnostic.
- `tools/export_mmocr_recog_dataset.py`: export processed SAQT-compatible datasets to MMOCR text-recognition format.
- `experiments/configs/mmocr/`: MMOCR comparison configs.
- `PROJECT_MAP.md`: concise repository map.

Important configs:

- `config/SAQT_MTHV2.py`: paper-facing MTHv2-combo CTC finetuning config.
- `config/SAQT_HDRC.py`: paper-facing HDRC CTC finetuning config.
- `config/MTH1000_dtlr.py`: legacy MTH1000/TKH/HDRC-style CTC finetuning config.
- `config/MTHV2_dtlr.py`: legacy MTHv2-combo CTC finetuning config.
- `config/MTHV2_stage1.py`: full MTHv2 stage-1 detection pretraining config.
- `config/MTH1000_MTH1200_stage1.py`: trusted real stage-1 baseline.
- `config/MTHV2_dtlr_sgq_short_ce.py`: SGQ short-query CE experiment.
- `config/HDRC_dtlr_sgq_short_ce.py`: HDRC SGQ short-query CE experiment.

Important data/log locations:

- `data/tkhmth2200_mth1000_dtlr/`
- `data/tkhmth2200_mth1200_dtlr/`
- `data/tkhmth2200_tkh_dtlr/`
- `data/hdrc_dtlr/`
- `data/mmocr_exports/`
- `logs/`
- `debug_vis/`
- `experiments/notes/goal_status.md`
- `experiments/notes/paper_draft_v1.md`
- `logs/paper_results_summary.md`
- `docs/EXPERIMENT_POSTPROCESS.md`: canonical workflow after a finetuning experiment finishes.

## Completed Changes

- Added project map:
  - `PROJECT_MAP.md`
- Added / updated paper draft:
  - `experiments/notes/paper_draft_v1.md`
- Added SGQ short-query CE experiment path:
  - `config/MTHV2_dtlr_sgq_short_ce.py`
  - `config/HDRC_dtlr_sgq_short_ce.py`
  - `tests/test_sgq_short_query_loss.py`
- Added adaptive decode calibration support:
  - `util/ctc_decoding.py`
  - `tools/sweep_ctc_decode_bias.py`
  - `tests/test_ctc_eval_utils.py`
- Added MMOCR export and comparison workflow files under `data/mmocr_exports/` and `experiments/configs/mmocr/`.
- Created full-MTHv2 stage-1 candidate and downstream head/full finetune runs.

## Key Experiment Results

Primary paper table is in `logs/paper_results_summary.md`.

Best SAQT-style results currently summarized there:

| Dataset | Decode | Test CER | Test AR | Test CR | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| MTH1000 | bias `-1.2/1.0` | 4.88 | 95.12 | 95.50 | better than clean 5.29 CER |
| MTH1200 | bias `-1.2/1.0` | 5.18 | 94.82 | 95.00 | clean 5.95 CER |
| TKH | bias `-1.6/1.0` | 1.04 | 98.96 | 99.02 | strongest dataset result |
| MTHv2-combo-qbudget | bias `-2.0/0.8` | 3.31 | 96.69 | 96.90 | current best combined-data result |
| HDRC | bias `-2.0/0.4` | 9.30 | 90.70 | 91.80 | harder domain, long-column weakness |

Important current MTHv2 qbudget result:

- Run: `logs/mthv2_qbudgetstage1pre_mthv2_full_0603`
- Checkpoint: `logs/mthv2_qbudgetstage1pre_mthv2_full_0603/checkpoint_best_regular.pth`
- Clean test: CER `3.83`, AR `96.17`, CR `96.29`
- Validation-selected bias: `blank=-2.0`, `nonblank=0.8`
- Bias test: CER `3.31`, AR `96.69`, CR `96.90`, empty `0.89`, Pred/GT `0.997`
- Result files:
  - `logs/mthv2_qbudgetstage1pre_mthv2_full_0603/micro_test_clean_0603.json`
  - `logs/mthv2_qbudgetstage1pre_mthv2_full_0603/decode_bias_sweep_valid_0603.json`
  - `logs/mthv2_qbudgetstage1pre_mthv2_full_0603/micro_test_bias_b-20_nb08_0603.json`

Important baseline run paths:

- Trusted real stage-1:
  - `logs/mth1000_mth1200_stage1_0512-1755/checkpoint.pth`
- Trusted MTH1000 full finetune:
  - `logs/mth1000mth1200pre_mth1000ft_full_0513-2257/checkpoint_best_regular.pth`
- MTHv2-combo head baseline:
  - `logs/mthv2_mth1000mth1200tkh_head_0528-0003/`
- MTHv2-combo full baseline:
  - `logs/mthv2_mth1000mth1200tkh_full_0528-0957/checkpoint_best_regular.pth`

MMOCR MTHv2-combo baselines:

| Model | Orientation | Test CER | Test AR | Test CR |
| --- | --- | ---: | ---: | ---: |
| CRNN | `Rot90(k=3)` | 4.49 | 95.51 | 95.63 |
| SVTR-tiny | `Rot90(k=3)` | 4.44 | 95.56 | 95.68 |
| SVTR-small | `Rot90(k=3)` | 4.66 | 95.34 | 95.44 |
| SVTR-L | `Rot90(k=3), 48x160` | 4.84 | 95.16 | 95.26 |
| CRNN | `Rot90(k=1)` | 27.26 | 72.74 | 73.81 |
| SVTR-tiny | `Rot90(k=1)` | 6.58 | 93.42 | 93.52 |

Full-MTHv2 stage-1 candidate:

- Stage-1 run: `logs/mthv2_full_stage1_det_0531-1258`
- Stage-1 final train loss: about `3.6403`
- Box diagnostic on MTH1000 valid:
  - `debug_vis/mthv2_full_stage1_det_on_mth1000_valid_0601`
  - avg blank ratio about `0.8416`
  - avg predicted nonblank count about `142.6`
  - avg GT length about `7.49`
- Interpretation: usable enough to try, but over-activated compared with trusted real stage-1 baseline.

Full-MTHv2 stage-1 -> MTHv2 head reconstruction:

- Run: `logs/mthv2_fullstage1pre_mthv2_head_0601`
- Best head checkpoint exists.
- CER by epoch:
  - epoch 0: `0.1687470099455089`
  - epoch 1: `0.1578794820103037`
  - epoch 2: `0.15283326830434354`
- Old MTHv2 head baseline epoch-2 CER: `0.14526833428705468`.
- Interpretation: new head is worse than old head but close enough that the user chose to try full finetuning.

Historical June full-MTHv2 finetune snapshot (no longer active):

- tmux session: `dtlr_mthv2_full_ft_0601`
- Log dir: `logs/mthv2_fullstage1pre_mthv2_full_tmux_0601`
- Source checkpoint: `logs/mthv2_fullstage1pre_mthv2_head_0601/checkpoint_best_regular.pth`
- Command uses `cuda:0`, `dataset_file=mth_combo`, `config/MTHV2_dtlr.py`, `num_classes=6727`, `batch_size=2`, `lr=1e-5`, `epochs=12`, `eval_epoch=1`, `max_iterations=10000`.
- As of the latest check, the run had reached epoch 8 training.
- `log.txt` contains validation rows through epoch 7:
  - epoch 0 CER `0.14209875725049612`
  - epoch 1 CER `0.14348676074442737`
  - epoch 2 CER `0.13439168210258723`
  - epoch 3 CER `0.13650454108627846`
  - epoch 4 CER `0.13991440891643625`
  - epoch 5 CER `0.14382442905243265`
  - epoch 6 CER `0.1522016002257195`
  - epoch 7 CER `0.23396587828711218`
- Best so far appears to be epoch 2 by validation CER.
- `checkpoint_best_regular.pth` exists in the active log dir.
- Do not judge final usefulness until full run finishes and clean/bias test summaries with length buckets are generated.

## Failed Attempts / Known Bad Runs

- `logs/mthv2_full_stage1_0531-1248` and early bad full-MTHv2 stage-1 attempts had incorrect/loss-exploding behavior and should not be reused.
- `logs/mthv2_fullstage1pre_mthv2_full_0601` was started through the Codex tool session and stopped at epoch 0 step `3160/42088` before validation; it saved no checkpoint. Do not use it as a result.
- SGQ short-query CE MTHv2 full run (`logs/mthv2_sgq_short_ce_full_0528-2240`) encountered stdout pipe blocking after the managing session was interrupted. Its epoch 0 validation completed with CER about `0.1270`, but the process management was fragile.
- MMOCR SAR runs repeatedly hung in this environment, including during training with workers enabled. A safer resume with all dataloader workers set to 0 was attempted; treat SAR as unstable until fully verified.

## Current Main Problems

- The strongest narrative is not "vertical text alone", because rotated MMOCR baselines can perform well. The paper should emphasize structured localization pretraining, unified single-column recognition, low annotation/transfer framing, and calibrated decoding rather than simply "vertical recognition".
- MTHv2 short columns remain weak, especially `len=1` and `len=2` buckets.
- HDRC remains harder, especially long columns.
- Decode-time bias/calibration improves metrics but must be described carefully as validation-selected post-processing, not as a learned model module unless using the adaptive calibration code path.
- Full-MTHv2 stage-1 over-activates nonblank queries in box diagnostics.
- Long SAQT training should not be launched in a Codex foreground exec session; use `tmux`.

## Historical 2026-06-09 Next Steps

1. Let `dtlr_mthv2_full_ft_0601` finish or stop only if validation clearly degrades for multiple epochs and a best checkpoint is already saved.
2. After completion, run clean CTC error analysis on valid/test for `logs/mthv2_fullstage1pre_mthv2_full_tmux_0601/checkpoint_best_regular.pth`.
3. Run decode-bias sweep on validation, then apply selected calibration to test.
4. Report CER/AR/CR and length buckets for clean and bias.
5. Compare against:
   - `logs/mthv2_mth1000mth1200tkh_full_0528-0957`
   - MMOCR CRNN/SVTR baselines in `logs/paper_results_summary.md`
6. Update `logs/paper_results_summary.md` only after clean/bias test summaries exist.
7. For the paper, keep the method framing around query-based localization pretraining + CTC recognition + validation-selected calibration; do not overclaim page-level detection.

## Common Commands

Attach to the active DTLR training:

```bash
tmux attach -t dtlr_mthv2_full_ft_0601
```

Detach without stopping:

```text
Ctrl-b, then d
```

Check current DTLR full-finetune log lightly:

```bash
tail -n 40 logs/mthv2_fullstage1pre_mthv2_full_tmux_0601/info.txt
tail -n 20 logs/mthv2_fullstage1pre_mthv2_full_tmux_0601/log.txt
```

Launch current DTLR full finetune in tmux:

```bash
tmux new-session -d -s dtlr_mthv2_full_ft_0601 'cd /home/ubuntu/DTLR && /home/ubuntu/miniconda3/envs/DTLR/bin/python finetuning.py --device cuda:0 --dataset_file mth_combo --num_workers 0 --resume logs/mthv2_fullstage1pre_mthv2_head_0601/checkpoint_best_regular.pth --new_class_embedding --smart_mapping --resume_finetuning --path_old_charset logs/mthv2_full_stage1_det_0531-1258/charset.pkl --save_log --output_dir logs/mthv2_fullstage1pre_mthv2_full_tmux_0601 -c config/MTHV2_dtlr.py --options num_classes=6727 batch_size=2 lr=1e-5 epochs=12 eval_epoch=1 max_iterations=10000'
```

Trusted MTHv2 full baseline path:

```bash
logs/mthv2_mth1000mth1200tkh_full_0528-0957/checkpoint_best_regular.pth
```

Stage-1 box diagnostic pattern:

```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=/home/ubuntu/DTLR /home/ubuntu/miniconda3/envs/DTLR/bin/python tools/diagnose_boxes.py -c config/MTHV2_stage1.py --dataset_file mth1000 --checkpoint logs/mthv2_full_stage1_det_0531-1258/checkpoint.pth --split valid --device cuda:0 --num_workers 0 --num_samples 100 --viz_topk 40 --show_gt_boxes --output_dir debug_vis/NEW_DEBUG_DIR
```

## Pitfalls To Avoid

- Do not run long training in a foreground Codex exec session; use `tmux`.
- Do not keep polling logs every 30 seconds; it wastes token/context. Only inspect at key points.
- Do not update `experiments/notes/goal_status.md` on every minor progress check. Update only for launches, validation results, completion, errors, or major file/result changes.
- Do not judge experiments by training loss alone. Use validation CER and final CER/AR/CR plus length buckets.
- Do not treat `max_iterations` in `finetuning.py` as a global optimizer-step limit. It is a per-epoch sample cap; with `batch_size=2` and `max_iterations=10000`, each epoch is about 5000 optimizer steps.
- Do not assume `max_iterations` stops `main_synthetic.py` stage-1 runs. Stage-1 budget is effectively `len(train_loader) * epochs`.
- Do not silently edit shared baseline configs. Prefer new config files or explicit `--options`.
- Do not reuse interrupted/no-checkpoint runs as results.
- Do not delete logs/checkpoints/datasets unless explicitly asked.
- Do not present SAQT stage-1 character boxes as page-level detection. It is query/localization pretraining for line-level recognition.
- Do not claim a module is valid for the paper without clean/bias test results and length-bucket analysis.
