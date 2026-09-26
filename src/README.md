# MetaKT-Verba Source Code

This directory contains the pipeline, experiments, and analysis scripts for MetaKT-Verba. Run all commands from the repository root. See the main `README.md` for setup and reproduction instructions and `data/README.md` for data and column descriptions.

## Pipeline

- `metakt_indicators.py` - computes the nine Meta-KT indicators (Table 1).
- `export_indicators.py` - exports the nine indicators and interaction-context fields from the raw ASSISTments data to `data/interim/full_predictions2.csv`.
- `stage1_grounding.py` - converts continuous indicators into symbolic facts.
- `stage2_generation.py` - generates citation-constrained narratives from grounded facts.
- `stage3_verification.py` - verifies generated sentences using NLI and structural checks.
- `run_pipeline.py` - runs the full pipeline, baselines, and ablations for RQ1 (Table 2).

## Human evaluation and RQ2

- `prepare_calibration_stimuli.py` - constructs the human-rating stimuli and negative controls.
- `aggregate_ratings.py` - aggregates expert ratings and computes reliability, control, and validity statistics (Table 3).
- `threshold_retuning_analysis.py` - evaluates NLI discrimination and threshold sensitivity (Table 3).
- `within_indicator_auroc.py` - computes within-indicator discrimination results.
- `few_shot_judge_experiment.py` - runs the few-shot LLM-judge experiment.
- `core_set_io.py` - shared utilities for loading the human-rated sentence set and reconstructing stimulus information.
- `nli_baseline_scores.py` - constructs NLI baseline scores for the judge comparison.
- `recover_missing_nli_scores.py` - recomputes missing NLI scores from reconstructed citations or explicitly marked inference procedures.
- `compare_auroc.py` - bootstrap utilities for AUROC confidence intervals and paired comparisons.
- `run_three_way_comparison.py` - compares the few-shot LLM judges and NLI baseline on a common sentence set.

## Verifier-family robustness

- `verifier_family_experiment.py` - evaluates four verifier families under both Stage-1 symbolic-fact (`fact`) and teacher-visible behavior-record (`record`) premise conditions. It supports mDeBERTa, DeBERTa-large, Bespoke-MiniCheck-7B, and Gemma-3-27B-it.
- `four_verifier_statistics.py` - reproduces verifier-family AUROC summaries, within-indicator results, paired comparisons against mDeBERTa, paired record-vs-fact comparisons, negative-control analyses, and diagnostics from `scores_all_verifiers.csv`.

Public verifier-family outputs are documented in `data/results/verifier_family/README.md` and `data/README.md`. Pickle files produced during model scoring are local checkpoints and are not part of the public result data.

## RQ3 analyses

- `export_disagreement_cases.py` - exports human-Stage 3 disagreement cases.
- `code_free_text_disagreements.py` - codes rater comments associated with disagreement cases.
- `audit_cognitive_avoidance.py` - audits cognitive avoidance against the full interaction logs.
- `analyze_cognitive_avoidance_survey.py` - analyzes the cognitive-avoidance minimal-pair survey.

## Notes

- Paths are relative to the repository root unless overridden by command-line arguments.
- Use seed `42` when reproducing the reported experiments.
- Korean and English strings used as model prompts or experimental materials are part of the experimental setup and should not be modified for reproduction.
- Generation, NLI verification, and the local verifier-family models require GPU access.
- Gemma verifier experiments use an OpenAI-compatible inference endpoint.
- Statistical aggregation and analysis scripts can be run on CPU once the sentence-level model scores have been generated.
- Intermediate checkpoints and model caches should be stored under `outputs/` or `ckpts/` rather than committed as public result data.

For the exact commands used to reproduce each paper result, see the repository-level `README.md` and `scripts/reproduce.sh`.
