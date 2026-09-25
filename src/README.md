# MetaKT-Verba Source Code

This directory contains the pipeline, experiments, and analysis scripts for MetaKT-Verba. Run all commands from the repository root. See the main `README.md` for setup and reproduction instructions and `data/README.md` for data and column descriptions.

## Pipeline

- `metakt_indicators.py` — computes the nine Meta-KT indicators (Table 1).
- `stage1_grounding.py` — converts continuous indicators into symbolic facts.
- `stage2_generation.py` — generates citation-constrained narratives from grounded facts.
- `stage3_verification.py` — verifies generated sentences using NLI and structural checks.
- `run_pipeline.py` — runs the full pipeline, baselines, and ablations for RQ1 (Table 2).

## Human evaluation and RQ2

- `prepare_calibration_stimuli.py` — constructs the human-rating stimuli and negative controls.
- `aggregate_ratings.py` — aggregates expert ratings and computes reliability, control, and validity statistics (Table 3).
- `threshold_retuning_analysis.py` — evaluates NLI discrimination and threshold sensitivity (Table 3).
- `within_indicator_auroc.py` — computes within-indicator discrimination results (Table 5).
- `few_shot_judge_experiment.py` — runs the few-shot LLM-judge experiment (Table 4).
- `core_set_io.py` — shared utilities for loading the core human-rated sentence set.
- `nli_baseline_scores.py` — constructs NLI baseline scores for the judge comparison.
- `compare_auroc.py` — bootstrap utilities for AUROC confidence intervals and paired comparisons.
- `run_three_way_comparison.py` — compares Ministral, Llama, and NLI on a common sentence set (Table 4).

## RQ3 analyses

- `export_disagreement_cases.py` — exports human–Stage 3 disagreement cases.
- `code_free_text_disagreements.py` — codes rater comments associated with disagreement cases.
- `audit_cognitive_avoidance.py` — audits cognitive avoidance against the full interaction logs.
- `analyze_cognitive_avoidance_survey.py` — analyzes the cognitive-avoidance minimal-pair survey.

## Notes

- Paths are relative to the repository root unless overridden by command-line arguments.
- Use seed `42` when reproducing the reported experiments.
- Korean and English strings used as model prompts or experimental materials are part of the experimental setup and should not be modified for reproduction.
- GPU access is required for generation and NLI verification; the remaining analysis scripts are primarily CPU-based.

For the exact commands used to reproduce each paper result, see the repository-level `README.md` and `scripts/reproduce.sh`.
