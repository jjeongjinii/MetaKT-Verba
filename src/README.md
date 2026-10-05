# MetaKT-Verba Source Code

This directory contains the pipeline, experiments, and analysis scripts for MetaKT-Verba. Run all commands from the repository root. See the main `README.md` for setup and reproduction instructions and `data/README.md` for data and column descriptions.

## §3 Pipeline

- `stage1_grounding.py` - converts continuous indicators into symbolic facts.
- `stage2_generation.py` - generates citation-constrained narratives from grounded facts.
- `stage3_verification.py` - verifies generated sentences using NLI and structural checks.

## Table 1, §5.2
- `run_pipeline.py` - runs the full pipeline, baselines, and ablations for RQ1 (Table 2).
- `rq1_recompute_from_json.py`
- `rq1_full_bestmatch.py`
- `cluster_bootstrap_ci.py`

## §4 Stimuli and control group
- `prepare_calibration_stimuli.py` - constructs the human-rating stimuli and negative controls.

## Table 2
- `aggregate_ratings.py` - aggregates expert ratings and computes reliability, control, and validity statistics (Table 3).
- `threshold_retuning_analysis.py` - evaluates NLI discrimination and threshold sensitivity (Table 3).
- `cluster_bootstrap_ci.py`

## Table 3
- `verifier_family_experiment.py` - evaluates four verifier families under both Stage-1 symbolic-fact (`fact`) and teacher-visible behavior-record (`record`) premise conditions. It supports mDeBERTa, DeBERTa-large, Bespoke-MiniCheck-7B, and Gemma-3-27B-it.
- `gemma_logprob_scores.py`
- `cluster_bootstrap_ci.py`

## §5.3.3 Gemma sensitivity
- `verifier_family_experiment.py`

## §5.4.1
- `within_indicator_auroc.py` - computes within-indicator discrimination results.

## §5.4.3
- `cluster_bootstrap_ci.py nli`
- `audit_cognitive_avoidance.py` - audits cognitive avoidance against the full interaction logs.
- `metakt_indicators.py`
For the exact commands used to reproduce each paper result, see the repository-level `README.md` and `scripts/reproduce.sh`.
