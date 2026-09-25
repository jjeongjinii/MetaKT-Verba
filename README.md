# MetaKT-Verba

Anonymized code and data for the paper **"Faithful to What? Auditing Automatic Faithfulness Verification of LLM Narratives for Metacognitive Knowledge Tracing"** (under double-blind review).

MetaKT-Verba turns the nine metacognitive indicators of Meta-KT into short narratives for teachers: Stage 1 grounds the indicators in symbolic facts, Stage 2 generates sentences that must cite a fact, and Stage 3 verifies each sentence with NLI. The paper audits Stage 3 against five expert raters.

## Layout

```
src/                    pipeline, experiments and analyses (run from the repository root)
data/stimuli/           the 138 rated items and the hidden answer key
data/ratings/           expert ratings R1-R5
data/minimal_pairs/     minimal-pair survey responses
data/results/rq1/       per-interaction pipeline summaries (Table 2)
data/results/rq2/       few-shot judge predictions and recovered NLI scores (Table 4)
```

See `data/README.md` for file and column descriptions.

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

Generation and NLI verification need a GPU. Gated models read `HF_TOKEN` from the environment (see `.env.example`). 
The few-shot judges query an OpenAI-compatible endpoint, for example `vllm serve mistralai/Ministral-8B-Instruct-2410 --port 8000`.

## Reproducing the paper

| Paper | Script | Needs |
|---|---|---|
| Table 1 (indicator definitions) | `metakt_indicators.py` | raw data |
| Table 2 (RQ1) | `run_pipeline.py` | raw data, GPU (released summaries in `data/results/rq1`) |
| Section 4 (stimuli, negative controls) | `prepare_calibration_stimuli.py` | raw data, GPU (released in `data/stimuli`) |
| Table 3 (reliability, controls, convergent validity) | `aggregate_ratings.py` | CPU |
| Table 3 (discrimination, threshold sweep) | `threshold_retuning_analysis.py` | CPU |
| Table 4 (few-shot judges vs. NLI) | `few_shot_judge_experiment.py`, then `run_three_way_comparison.py` | LLM server, then CPU |
| Table 5 (within-indicator discrimination) | `within_indicator_auroc.py` | CPU |
| Section 5.4 (error direction, rater comments) | `export_disagreement_cases.py`, then `code_free_text_disagreements.py` | CPU |
| Section 5.4 (full-log audit of cognitive avoidance) | `audit_cognitive_avoidance.py` | raw data |
| Section 5.4 (minimal pairs) | `analyze_cognitive_avoidance_survey.py` | CPU |

**Full regeneration (raw data and GPU):**

1. Place the ASSISTments 2017 files (`student_log_*.csv` and the student label file) in `data/raw/assistments2017/`.
2. Build `data/interim/full_predictions2.csv` (indicator values and step context):
   `python src/export_indicators.py`.
3. RQ1: `python src/run_pipeline.py --model <model> --language en --sample-size 400 --seed 42`
   for `Qwen/Qwen3-8B`, `meta-llama/Meta-Llama-3.1-8B-Instruct` and `mistralai/Ministral-8B-Instruct-2410`.
4. Stimuli: `python src/prepare_calibration_stimuli.py --model Qwen/Qwen3-8B --seed 42`.
5. Judges (one endpoint per model): `python src/few_shot_judge_experiment.py --model <judge> --min-raters 5 --seed 42`.
6. NLI scores for sentences without a stored score: `python src/recover_missing_nli_scores.py`.
7. Cognitive-avoidance audit: `python src/audit_cognitive_avoidance.py`.

## Notes

- Table 4 uses NLI scores recovered for negative-control sentences (`--include-recovery-methods inferred_negative_control`, the default). Also using the 16 sentences recovered from tag residue (`all`) gives NLI AUROC 0.591 (strict) and 0.596 (lenient), with the same conclusions.
- All random seeds are 42.
- Comments and some console messages are in Korean.

## License

Code: MIT (`LICENSE`). Data in `data/`: CC BY 4.0. ASSISTments 2017 is not redistributed; obtain it from its providers.
