# MetaKT-Verba

Code and study artifacts for **“Faithful to What? Auditing Automatic Faithfulness Verification of LLM Narratives for Metacognitive Knowledge Tracing.”**

MetaKT-Verba converts Meta-KT's nine metacognitive indicators into short narratives for teachers. Stage 1 grounds indicators in symbolic facts, Stage 2 generates sentences with fact citations, and Stage 3 checks the sentences using NLI. The study evaluates narrative generation and audits automatic verification against five expert raters.

This repository focuses on the experiments in Tables 1, 2, 3, and their supporting analyses. Earlier Ministral/Llama few-shot judge comparisons, minimal-pair surveys, and MetaKT+ feature-repair experiments are outside this release's scope.

## Repository layout

```text
src/                              pipeline, scoring, and analysis scripts
instruments/                      rating protocol and rater interface
data/stimuli/                     fixed rating stimuli and answer keys
data/ratings/                     expert responses R1–R5
data/derived/                     indicator/context input for regeneration
data/results/rq1/                 generation logs and Full best-match scores
data/results/verifier_family/     cases, main scores, and Gemma sensitivity scores
data/record_controls/              optional shuffled-record/hypothesis-only inputs
results/                          reference tables and diagnostics
outputs/                          locally generated analysis outputs
```

See [data/README.md](data/README.md) for inputs and score definitions and [src/README.md](src/README.md) for script responsibilities. This is the intended release layout; optional control inputs and additional analysis/helper scripts must be included before their corresponding commands are used.

## Setup

Use Python 3.10 or later and install the repository dependencies:

```bash
pip install -r requirements.txt
```

Analysis of released scores runs on CPU. Regenerating narratives or verifier scores requires the relevant models, a suitable GPU, and, for Gemma scoring, an OpenAI-compatible inference endpoint. Provide credentials through environment variables such as `HF_TOKEN`; see `.env.example`.

Run commands from the repository root. Some research scripts retain server-specific default paths, so pass the input and output paths explicitly.

## Paper results and inputs

| Result | Scripts | Released inputs |
|---|---|---|
| Table 1: generation, baselines, and ablations | `rq1_recompute_from_json.py`, `rq1_full_bestmatch.py` | three generation JSON files and three best-match CSV files |
| Table 2: human agreement and automatic–human alignment | `aggregate_ratings.py`, `threshold_retuning_analysis.py`, `cluster_bootstrap_ci.py` | fixed answer keys, R1–R5 responses, sentence-level scores |
| Table 3: verifier discrimination | `cluster_bootstrap_ci.py verifiers` | `scores_main.csv` | `cluster_bootstrap_ci.py verifiers` | `scores_main.csv`; Holm correction across all 16 comparisons |
| Gemma scoring sensitivity | `verifier_family_experiment.py` | `scores_gemma_sensitivity.csv` |
| Within-indicator and cognitive-avoidance analyses | `within_indicator_auroc.py`, `cluster_bootstrap_ci.py nli`, `audit_cognitive_avoidance.py` | sentence-level scores; derived/full-log inputs for the log audit |
| Nonaffective, leave-one-rater-out, and restatement diagnostics | `review_subsets.py` | main scores and, for leave-one-rater-out, R1–R5 responses |
| Optional record-premise controls | `make_control_cases.py`, `review_subsets.py` | shuffled-record and hypothesis-only cases/scores |

Table 1 uses `naive`, `template`, `full`, `ablate_stage1`, and `ablate_citation`. `Full†` is a best-match re-evaluation of Full sentences, not a sixth generation condition. The stored JSON also contains `ablate_stage3`; exclude that condition when presenting the paper table.

### Recompute RQ1 from stored generation logs

```bash
python src/rq1_recompute_from_json.py --inputs "data/results/rq1/pipeline_results_*_en.json" --out-dir outputs/rq1_recompute --agg mean --cluster interaction --n-boot 10000 --seed 42
```

This command does not load a language model. Inspect the generated condition table using the five conditions above. Best-match CSVs are separate released results; rerunning `rq1_full_bestmatch.py` loads an NLI model.

### Recompute the four-verifier analysis

```bash
python src/cluster_bootstrap_ci.py --n-boot 10000 --seed 42 --out-dir outputs/verifier_cluster verifiers --scores data/results/verifier_family/scores_main.csv --answer-key data/stimuli/internal_answer_key.csv --verifiers "mDeBERTa-v3-base-mnli-xnli,DeBERTa-v3-large-mnli-fever-anli-ling-wanli,Bespoke-MiniCheck-7B,Gemma-3-27B-it-logprob" --holm-family all
```

The script currently writes `table5_verifier_auroc_cluster.csv`; its **core/shared** rows correspond to the current paper's Table 3. It also writes `paired_record_minus_fact_cluster.csv`. Historical filenames and script comments do not define the current paper's table numbering.

## Analysis conventions

- `fact` means Stage-1 symbolic facts; `record` means the teacher-visible behavior record.
- The main Gemma score is normalized Yes/No next-token probability, not verbalized confidence. Verbalized confidence and in-context settings are stored separately.
- Strict positives are `supported`; lenient positives are `supported` or `partial`.
- The main sentence analysis excludes negative controls, unknown labels, missing scores, and sentences without a reconstructable fact premise. Both premises use the same eligible sentences: **83 core sentences** or **287 sentences overall**. The five-rater core contains 40 items and 97 sentences before sentence eligibility filtering.
- Verifier uncertainty is estimated by resampling items, with 10,000 bootstrap draws and seed 42. Table 3 uses one Holm family across 16 comparisons. RQ1 resamples interactions. Input order matters for reproducing seeded bootstrap results.
- Fixed rating stimuli and original answer keys are preserved. Regenerating stimuli does not replace the material shown to the raters.
- For shuffled-record controls, average each sentence's scores across five shuffles before computing AUROC. This differs from averaging five AUROCs. Control-generation seeds use 2027+k, k=1,…,5.

## Reproduction status

The prepared main score file was checked for unique keys, complete scores, matching sentence metadata, logprob normalization, and the 83/287 analysis counts. Preparation of these files does not constitute a complete rerun of the paper.

The following issues remain explicit rather than silently changing inputs to match the manuscript:

- Table 2's item correlations use the **original item answer-key scores**, while sentence discrimination uses reconstructed scores. Weighted kappa excludes unknown ratings pairwise; the current aggregation implementation needs to be aligned with that definition.
- Table 3 contains a remaining adjusted-p-value discrepancy. Regenerate all comparisons together from the main logprob scores before treating the released table as final.
- The supplied restatement code identifies 42 sentences, whereas the manuscript reports 43. The leave-one-rater-out lenient minimum also needs a rounding/definition check.
- The complete 130-sentence counter-evidence coding and fidelity labels are not provided. Unsupported-split and fidelity analyses are therefore outside the executable reproduction scope.
- The stratified within-source AUROC and ordinal-median sensitivity require an explicit implementation. The original checkpoint AUC and full-log audit have not been independently rerun in this release preparation.

Generation/scoring scripts also require public-path cleanup and helper-import consolidation before a complete regeneration workflow is available. No single-command full reproduction is claimed at this stage.

## License

Code is licensed under MIT; see `LICENSE`. Released study artifacts follow the repository's CC BY 4.0 data terms. ASSISTments source data are not redistributed here; obtain them from their providers and follow their applicable terms.
