# MetaKT-Verba

Code and study artifacts for **“Faithful to What? Auditing Automatic Faithfulness Verification of LLM Narratives for Metacognitive Knowledge Tracing.”**

MetaKT-Verba converts Meta-KT's nine metacognitive indicators into short narratives for teachers. Stage 1 grounds indicators in symbolic facts, Stage 2 generates sentences with fact citations, and Stage 3 checks the sentences using NLI. The study evaluates narrative generation and audits automatic verification against five expert raters.

This repository follows the current manuscript's **Tables 1, 2, and 3** and supporting analyses in Sections 5.2–5.4. The manuscript has no Table A1. Earlier Ministral/Llama few-shot judge comparisons, minimal-pair surveys, and MetaKT+ feature-repair experiments are outside this release's scope.

## Repository layout

```text
src/                              pipeline, scoring, and analysis scripts
instruments/                      rating protocol and rater interface
data/stimuli/                     fixed rating stimuli and answer keys
data/ratings/                     expert responses R1–R5
data/derived/                     indicator/context input for regeneration
data/results/rq1/                 generation logs and Full best-match scores
data/results/verifier_family/     cases, main scores, and Gemma sensitivity scores
data/record_controls/              shuffled-record/hypothesis-only inputs for Table 3
data/annotations/                  fidelity and counter-evidence labels, when supplied
results/                          reference tables and diagnostics
outputs/                          locally generated analysis outputs
```

See [data/README.md](data/README.md) for inputs and score definitions and [src/README.md](src/README.md) for script responsibilities. This is the intended release layout; control inputs and additional analysis/helper scripts must be included before their corresponding commands are used. Annotation files are still needed for the fidelity and counter-evidence results reported in the manuscript.

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
| Table 3: Fact and Record rows, core and broader sets | `cluster_bootstrap_ci.py verifiers` | `scores_main.csv` |
| Table 3: Neutral and Shuffled rows; Section 5.4.2 | `make_control_cases.py`, `review_subsets.py` | control cases/scores plus `scores_main.csv` |
| Table 3 significance markers and Section 5.3.3 paired gains | `cluster_bootstrap_ci.py verifiers` | `scores_main.csv`; Holm correction across all 16 record-minus-fact comparisons |
| Within-indicator and cognitive-avoidance analyses | `within_indicator_auroc.py`, `cluster_bootstrap_ci.py nli`, `audit_cognitive_avoidance.py` | sentence-level scores; derived/full-log inputs for the log audit |
| Nonaffective, leave-one-rater-out, and restatement diagnostics | `review_subsets.py` | main scores and, for leave-one-rater-out, R1–R5 responses |
| Section 5.3.3: fidelity against symbolic facts | `fidelity_analysis.py`; compatible annotation preparation via `review_analyses.py` | main scores, fidelity labels, key, and retest labels; annotation files not yet supplied |
| Section 5.4.2: contradicted-versus-unrelated diagnostic | `review_subsets.py unsupported-split` | main scores and complete counter-evidence coding; coding file not yet supplied |

Table 1 displays **Full, Full†, w/o Stage 1†, w/o Citation†, and Naive†**, reporting F and SVR. `Full†` is a best-match re-evaluation of Full sentences, not a separate generation condition. Template-only is reported in Section 5.2.2, along with conditional faithfulness (cF), rather than as a column in Table 1. The generation analysis uses `naive`, `template`, `full`, `ablate_stage1`, and `ablate_citation`; exclude the additional stored `ablate_stage3` condition from the paper results.

### Recompute RQ1 from stored generation logs

```bash
python src/rq1_recompute_from_json.py --inputs "data/results/rq1/pipeline_results_*_en.json" --out-dir outputs/rq1_recompute --agg mean --cluster interaction --n-boot 10000 --seed 42
```

This command does not load a language model. Map the generated condition table to Table 1's displayed columns above; keep Template-only and cF as Section 5.2.2 results. Best-match CSVs are separate released results; rerunning `rq1_full_bestmatch.py` loads an NLI model.

### Recompute the four-verifier analysis

```bash
python src/cluster_bootstrap_ci.py --n-boot 10000 --seed 42 --out-dir outputs/verifier_cluster verifiers --scores data/results/verifier_family/scores_main.csv --answer-key data/stimuli/internal_answer_key.csv --verifiers "mDeBERTa-v3-base-mnli-xnli,DeBERTa-v3-large-mnli-fever-anli-ling-wanli,Bespoke-MiniCheck-7B,Gemma-3-27B-it-logprob" --holm-family all
```

The script currently writes `table5_verifier_auroc_cluster.csv`. Its **core/shared** and **all/shared** rows supply the **Fact and Record** entries in Table 3. Neutral and Shuffled entries must be added from the control analysis. It also writes `tableA1_paired_record_minus_fact_cluster.csv`, which supplies Table 3's significance markers and the paired-gain results in Section 5.3.3; this historical filename does not refer to a table in the current manuscript.

### Complete Table 3 with record controls

Use the six released control variants with `review_subsets.py record-controls`. Map `hyponly` to **Neutral**, and the per-sentence mean across `shuffle1`–`shuffle5` to **Shuffled**. Report both strict and lenient AUROC for the core (83) and broader (287) sets. Keep matched-minus-control paired intervals separate from the 16-comparison record-minus-fact Holm family: the manuscript reports control intervals without multiplicity adjustment.

`scores_gemma_sensitivity.csv` contains supplementary verbalized-confidence settings. These are not Table 3's Gemma rows: the current manuscript uses **zero-shot logprob** scores under all four premise conditions.

## Analysis conventions

- `fact` means Stage-1 symbolic facts; `record` means the teacher-visible behavior record.
- The main Gemma score is normalized Yes/No next-token probability, not verbalized confidence. Verbalized confidence and in-context settings are stored separately.
- Strict positives are `supported`; lenient positives are `supported` or `partial`.
- The main sentence analysis excludes negative controls, unknown labels, missing scores, and sentences without a reconstructable fact premise. Both premises use the same eligible sentences: **83 core sentences** or **287 sentences overall**. The five-rater core contains 40 items and 97 sentences before sentence eligibility filtering.
- Verifier uncertainty is estimated by resampling items, with 10,000 bootstrap draws and seed 42. Record-minus-fact comparisons use one Holm family across 16 comparisons; Table 3's asterisks indicate significant broader-set gains. Matched-minus-control intervals are unadjusted diagnostic intervals. RQ1 resamples interactions. Input order matters for reproducing seeded bootstrap results.
- Fixed rating stimuli and original answer keys are preserved. Regenerating stimuli does not replace the material shown to the raters.
- For shuffled-record controls, average each sentence's scores across five shuffles before computing AUROC. This differs from averaging five AUROCs. Control-generation seeds use 2027+k, k=1,…,5.

## Reproduction status

The prepared main score file was checked for unique keys, complete scores, matching sentence metadata, logprob normalization, and the 83/287 analysis counts. Preparation of these files does not constitute a complete rerun of the paper.

The following issues remain explicit rather than silently changing inputs to match the manuscript:

- Table 2's item correlations use the **original item answer-key scores**, while sentence discrimination uses reconstructed scores. Weighted kappa excludes unknown ratings pairwise; the current aggregation implementation needs to be aligned with that definition.
- Regenerate paired record-minus-fact comparisons from the main logprob scores, rather than using older tables based on verbalized Gemma confidence. The latest manuscript reports these results in Table 3 and Section 5.3.3.
- The latest manuscript's restatement result is 42/287 sentences (14.6%), and its leave-one-rater-out ranges are .72–.84 strict and .81–.93 lenient. These replace the older manuscript values and agree with the checked code results.
- The complete 130-sentence counter-evidence coding and fidelity labels are not provided. They are needed for manuscript results, although their analyses cannot yet be reproduced from the available inputs. The latest coding totals are 76 contradicted and 54 unrelated (22/40 passed and 54/90 failed sentences cite counter-evidence); the earlier partial coding file is not a substitute.
- The stratified within-source AUROC and ordinal-median sensitivity require an explicit implementation. The original checkpoint AUC and full-log audit have not been independently rerun in this release preparation.

Generation/scoring scripts also require public-path cleanup and helper-import consolidation before a complete regeneration workflow is available. No single-command full reproduction is claimed at this stage.

## License

Code is licensed under MIT; see `LICENSE`. Released study artifacts follow the repository's CC BY 4.0 data terms. ASSISTments source data are not redistributed here; obtain them from their providers and follow their applicable terms.
