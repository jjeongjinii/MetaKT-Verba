# MetaKT-Verba source code

Run scripts from the repository root and pass public input/output paths explicitly. See [../README.md](../README.md) for setup and example CPU commands and [../data/README.md](../data/README.md) for schemas and analysis filters.

The mapping below follows the latest manuscript's Tables **1, 2, and 3** and Sections 5.2–5.4. There is no Table A1 in this manuscript. Some research script comments and generated filenames still use older table numbers.

## Pipeline and RQ1 — Table 1

| Script | Role |
|---|---|
| `stage1_grounding.py` | map continuous metacognitive indicators to symbolic facts |
| `stage2_generation.py` | generate narratives constrained to cite grounded facts |
| `stage3_verification.py` | perform sentence-level NLI and structural checks |
| `run_pipeline.py` | run generation baselines and ablations |
| `rq1_recompute_from_json.py` | recompute generation metrics and paired interaction-bootstrap comparisons from stored JSON, on CPU |
| `rq1_full_bestmatch.py` | re-score Full sentences against candidate facts for `Full†`; requires an NLI model |

The manuscript analyzes `naive`, `template`, `full`, `ablate_stage1`, and `ablate_citation`. Table 1 displays Full, Full†, w/o Stage 1†, w/o Citation†, and Naive†, with F and SVR. Template-only and cF belong to Section 5.2.2. Exclude the stored `ablate_stage3` condition from the paper results. `Full†` is re-scoring, not regeneration.

## Rating protocol and alignment — Table 2

- `prepare_calibration_stimuli.py`: create stimuli, practice material, answer keys, and negative controls. Re-analysis uses the fixed released stimuli rather than freshly generated replacements.
- `aggregate_ratings.py`: join R1–R5 responses, aggregate sentence labels, and compute reliability and item-level alignment.
- `threshold_retuning_analysis.py`: evaluate automatic-score discrimination and threshold sensitivity against sentence labels.
- `cluster_bootstrap_ci.py nli`: compute item-cluster AUROC intervals, including cognitive-avoidance exclusions and supported sensitivity options.

The item correlations in Table 2 use the original answer-key scores. Sentence discrimination uses reconstructed sentence scores. Weighted kappa excludes unknown ratings pairwise. The current aggregation code requires adjustment to follow these definitions consistently; changing README terminology alone does not change its calculations.

## Verifier comparison — Table 3 and Section 5.3.3

- `verifier_family_experiment.py`: score mDeBERTa, DeBERTa-large, and MiniCheck under symbolic-fact, behavior-record, and control premises. Its verbalized-confidence Gemma path supplies supplementary settings rather than the latest manuscript's primary Gemma rows.
- `gemma_logprob_scores.py`: obtain normalized Yes/No next-token probabilities for the main Gemma analysis; preserve raw Yes/No probabilities.
- `cluster_bootstrap_ci.py verifiers`: aggregate stored main scores, estimate item-cluster intervals, and compare record minus fact on shared sentences.

Use `data/results/verifier_family/scores_main.csv` for Table 3's Fact/Record entries and record-minus-fact comparisons. Add Neutral/Shuffled results from `review_subsets.py record-controls` to complete the table. Use `scores_gemma_sensitivity.csv` only for supplementary verbalized-confidence settings; the latest manuscript uses zero-shot Gemma logprob for all four premise conditions.

The paired analysis uses 83 core or 287 broader-set eligible sentences, strict and lenient labels, 10,000 item-bootstrap draws, and seed 42. Pass the full verifier names from `scores_main.csv` explicitly; the script's default aliases differ. Use `--holm-family all` for the single 16-comparison record-minus-fact family supporting Table 3's asterisks and Section 5.3.3.

`table5_verifier_auroc_cluster.csv` supplies current Table 3's Fact/Record rows: select both `scope=core` and `scope=all`, with `set=shared`. The historical `tableA1_paired_record_minus_fact_cluster.csv` supplies paired-gain statistics; it does not correspond to a numbered Table A1 in the latest manuscript. Neither output alone is the complete Table 3 because control rows come from the control analysis.

## Indicator and qualitative diagnostics

- `within_indicator_auroc.py`: summarize scores and discrimination within cited indicators.
- `audit_cognitive_avoidance.py`: audit indicator direction against full-log inputs; requires the original/verified derived data.
- `review_subsets.py`: nonaffective subset, human leave-one-rater-out, restatement, record-control comparisons, and unsupported-split analysis (requires missing coding input).

`review_subsets.py` must be added to the assembled release before these diagnostics are executable. Nonaffective filtering must use a fixed lexicon or frozen sentence IDs. The latest manuscript's restatement result is 42/287 (14.6%), matching the checked implementation; leave-one-rater-out ranges are .72–.84 strict and .81–.93 lenient. Unsupported-split requires the complete manual coding, which is not supplied.

## Record controls — Table 3 and Section 5.4.2

`make_control_cases.py` constructs five shuffled-record variants and one content-free premise, preserving sentence text, facts, and human labels. It and `review_subsets.py` are required for the latest manuscript's Table 3 control rows and Section 5.4.2 diagnostics.

Use the updated scoring scripts' premise selection and output tags to keep control scores separate from main scores. Map `hyponly` to Table 3's Neutral rows; average scores across five shuffles per sentence before computing Shuffled AUROC. `review_subsets.py` computes paired item-bootstrap differences. Follow the latest manuscript's unadjusted diagnostic intervals for matched-minus-control contrasts, separately from the record-minus-fact Holm family.

## Fidelity — Section 5.3.3

`fidelity_analysis.py` evaluates fact/record scores against the blind single-author fidelity labels and compares fidelity with educator support. Retest analysis uses 20 repeated labels in the latest manuscript. `review_analyses.py` provides annotation-sheet preparation and overlapping analysis functions.

Fidelity is a manuscript experiment, rather than an optional future study. Its labels, mapping key, and retest data have not been supplied, so this analysis is not yet executable from released inputs. The two scripts use different identifier/label schemas; choose a canonical schema or provide a converter. The exploratory greedy label-flip search is not a manuscript table and must not be described as a guaranteed global minimum.

## Shared helpers and release assembly

Preserve the existing public `metakt_indicators.py`, `export_indicators.py`, `audit_cognitive_avoidance.py`, and `core_set_io.py`, even where they are absent from a local research copy. Verify exporter schema and row-ID equivalence before promising full data regeneration.

`verifier_common.py` is a proposed consolidation, not an implemented module. The current verifier script still imports helpers from older few-shot and score-recovery scripts. Migrate the required prompt, case-table, sampling, citation, and I/O helpers and check imports before removing those scripts. Earlier source-file counts predated the latest manuscript's required control and fidelity analyses; they are not a complete release checklist.

Use the latest verifier and logprob scorer versions with record-premise selection and tagged control outputs. Move credentials to environment variables and replace server-specific default paths before distribution.

## Outside the minimal paper release

Earlier standalone Ministral/Llama few-shot experiments, minimal-pair survey analysis, and MetaKT+ repair/training experiments are excluded. Retain any helpers that current scripts still import until consolidation is complete.

Extra verbalized-confidence Gemma settings are supplementary artifacts with no numbered table in the latest manuscript. Missing annotation inputs constrain reproduction of fidelity and counter-evidence results; they do not remove those experiments from the paper's scope.

The release does not yet provide a verified single-command end-to-end regeneration workflow. Remaining definition differences and unavailable analyses are documented in the repository-level README.
