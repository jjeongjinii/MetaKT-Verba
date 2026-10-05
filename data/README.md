# Data

This directory contains the fixed study inputs and stored scores used for CPU re-analysis of the current manuscript's Tables 1–3 and Sections 5.2–5.4. Model regeneration and analysis of stored results are separate workflows. Paths below describe the intended public release; missing annotation inputs are identified explicitly.

## `stimuli/`

| File | Contents |
|---|---|
| `stimuli_master.json` | 138 items as shown to raters: display `item_id`, skill, recent steps, cumulative skill accuracy, and English/Korean sentences |
| `internal_answer_key.csv` | display-to-internal item mapping, negative-control and donor information, original automatic item scores, and core membership |
| `internal_answer_key_sentences.csv` | sentence IDs/text, translation checks, citations, original Stage-3 verdicts and scores |
| `calibration_practice.json` | practice material used before the rating task |

Step context includes correctness, hint count, and response-time ratio relative to the student's own mean. Generation and automatic verification used English; Korean translations were reading aids.

The answer keys were hidden from raters during data collection. They are released for ID mapping, control identification, and reproduction. Their original automatic scores must remain distinct from subsequently reconstructed verifier scores.

## `ratings/`

```text
metakt_verba_ratings_R1.csv
metakt_verba_ratings_R2.csv
metakt_verba_ratings_R3.csv
metakt_verba_ratings_R4.csv
metakt_verba_ratings_R5.csv
```

Each row is one rater's response for an item. Fields include `item_id`, `rater_id`, sentence-level `level1_tags`, item-level `faithfulness`, `actionability`, `clarity`, free-text comments, and response duration. Sentence tags are encoded as entries such as `S1:supported;S2:partial`.

The five files contain 298 item-rating responses. The 40 items rated by all five raters contain 97 sentences before automatic-score eligibility filtering. The paired core analysis uses 83 eligible sentences.

## `results/rq1/`

Generation logs:

```text
pipeline_results_Qwen3-8B_en.json
pipeline_results_Llama-3.1-8B-Instruct_en.json
pipeline_results_Ministral-8B-Instruct-2410_en.json
```

Full best-match re-evaluations:

```text
full_bestmatch_Qwen3-8B.csv
full_bestmatch_Llama-3.1-8B-Instruct.csv
full_bestmatch_Ministral-8B-Instruct-2410.csv
```

Each JSON stores generation and per-sentence verification records for six conditions and 400 sampled interactions per condition. The paper analyzes five generation conditions: `naive`, `template`, `full`, `ablate_stage1`, and `ablate_citation`. Table 1 displays Full, Full†, w/o Stage 1†, w/o Citation†, and Naive†, with F and SVR; Template-only and cF are reported in Section 5.2.2. Preserve failure records; the valid Ministral sample is 399 for Full and the Stage-1 ablation. Stored records are not all successful observations.

The best-match CSVs support `Full†`: each Full sentence is re-evaluated against candidate facts. They are not outputs of an additional generation condition. Older `pipeline_summary_*.csv` files are derived summaries and must not replace the JSON-based recomputation when definitions differ.

## `results/verifier_family/`

| File | Purpose |
|---|---|
| `cases.csv` | sentence-level inputs, labels, facts, records, citations, and core/control flags |
| `scores_main.csv` | **2,584 rows**: three non-Gemma verifiers plus Gemma logprob; Table 3's Fact/Record entries and paired comparisons |
| `scores_gemma_sensitivity.csv` | **999 rows**: supplementary Gemma verbalized-confidence settings; not the current Table 3 input |
| `sentence_level_with_scores.csv` | 287-row sentence analysis input for threshold, within-indicator, and cognitive-avoidance analyses |
| `score_file_manifest.json` | input/output SHA-256 hashes and main-analysis counts for the prepared score files |

`sentence_level_with_scores.csv` is the threshold-analysis output, distinct from the rating aggregator's `sentence_level_summary.csv`.

### Main scores

`scores_main.csv` replaces the old `scores_all_verifiers.csv` as the main-analysis input. It contains these verifier names:

```text
mDeBERTa-v3-base-mnli-xnli
DeBERTa-v3-large-mnli-fever-anli-ling-wanli
Bespoke-MiniCheck-7B
Gemma-3-27B-it-logprob
```

For each verifier, fact has 293 stored rows and record has 353 stored rows. The file includes negative controls and other rows excluded from the main analysis. Filtering yields 287 eligible sentences overall and 83 in the core for each verifier/premise.

Key fields are `item_id`, `sentence_id`, `sentence_text`, `majority_tag`, `n_raters`, `cited_indicators`, `indicator`, `is_core`, `is_negative_control`, `fact_premise_list`, `fact_premise`, `record_premise`, `verifier`, `premise`, and `score`. Additional context and judge outputs are preserved when available. The row key is `(item_id, sentence_id, verifier, premise)`.

`fact_premise_list` stores separate cited facts; `fact_premise` stores their joined representation. Complex metadata fields retain their CSV string representation and are not necessarily JSON. Scores are numbers in [0,1], but they have model-specific definitions and should not be treated as identically calibrated probabilities.

For Gemma logprob:

```text
score = p_yes_raw / (p_yes_raw + p_no_raw)
```

The raw Yes/No probability columns are preserved. They are empty for non-Gemma rows. Original verbalized Gemma scores are excluded from this file.

### Gemma sensitivity scores

| `setting` | `premise` | `verifier` | Rows |
|---|---|---|---:|
| `verbalized_fact_zeroshot` | fact | `Gemma-3-27B-it` | 293 |
| `verbalized_record_incontext` | record | `Gemma-3-27B-it` | 353 |
| `verbalized_record_zeroshot` | record | `Gemma-3-27B-it-0shot` | 353 |

Here `score` is the model's verbalized `confidence_supported` divided by 100. Settings are identified from the scoring code's prompt paths. Exact shot counts and model revisions are not newly inferred from these files. The key is `(item_id, sentence_id, setting, premise)`.

The fact zero-shot rows are stored once. Pair either record setting with those fact rows using item/sentence IDs and the shared eligible set. In-context and zero-shot record settings must not be averaged together. Gemma logprob scores are in the main file; Table 3's Neutral/Shuffled Gemma scores are in the record-control files. The manuscript does not assign a numbered table to the supplementary verbalized-confidence file.

### Analysis filters and labels

Exclude negative controls, unknown majority tags, missing scores, and sentences without a fact premise. For a paired comparison, require both compared scores. Use `is_core` or the original key's `in_core_set` for core membership.

Strict coding treats only `supported` as positive. Lenient coding treats `supported` and `partial` as positive. Control analyses deliberately use a different inclusion rule and should be labelled separately.

## `record_controls/` — Table 3 and Section 5.4.2

These controls are part of the latest manuscript's Table 3, rather than an optional experiment outside the paper. They contain six case files and 24 score files in PKL format:

- `cases_shuffle1.pkl` through `cases_shuffle5.pkl`, and `cases_hyponly.pkl`.
- For each of `mdeberta`, `deberta_large`, and `minicheck`: `scores_<model>_shuffle1.pkl` through `scores_<model>_shuffle5.pkl`, and `scores_<model>_hyponly.pkl`.
- `scores_gemma_logprob_shuffle1.pkl` through `scores_gemma_logprob_shuffle5.pkl`, and `scores_gemma_logprob_hyponly.pkl`.

All 30 files have 353 rows each. The six Gemma CSV counterparts are redundant storage copies and are unnecessary if the PKLs are released. PKL loading requires a compatible pandas/pyarrow environment.

Only record premises change; sentence text, facts, and majority labels remain fixed. Shuffled records come from a different item and use the same donor for all sentences within an item. Donors share the skill where possible, with a fallback to other skills; the supplied data have 88.4% same-skill donors by item. Retain `donor_item_id` and `donor_same_skill`.

The hypothesis-only condition uses the content-free premise `The student is practicing a skill.` Map it to **Neutral** in Table 3. For **Shuffled**, compute a sentence's mean score across all five shuffles, then compute AUROC. Keep variants separate and verify that every analyzed sentence has all five scores. Combine these results with Fact and Record from `scores_main.csv`, for both core/broader sets and strict/lenient coding.

The matched-minus-control paired item-bootstrap intervals in Section 5.4.2 are unadjusted diagnostic intervals. They are separate from the 16 Holm-adjusted record-minus-fact comparisons used in Table 3's significance markers and Section 5.3.3. Historical `tableA1_*` filenames refer to these paired results, not a current manuscript Table A1.
