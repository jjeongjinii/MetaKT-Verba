# Data

## stimuli/

- `stimuli_master.json`: the 138 items exactly as shown to raters. Each item has `item_id` (display ID), `skill`, `steps` (last four attempts: `step`, `correct`, `hint_count`, `response_time_ratio` relative to the student's own mean), `skill_cum_accuracy`, and `sentences` (`id`, `en`, `ko`).
- Narratives were generated and verified in English; the Korean translations were reading aids and were not verified.
- `internal_answer_key.csv`: hidden item key: `display_id`, `internal_item_id`, `is_negative_control`, `true_origin_item_id` (donor item of a negative control), `category`, `nli_tertile`, `mean_entailment`, `faithfulness_rate`, `in_core_set`.
- `internal_answer_key_sentences.csv`: hidden sentence key: `display_id`, `sentence_id`, `sentence_en`, `sentence_ko`, `translation_ok`, `cited_indicators`, `auto_status` (Stage 3 verdict), `auto_entailment`, `auto_detail`.

## ratings/

`metakt_verba_ratings_R1.csv` ... `R5.csv`, one per rater: `item_id`, `rater_id`, `level1_tags`(per sentence: `S1:supported;S2:partial;...`), `faithfulness`, `actionability`, `clarity` (1-5), `free_text`, `duration_sec`. The 40 items rated by all five raters form the core set.

## minimal_pairs/

`mini_survey_responses.csv`: `rater_id`, `item_id` (1A, 1B, 2A, 2B, 3, 4), `agreement_score` (1-5), `reason`.

## results/

- `rq1/pipeline_summary_<model>.csv`: one row per interaction and condition (400 x 6) with per-interaction `faithfulness_rate`, `hallucination_rate`, `structural_violation_rate`, `conditional_faithfulness_rate` and related fields. Table 2 reports the condition means.
- `rq2/judge_predictions_<model>.csv`: few-shot judge output on the 97 core-set sentences: `fold`, `item_id`, `sentence_id`, `majority_tag`, `pred_label`, `pred_confidence_supported` (0-100), `pred_reasoning`.
- `rq2/recovered_nli_scores.csv`: NLI scores for sentences without a stored score: `recovered_entailment`, `recovered_citation`, `recovery_method` (`inferred_negative_control` or `recovered_from_tag_residue`).

### verifier_family/

Results from the verifier-family robustness experiment comparing four verifier families under two premise conditions: Stage-1 symbolic facts (`fact`) and teacher-visible behavior records (`record`).

- `cases.csv`: public case-level inputs used in the verifier comparison. Columns: `item_id`, `sentence_id`, `sentence_text`, `majority_tag`, `n_raters`, `cited_indicators`, `indicator`, `is_core`, `is_negative_control`, `fact_premise_list`, `fact_premise`, `record_premise`. `fact_premise_list` preserves separate cited facts for verifiers that score each fact independently; `fact_premise` is the joined representation used by the LLM verifier.
- `scores_all_verifiers.csv`: sentence-level verifier scores used to reproduce the statistical analyses.
- `verifier_x_premise.csv`: AUROC estimates and bootstrap confidence intervals by verifier, premise type, analysis subset, and strict/lenient label coding.
- `within_indicator_by_verifier_premise.csv`: within-indicator discrimination results by verifier and premise type.
- `paired_vs_mdeberta_holm.csv`: paired AUROC comparisons between each alternative verifier and the mDeBERTa baseline, with Holm-adjusted p-values.
- `paired_record_vs_fact_holm.csv`: paired within-verifier comparisons of record-premise versus fact-premise AUROC, with Holm-adjusted p-values.
- `negative_controls_by_premise.csv`: verifier scores and pass rates for normal versus negative-control items under each premise condition.
- `diagnostics.csv`: analysis-set sizes, missing-score counts, label counts, and negative-control counts used to check the verifier-family analysis inputs.
