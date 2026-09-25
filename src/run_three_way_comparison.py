
import argparse
from typing import Optional

import numpy as np
import pandas as pd

from compare_auroc import bootstrap_auroc_ci, compare_two_judges
from nli_baseline_scores import combine_baseline_with_recovered

_ID_CANDIDATES = ["item_id", "display_id"]
_SID_CANDIDATES = ["sentence_id", "sid"]
_TAG_CANDIDATES = ["majority_tag", "true_label", "label"]
_CONF_CANDIDATES = ["pred_confidence_supported", "confidence_supported"]


def compute_y_true(tags: pd.Series, label_mode: str = "strict") -> np.ndarray:
    if label_mode not in ("strict", "lenient"):
        raise ValueError(f"label_mode must be 'strict' or 'lenient': {label_mode!r}")
    if label_mode == "strict":
        return (tags == "supported").astype(int).to_numpy()
    return tags.isin(["supported", "partial"]).astype(int).to_numpy()


def _find_col(df: pd.DataFrame, candidates, what: str, path: str) -> str:
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise KeyError(f"Could not find the {what} column in {path}. Tried: {candidates}; "
                        f"available columns: {list(df.columns)}")
    return col


def load_predictions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = _find_col(df, _ID_CANDIDATES, "item id", path)
    sid_col = _find_col(df, _SID_CANDIDATES, "sentence id", path)
    tag_col = _find_col(df, _TAG_CANDIDATES, "answer tag", path)
    conf_col = _find_col(df, _CONF_CANDIDATES, "predicted confidence", path)
    out = pd.DataFrame({
        "item_id": df[id_col], "sentence_id": df[sid_col],
        "majority_tag": df[tag_col], "pred_confidence_supported": df[conf_col],
    })
    n_before = len(out)
    out = out.dropna(subset=["pred_confidence_supported"])
    out = out[out["majority_tag"] != "unknown"]
    print(f"  {path}: {len(out)} valid rows out of {n_before} (excluded {n_before - len(out)} failed/unknown rows)")
    return out


def load_n_raters_map(core_set_cases_csv: str) -> pd.DataFrame:
    df = pd.read_csv(core_set_cases_csv, usecols=["item_id", "sentence_id", "n_raters"])
    return df


def run_three_way_comparison(
    ministral_predictions_csv: str, llama_predictions_csv: str, sentence_answer_key_csv: str,
    min_sample_size: int = 10, recovered_nli_scores_csv: Optional[str] = None,
    include_recovery_methods=("inferred_negative_control",),
    core_set_cases_csv: Optional[str] = None, label_mode: str = "strict",
) -> pd.DataFrame:
    print(f"Loading predictions.csv... (label coding: {label_mode})")
    ministral = load_predictions_csv(ministral_predictions_csv)
    llama = load_predictions_csv(llama_predictions_csv)

    common = ministral.merge(llama, on=["item_id", "sentence_id", "majority_tag"],
                              suffixes=("_ministral", "_llama"))
    print(f"\nSentences successfully judged by both Ministral and Llama: {len(common)}")

    target_keys = list(zip(common["item_id"], common["sentence_id"]))
    nli = combine_baseline_with_recovered(
        sentence_answer_key_csv, recovered_csv=recovered_nli_scores_csv, target_keys=target_keys,
        include_recovery_methods=include_recovery_methods,
    )

    merged = common.merge(nli[["item_id", "sentence_id", "confidence_supported"]],
                           on=["item_id", "sentence_id"], how="inner")
    n_after_key_merge = len(merged)
    print(f"Sentences with NLI baseline keys: {n_after_key_merge} "
          f"(of {len(common)} shared sentences, {len(common) - n_after_key_merge} have no matching key — "
          f"likely negative controls)")

    merged = merged.dropna(subset=["confidence_supported"])
    if len(merged) < n_after_key_merge:
        print(f"  Warning: {n_after_key_merge - len(merged)} rows have NaN auto_entailment "
              f"(no verification evidence, e.g. FAIL_INVALID_REF/FAIL_NO_CITATION); "
              f"excluded from final sample of {len(merged)} rows")

    if len(merged) < min_sample_size:
        raise ValueError(f"Final sample contains only {len(merged)} rows; check the column names and keys.")

    y_true = compute_y_true(merged["majority_tag"], label_mode)

    if core_set_cases_csv:
        n_raters_df = load_n_raters_map(core_set_cases_csv)
        before_n = len(merged)
        merged = merged.merge(n_raters_df, on=["item_id", "sentence_id"], how="left")
        n_missing = merged["n_raters"].isna().sum()
        if n_missing:
            print(f"\nWarning: {n_missing}/{before_n} sentences in core_set_cases.csv are missing n_raters "
                  f"(possible join-key mismatch; continuing)")

    print(f"\n{'=' * 60}\nPer-method AUROC, full sample (n={len(merged)}, label coding={label_mode})\n{'=' * 60}")
    for name, col in [
        ("Ministral-8B", "pred_confidence_supported_ministral"),
        ("Llama-3.1-8B", "pred_confidence_supported_llama"),
        ("NLI (mDeBERTa)", "confidence_supported"),
    ]:
        point, lo, hi = bootstrap_auroc_ci(y_true, merged[col].to_numpy())
        print(f"{name}: AUROC={point:.4f} [{lo:.4f}, {hi:.4f}]")

    other_mode = "lenient" if label_mode == "strict" else "strict"
    y_true_other = compute_y_true(merged["majority_tag"], other_mode)
    print(f"\n[Reference: recomputed with {other_mode} coding on the same sample; differences indicate an effect from partial labels]")
    for name, col in [
        ("Ministral-8B", "pred_confidence_supported_ministral"),
        ("Llama-3.1-8B", "pred_confidence_supported_llama"),
        ("NLI (mDeBERTa)", "confidence_supported"),
    ]:
        point, lo, hi = bootstrap_auroc_ci(y_true_other, merged[col].to_numpy())
        print(f"{name}: AUROC={point:.4f} [{lo:.4f}, {hi:.4f}]")

    if core_set_cases_csv and "n_raters" in merged.columns:
        print(f"\n{'=' * 60}\nn_raters-stratified analysis — diagnostic (label coding={label_mode})\n{'=' * 60}")
        print("If AUROC decreases after expanding the sample, determine whether this is due to judge performance "
              "or the expanded subset (single-rater, noisier labels). A result close to the original core-set "
              "range provides a useful diagnostic for the source of the change.")
        for label, mask in [
            ("n_raters=5 (original core-set)", merged["n_raters"] == 5),
            ("n_raters=1 (expanded subset, single rater)", merged["n_raters"] == 1),
        ]:
            sub = merged[mask]
            print(f"\n[{label}] n={len(sub)}")
            if len(sub) < 10:
                print("  Skipped because the sample is too small")
                continue
            y_true_sub = compute_y_true(sub["majority_tag"], label_mode)
            for name, col in [
                ("Ministral-8B", "pred_confidence_supported_ministral"),
                ("Llama-3.1-8B", "pred_confidence_supported_llama"),
                ("NLI (mDeBERTa)", "confidence_supported"),
            ]:
                point, lo, hi = bootstrap_auroc_ci(y_true_sub, sub[col].to_numpy())
                print(f"  {name}: AUROC={point:.4f} [{lo:.4f}, {hi:.4f}]")

    print(f"\n{'=' * 60}\nPrimary comparison: each LLM judge vs NLI baseline (n={len(merged)}, label coding={label_mode})\n{'=' * 60}")
    compare_two_judges(y_true, merged["pred_confidence_supported_ministral"].to_numpy(),
                        merged["confidence_supported"].to_numpy(), label_a="Ministral-8B", label_b="NLI")
    compare_two_judges(y_true, merged["pred_confidence_supported_llama"].to_numpy(),
                        merged["confidence_supported"].to_numpy(), label_a="Llama-3.1-8B", label_b="NLI")

    print(f"\n{'=' * 60}\nReference: compare the two LLM judges\n{'=' * 60}")
    compare_two_judges(y_true, merged["pred_confidence_supported_ministral"].to_numpy(),
                        merged["pred_confidence_supported_llama"].to_numpy(),
                        label_a="Ministral-8B", label_b="Llama-3.1-8B")

    return merged



def main():
    parser = argparse.ArgumentParser(description='Compare Ministral, Llama, and the NLI baseline.')
    parser.add_argument("--ministral-predictions", type=str, default='data/results/rq2/predictions_Ministral-8B.csv')
    parser.add_argument("--llama-predictions", type=str, default='data/results/rq2/predictions_Llama-3.1-8B.csv')
    parser.add_argument("--sentence-answer-key", type=str, default='data/stimuli/internal_answer_key_sentences.csv')
    parser.add_argument("--recovered-nli-scores", type=str, default='data/results/rq2/recovered_nli_scores.csv',
                         help="Output CSV from recover_missing_nli_scores.py. If provided, sentences absent from the "
                              "answer key can be included. By default, only inferred_negative_control is used "
                              "to avoid conflicting with the FAIL_INVALID_REF zero-score handling.")
    parser.add_argument("--include-recovery-methods", type=str, default="inferred_negative_control",
                         help="Comma-separated recovery_method values to include from --recovered-nli-scores. "
                              "Default: 'inferred_negative_control' only. Adding 'recovered_from_tag_residue' "
                              "replaces automatic zero scores for FAIL_INVALID_REF with recovered scores; use with "
                              "caution because the two approaches differ. 'all' includes every method in the CSV.")
    parser.add_argument("--core-set-cases", type=str, default=None,
                         help="Path to core_set_cases.csv saved by few_shot_judge_experiment.py. If provided, "
                              "the script reports stratified AUROC for n_raters=5 (original core-set) versus "
                              "n_raters=1 (expanded subset) as a diagnostic when using --min-raters 1.")
    parser.add_argument("--label-mode", type=str, default="strict", choices=["strict", "lenient"],
                         help="'strict' (default): only majority_tag=='supported' is positive. 'lenient': both "
                              "supported and partial are positive, matching the lenient AUROC definition used by "
                              "few_shot_judge_experiment.py. The script reports both modes on the same sample; "
                              "--label-mode selects which one is used for the primary comparison.")
    args = parser.parse_args()


    include_methods = None if args.include_recovery_methods.strip().lower() == "all" else \
        [m.strip() for m in args.include_recovery_methods.split(",") if m.strip()]

    run_three_way_comparison(args.ministral_predictions, args.llama_predictions, args.sentence_answer_key,
                              recovered_nli_scores_csv=args.recovered_nli_scores,
                              include_recovery_methods=include_methods,
                              core_set_cases_csv=args.core_set_cases, label_mode=args.label_mode)


if __name__ == "__main__":
    main()
