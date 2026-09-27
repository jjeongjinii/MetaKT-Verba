import os
import argparse

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

import aggregate_ratings as ar
from compare_auroc import bootstrap_auroc_ci

CURRENT_THRESHOLD = 0.5

DEFAULT_THRESHOLD_GRID = tuple(round(t, 2) for t in np.arange(0.05, 0.96, 0.05))

THRESHOLD_INVARIANT_STATUSES = {"FAIL_NO_CITATION", "FAIL_INVALID_REF", "FAIL_HEDGE"}

PASS_LIKE_STATUSES = {"PASS", "PASS_NO_SIGNAL", "PASS_INFERRED"}

GROUND_TRUTH_DEFS = {
    "strict": {"supported"},
    "lenient": {"supported", "partial"},
}

def load_sentence_level_with_scores(responses_dir: str, sentence_answer_key_path: str,
                                     answer_key_path: str = None) -> pd.DataFrame:
    resp_df = ar.load_responses(responses_dir)
    sentence_key_df = ar.load_sentence_answer_key(sentence_answer_key_path)
    sentence_summary = ar.build_sentence_level_summary(resp_df, sentence_key_df)

    if sentence_summary.empty:
        raise RuntimeError(
            "Sentence-level summary is empty. Check rating files and sentence-key IDs."
        )
    if "auto_status" not in sentence_summary.columns or sentence_summary["auto_status"].isna().all():
        raise RuntimeError(
            "auto_status is missing. Check the sentence answer key and required columns."
        )

    if answer_key_path:
        key_df = ar.load_answer_key(answer_key_path)
        sentence_summary = sentence_summary.merge(
            key_df[["in_core_set"]], left_on="item_id", right_index=True, how="left"
        )
        n_missing = sentence_summary["in_core_set"].isna().sum()
        if n_missing:
            print(
                f"[warning] {n_missing} sentence(s) have no in_core_set match "
                "and will be excluded from --core-only."
            )

    return sentence_summary

def load_sentence_level_from_verifier_scores(
    verifier_scores_path: str,
    verifier: str = "mDeBERTa-v3-base-mnli-xnli",
    premise: str = "fact",
) -> pd.DataFrame:
    if verifier_scores_path.endswith((".pkl", ".pickle")):
        df = pd.read_pickle(verifier_scores_path)
    else:
        df = pd.read_csv(verifier_scores_path)

    required = {
        "item_id", "sentence_id", "majority_tag",
        "is_negative_control", "score", "verifier", "premise",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required verifier-score columns: {sorted(missing)}"
        )

    df = df[
        (df["verifier"] == verifier)
        & (df["premise"] == premise)
        & (~df["is_negative_control"].astype(bool))
        & (df["majority_tag"] != "unknown")
        & df["score"].notna()
    ].copy()

    dup = df.duplicated(["item_id", "sentence_id"])
    if dup.any():
        raise RuntimeError(
            f"Found {dup.sum()} duplicate sentence keys."
        )

    if len(df) != 287:
        raise RuntimeError(
            f"Expected 287 sentences in the broader analysis set, found {len(df)}."
        )

    df["auto_entailment"] = df["score"].astype(float)

    df["auto_status"] = np.where(
        df["auto_entailment"] >= CURRENT_THRESHOLD,
        "PASS",
        "FAIL_NLI",
    )

    df["is_threshold_governed"] = True

    print(
        f"[287-set] verifier={verifier}, premise={premise}, "
        f"analysis sentences={len(df)}"
    )

    return df

def _confusion_metrics(pred_positive: np.ndarray, truth_positive: np.ndarray) -> dict:
    pred_positive = np.asarray(pred_positive, dtype=bool)
    truth_positive = np.asarray(truth_positive, dtype=bool)
    tp = int(np.sum(pred_positive & truth_positive))
    fp = int(np.sum(pred_positive & ~truth_positive))
    fn = int(np.sum(~pred_positive & truth_positive))
    tn = int(np.sum(~pred_positive & ~truth_positive))
    n = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None          # = sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)
    accuracy = (tp + tn) / n if n > 0 else None
    youden_j = (recall + specificity - 1) if recall is not None and specificity is not None else None

    overflag_rate_truth_cond = fn / (tp + fn) if (tp + fn) > 0 else None   # = 1-recall
    underdetect_rate_truth_cond = fp / (fp + tn) if (fp + tn) > 0 else None  # = 1-specificity

    fail_is_actually_ok_rate = fn / (fn + tn) if (fn + tn) > 0 else None
    pass_is_actually_bad_rate = fp / (fp + tp) if (fp + tp) > 0 else None

    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1, "accuracy": accuracy, "youden_j": youden_j,
        "overflag_rate_truth_cond": overflag_rate_truth_cond,
        "underdetect_rate_truth_cond": underdetect_rate_truth_cond,
        "fail_is_actually_ok_rate": fail_is_actually_ok_rate,
        "pass_is_actually_bad_rate": pass_is_actually_bad_rate,
    }

def _auc_mann_whitney(scores: np.ndarray, truth_positive: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    truth_positive = np.asarray(truth_positive, dtype=bool)
    n_pos = int(truth_positive.sum())
    n_neg = int((~truth_positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    sum_ranks_pos = ranks[truth_positive].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))

def prepare_binary_frame(sentence_summary: pd.DataFrame, gt_key: str,
                          core_only: bool = False) -> pd.DataFrame:
    df = sentence_summary.dropna(subset=["majority_tag", "auto_status"]).copy()
    if core_only:
        if "in_core_set" not in df.columns:
            raise ValueError("--core-only requires --answer-key.")
        df = df[df["in_core_set"] == True]  # noqa: E712
    n_unknown = int((df["majority_tag"] == "unknown").sum())
    df = df[df["majority_tag"] != "unknown"].copy()
    df["truth_positive"] = df["majority_tag"].isin(GROUND_TRUTH_DEFS[gt_key])

    if "is_threshold_governed" not in df.columns:
        df["is_threshold_governed"] = df["auto_entailment"].notna() & (df["auto_status"] != "FAIL_HEDGE")

    df["_n_unknown_excluded"] = n_unknown
    return df

def threshold_sweep(df: pd.DataFrame, thresholds=DEFAULT_THRESHOLD_GRID) -> pd.DataFrame:
    governed = df[df["is_threshold_governed"]].copy()
    rows = []
    grid = sorted(set(list(thresholds) + [CURRENT_THRESHOLD]))
    for t in grid:
        pred_positive = governed["auto_entailment"].to_numpy(dtype=float) >= t
        m = _confusion_metrics(pred_positive, governed["truth_positive"].to_numpy())
        m["threshold"] = t
        m["is_current_default"] = np.isclose(t, CURRENT_THRESHOLD)
        rows.append(m)
    out = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return out

def corpus_whatif_sweep(df: pd.DataFrame, thresholds=DEFAULT_THRESHOLD_GRID) -> pd.DataFrame:
    fixed_pred_positive = df["auto_status"].isin(PASS_LIKE_STATUSES).to_numpy()
    governed_mask = df["is_threshold_governed"].to_numpy()
    entail = df["auto_entailment"].to_numpy(dtype=float)
    truth = df["truth_positive"].to_numpy()

    rows = []
    grid = sorted(set(list(thresholds) + [CURRENT_THRESHOLD]))
    for t in grid:
        pred_positive = fixed_pred_positive.copy()
        pred_positive[governed_mask] = entail[governed_mask] >= t
        m = _confusion_metrics(pred_positive, truth)
        m["threshold"] = t
        m["is_current_default"] = np.isclose(t, CURRENT_THRESHOLD)
        m["corpus_faithfulness_rate_if_this_threshold"] = pred_positive.mean()
        rows.append(m)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

def summarize_best_thresholds(sweep_df: pd.DataFrame) -> dict:
    valid = sweep_df.dropna(subset=["f1"])
    best_f1_row = valid.loc[valid["f1"].idxmax()] if not valid.empty else None
    valid_j = sweep_df.dropna(subset=["youden_j"])
    best_j_row = valid_j.loc[valid_j["youden_j"].idxmax()] if not valid_j.empty else None
    current_row = sweep_df[sweep_df["is_current_default"]]
    current_row = current_row.iloc[0] if not current_row.empty else None
    return {
        "current_threshold": None if current_row is None else current_row.to_dict(),
        "best_by_f1": None if best_f1_row is None else best_f1_row.to_dict(),
        "best_by_youden_j": None if best_j_row is None else best_j_row.to_dict(),
    }

def convergence_stats(df: pd.DataFrame) -> dict:
    governed = df[df["is_threshold_governed"]].dropna(subset=["auto_entailment"])
    if governed.empty:
        return {"error": "No entailment-scored sentences are available."}
    x = governed["auto_entailment"].to_numpy(dtype=float)
    y = governed["truth_positive"].to_numpy(dtype=float)
    n = len(x)

    truth = governed["truth_positive"].astype(int).to_numpy()

    auc_point, auc_lo, auc_hi = bootstrap_auroc_ci(truth, x, n_boot=2000, seed=42)

    result = {
        "n": n,
        "auc": auc_point,
        "auc_ci_lo": auc_lo,
        "auc_ci_hi": auc_hi,
    }
    if n >= 3 and np.std(x) > 0 and np.std(y) > 0:
        if _HAS_SCIPY:
            r, p = scipy_stats.pointbiserialr(governed["truth_positive"].to_numpy(), x)
            result["point_biserial_r"] = float(r)
            result["point_biserial_p"] = float(p)
        else:
            result["point_biserial_r"] = float(np.corrcoef(x, y)[0, 1])
            result["point_biserial_p"] = None
    else:
        result["point_biserial_r"] = None
        result["point_biserial_p"] = None
    return result

def chi_square_at_current_threshold(df: pd.DataFrame) -> dict:
    governed = df[df["is_threshold_governed"]].copy()
    if governed.empty:
        return {"error": "No entailment-scored sentences are available."}
    pred_positive = governed["auto_entailment"].to_numpy(dtype=float) >= CURRENT_THRESHOLD
    table = pd.crosstab(pred_positive, governed["truth_positive"].to_numpy())
    result = {"table": table.to_dict()}
    if _HAS_SCIPY and table.shape == (2, 2):
        chi2, p, dof, expected = scipy_stats.chi2_contingency(table.to_numpy(), correction=True)
        n = table.to_numpy().sum()
        phi2 = chi2 / n
        r, k = table.shape
        cramers_v = float(np.sqrt(phi2 / min(k - 1, r - 1))) if min(k - 1, r - 1) > 0 else float("nan")
        result.update({"chi2": float(chi2), "p_value": float(p), "cramers_v": cramers_v,
                        "min_expected_count": float(expected.min())})
        if expected.min() < 5:
            result["caveat"] = ("At least one expected cell count is below 5; consider Fisher's exact test.")
    elif table.shape == (2, 2):
        result["note"] = "SciPy is not installed; chi-square statistics are unavailable."
    return result

def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{digits}f}"

def format_report(
    gt_key: str,
    df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    whatif_df: pd.DataFrame,
    best: dict,
    conv: dict,
    chi2: dict,
    core_only: bool,
) -> str:
    lines = []
    scope = "core set" if core_only else "all eligible sentences"
    n_total = len(df)
    n_governed = int(df["is_threshold_governed"].sum())
    n_unknown = int(df["_n_unknown_excluded"].iloc[0]) if n_total else 0

    lines.append(f"Ground truth: {gt_key} ({scope})")
    lines.append(
        f"n_total={n_total}, n_threshold_governed={n_governed}, "
        f"n_unknown_excluded={n_unknown}"
    )

    if "error" in conv:
        lines.append(f"AUROC: {conv['error']}")
    else:
        lines.append(
            f"AUROC={_fmt(conv['auc'])} "
            f"[95% CI {_fmt(conv.get('auc_ci_lo'))}, {_fmt(conv.get('auc_ci_hi'))}], "
            f"n={conv['n']}"
        )
        line = f"point_biserial_r={_fmt(conv.get('point_biserial_r'))}"
        if conv.get("point_biserial_p") is not None:
            line += f", p={_fmt(conv.get('point_biserial_p'), 4)}"
        lines.append(line)

    if "chi2" in chi2:
        lines.append(
            f"chi2={_fmt(chi2['chi2'])}, p={_fmt(chi2['p_value'], 4)}, "
            f"CramersV={_fmt(chi2['cramers_v'])}"
        )
        if "caveat" in chi2:
            lines.append(f"chi_square_caveat={chi2['caveat']}")
    elif "note" in chi2:
        lines.append(f"chi_square_note={chi2['note']}")

    current = best["current_threshold"]
    best_f1 = best["best_by_f1"]
    best_j = best["best_by_youden_j"]

    if current:
        lines.append(
            f"current_threshold={current['threshold']:.2f}, "
            f"precision={_fmt(current['precision'])}, "
            f"recall={_fmt(current['recall'])}, "
            f"specificity={_fmt(current['specificity'])}, "
            f"f1={_fmt(current['f1'])}"
        )
    if best_f1:
        lines.append(
            f"best_f1_threshold={best_f1['threshold']:.2f}, "
            f"precision={_fmt(best_f1['precision'])}, "
            f"recall={_fmt(best_f1['recall'])}, "
            f"f1={_fmt(best_f1['f1'])}"
        )
    if best_j:
        lines.append(
            f"best_youden_threshold={best_j['threshold']:.2f}, "
            f"recall={_fmt(best_j['recall'])}, "
            f"specificity={_fmt(best_j['specificity'])}, "
            f"youden_j={_fmt(best_j['youden_j'])}"
        )

    selected = {CURRENT_THRESHOLD}
    if best_f1:
        selected.add(best_f1["threshold"])
    selected_rows = whatif_df[whatif_df["threshold"].isin(sorted(selected))]
    for _, row in selected_rows.iterrows():
        lines.append(
            f"whatif_threshold={row['threshold']:.2f}, "
            f"corpus_faithfulness_rate="
            f"{_fmt(row['corpus_faithfulness_rate_if_this_threshold'])}, "
            f"accuracy_vs_expert={_fmt(row['accuracy'])}"
        )

    return "\n".join(lines)

def run(responses_dir, sentence_answer_key, answer_key, output_dir, core_only, thresholds=DEFAULT_THRESHOLD_GRID, 
    verifier_scores=None,
    verifier="mDeBERTa-v3-base-mnli-xnli",
    premise="fact",
):
    os.makedirs(output_dir, exist_ok=True)

    if verifier_scores:
        sentence_summary = load_sentence_level_from_verifier_scores(verifier_scores, verifier=verifier, premise=premise)

        if core_only:
            raise ValueError(
                "--core-only is not supported with --verifier-scores."
            )

    else:
        sentence_summary = load_sentence_level_with_scores(responses_dir, sentence_answer_key, answer_key)

    sentence_summary.to_csv(os.path.join(output_dir, "sentence_level_with_scores.csv"), index=False)

    full_report = []
    for gt_key in GROUND_TRUTH_DEFS:
        df = prepare_binary_frame(sentence_summary, gt_key, core_only=core_only)
        sweep_df = threshold_sweep(df, thresholds)
        whatif_df = corpus_whatif_sweep(df, thresholds)
        best = summarize_best_thresholds(sweep_df)
        conv = convergence_stats(df)
        chi2 = chi_square_at_current_threshold(df)

        suffix = f"_{gt_key}" + ("_core" if core_only else "")
        sweep_df.to_csv(os.path.join(output_dir, f"threshold_sweep{suffix}.csv"), index=False)
        whatif_df.to_csv(os.path.join(output_dir, f"corpus_whatif{suffix}.csv"), index=False)

        report = format_report(gt_key, df, sweep_df, whatif_df, best, conv, chi2, core_only)
        full_report.append(report)

    report_text = "\n\n".join(full_report)
    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\n" + report_text)
    print(f"Saved analysis outputs to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Threshold sensitivity analysis for Stage 3 entailment scores."
    )
    parser.add_argument("--responses-dir", default="data/ratings")
    parser.add_argument(
        "--sentence-answer-key",
        default="data/stimuli/internal_answer_key_sentences.csv",
        help="Sentence-level answer key with auto_status and auto_entailment.",
    )
    parser.add_argument(
        "--answer-key",
        default="data/stimuli/internal_answer_key.csv",
        help="Item-level answer key; required for --core-only in legacy mode.",
    )
    parser.add_argument(
        "--verifier-scores",
        default="data/results/verifier_family/scores_all_verifiers.csv",
        help=(
            "Verifier-family score CSV or PKL. When provided, use verifier scores "
            "instead of legacy auto_entailment values."
        ),
    )
    parser.add_argument(
        "--verifier",
        default="mDeBERTa-v3-base-mnli-xnli",
    )
    parser.add_argument(
        "--premise",
        default="fact",
        choices=["fact", "record"],
    )
    parser.add_argument("--output-dir", default="outputs/threshold_retuning")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Restrict the legacy analysis to the core set.",
    )
    args = parser.parse_args()

    if not args.verifier_scores and (
        not args.responses_dir or not args.sentence_answer_key
    ):
        parser.error(
            "Legacy mode requires --responses-dir and --sentence-answer-key."
        )

    run(
        args.responses_dir,
        args.sentence_answer_key,
        args.answer_key,
        args.output_dir,
        args.core_only,
        verifier_scores=args.verifier_scores,
        verifier=args.verifier,
        premise=args.premise,
    )

if __name__ == "__main__":
    main()
