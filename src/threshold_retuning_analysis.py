
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
            "sentence_level_summary가 비어 있습니다. responses-dir에 level1_tags가 채워진 "
            "응답이 있는지, sentence-answer-key의 display_id/sentence_id가 응답의 "
            "item_id/sentence_id와 실제로 겹치는지 확인하세요."
        )
    if "auto_status" not in sentence_summary.columns or sentence_summary["auto_status"].isna().all():
        raise RuntimeError(
            "auto_status가 비어 있습니다. --sentence-answer-key 경로와 컬럼명"
            "(display_id, sentence_id, auto_status, auto_entailment, cited_indicators)을 확인하세요."
        )

    if answer_key_path:
        key_df = ar.load_answer_key(answer_key_path)
        sentence_summary = sentence_summary.merge(
            key_df[["in_core_set"]], left_on="item_id", right_index=True, how="left"
        )
        n_missing = sentence_summary["in_core_set"].isna().sum()
        if n_missing:
            print(f"[Warning] {n_missing} sentences are missing in_core_set from the answer key "
                  "and will be excluded from --core-only analysis.")

    return sentence_summary



def _confusion_metrics(pred_positive: np.ndarray, truth_positive: np.ndarray) -> dict:
    pred_positive = np.asarray(pred_positive, dtype=bool)
    truth_positive = np.asarray(truth_positive, dtype=bool)
    tp = int(np.sum(pred_positive & truth_positive))
    fp = int(np.sum(pred_positive & ~truth_positive))
    fn = int(np.sum(~pred_positive & truth_positive))
    tn = int(np.sum(~pred_positive & ~truth_positive))
    n = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)
    accuracy = (tp + tn) / n if n > 0 else None
    youden_j = (recall + specificity - 1) if recall is not None and specificity is not None else None

    overflag_rate_truth_cond = fn / (tp + fn) if (tp + fn) > 0 else None
    underdetect_rate_truth_cond = fp / (fp + tn) if (fp + tn) > 0 else None

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
            raise ValueError("--core-only를 쓰려면 --answer-key도 함께 지정해야 합니다.")
        df = df[df["in_core_set"] == True]
    n_unknown = int((df["majority_tag"] == "unknown").sum())
    df = df[df["majority_tag"] != "unknown"].copy()
    df["truth_positive"] = df["majority_tag"].isin(GROUND_TRUTH_DEFS[gt_key])
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
        return {"error": "entailment로 채점된 문장이 없습니다."}
    x = governed["auto_entailment"].to_numpy(dtype=float)
    y = governed["truth_positive"].to_numpy(dtype=float)
    n = len(x)
    result = {"n": n, "auc": _auc_mann_whitney(x, governed["truth_positive"].to_numpy())}
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
        return {"error": "entailment로 채점된 문장이 없습니다."}
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
            result["caveat"] = ("기대빈도 5 미만 셀이 있어 카이제곱 근사가 부정확할 수 있습니다 — "
                                 "Fisher's exact test 사용을 권장합니다 (scipy.stats.fisher_exact).")
    elif table.shape == (2, 2):
        result["note"] = "scipy 미설치 — chi2_contingency를 계산하려면 scipy를 설치하세요."
    return result



def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{digits}f}"


def format_report(gt_key: str, df: pd.DataFrame, sweep_df: pd.DataFrame,
                   whatif_df: pd.DataFrame, best: dict, conv: dict, chi2: dict,
                   core_only: bool) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"MetaKT-Verba Stage 3 entailment_threshold 재튜닝 분석"
                  f" — ground truth: {gt_key}" + (" (core set만)" if core_only else ""))
    lines.append("=" * 70)
    n_gov = int(df["is_threshold_governed"].sum())
    n_total = len(df)
    lines.append(f"분석 대상 문장: 전체 {n_total}개 중 entailment로 채점된 문장 {n_gov}개 "
                 f"(나머지 {n_total - n_gov}개는 FAIL_NO_CITATION/FAIL_INVALID_REF/FAIL_HEDGE — "
                 f"threshold와 무관하게 판정 고정)")
    lines.append(f"판단불가(unknown) 제외 건수: {int(df['_n_unknown_excluded'].iloc[0]) if n_total else 0}")
    lines.append("")

    lines.append("[1] 연속 점수 자체의 신호력 (threshold와 무관)")
    if "error" in conv:
        lines.append(f"  - {conv['error']}")
    else:
        lines.append(f"  - AUROC = {_fmt(conv['auc'])} (0.5=무작위, 1.0=완벽 분리), n={conv['n']}")
        lines.append(f"  - point-biserial r = {_fmt(conv.get('point_biserial_r'))}"
                     + (f" (p={_fmt(conv.get('point_biserial_p'))})" if conv.get('point_biserial_p') is not None else ""))
    lines.append("")

    lines.append(f"[2] 현재 threshold({CURRENT_THRESHOLD})에서의 2x2 연관성 검정")
    if "error" in chi2:
        lines.append(f"  - {chi2['error']}")
    elif "chi2" in chi2:
        lines.append(f"  - chi2 = {_fmt(chi2['chi2'])}, p = {_fmt(chi2['p_value'], 4)}, "
                     f"Cramer's V = {_fmt(chi2['cramers_v'])}")
        if "caveat" in chi2:
            lines.append(f"  - ⚠ {chi2['caveat']}")
    else:
        lines.append(f"  - {chi2.get('note', '표 형태가 2x2가 아니어서 카이제곱을 계산하지 않았습니다.')}")
    lines.append("")

    cur = best["current_threshold"]
    bf1 = best["best_by_f1"]
    bj = best["best_by_youden_j"]
    lines.append("[3] Threshold 비교 (entailment로 채점된 문장만 대상, 표본 n 고정)")
    if cur:
        lines.append(f"  - 현재(t={cur['threshold']:.2f}): precision={_fmt(cur['precision'])}, "
                     f"recall={_fmt(cur['recall'])}, specificity={_fmt(cur['specificity'])}, F1={_fmt(cur['f1'])}")
        lines.append(f"    · FAIL 판정 중 실제로는 충실했던 비율(과잉플래그, aggregate_ratings.py [4-나]의 "
                     f"33%/43%와 동일 정의) = {_fmt(cur['fail_is_actually_ok_rate'])}")
        lines.append(f"    · 충실한 문장 중 억울하게 FAIL된 비율(1-recall) = {_fmt(cur['overflag_rate_truth_cond'])}")
    if bf1:
        lines.append(f"  - F1 최적(t={bf1['threshold']:.2f}): precision={_fmt(bf1['precision'])}, "
                     f"recall={_fmt(bf1['recall'])}, F1={_fmt(bf1['f1'])}, "
                     f"FAIL 판정 중 억울한 비율={_fmt(bf1['fail_is_actually_ok_rate'])}")
    if bj:
        lines.append(f"  - Youden's J 최적(t={bj['threshold']:.2f}): recall={_fmt(bj['recall'])}, "
                     f"specificity={_fmt(bj['specificity'])}, J={_fmt(bj['youden_j'])}")
    lines.append("  - 전체 grid는 threshold_sweep.csv 참고 (논문 부록/Ablation용 표로 바로 사용 가능)")
    lines.append("")

    if cur and bf1:
        lines.append(f"[4] 해석 메모")
        lines.append(f"  - F1 기준 최적 threshold로 옮기면 'FAIL 판정 중 억울했던 비율'이 "
                     f"{_fmt(cur['fail_is_actually_ok_rate'])} → {_fmt(bf1['fail_is_actually_ok_rate'])}로, "
                     f"'PASS 판정 중 실제론 문제였던 비율'은 {_fmt(cur['pass_is_actually_bad_rate'])} → "
                     f"{_fmt(bf1['pass_is_actually_bad_rate'])}로 변화합니다 — 두 방향의 트레이드오프를 "
                     f"함께 보고 threshold를 정해야 합니다 (한쪽만 줄이면 다른 쪽이 늘어남).")
        lines.append(f"  - AUROC가 0.5에 가깝다면 threshold를 어디로 옮겨도 근본적 개선은 어렵습니다 "
                     f"— 그 경우 문제는 threshold가 아니라 premise 설계/모델 선택입니다 "
                     f"(_fact_to_premise_text의 축약 premise 문제, 3장 근본원인 가설 2번 참고).")
    lines.append("")

    lines.append("[5] Corpus-wide what-if (전체 문장 기준, 표 4.5 faithfulness_rate 참고용)")
    for _, row in whatif_df[whatif_df["threshold"].isin(
            sorted({CURRENT_THRESHOLD, bf1["threshold"] if bf1 else CURRENT_THRESHOLD}))].iterrows():
        tag = " (현재)" if row["is_current_default"] else " (F1 최적)"
        lines.append(f"  - t={row['threshold']:.2f}{tag}: corpus faithfulness_rate="
                     f"{_fmt(row['corpus_faithfulness_rate_if_this_threshold'])}, "
                     f"accuracy_vs_expert={_fmt(row['accuracy'])}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)



def run(responses_dir, sentence_answer_key, answer_key, output_dir, core_only,
        thresholds=DEFAULT_THRESHOLD_GRID):
    os.makedirs(output_dir, exist_ok=True)
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
    print(f"\n✅ Analysis complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run the analysis.')
    parser.add_argument("--responses-dir", type=str, default='data/ratings')
    parser.add_argument("--sentence-answer-key", type=str, default='data/stimuli/internal_answer_key_sentences.csv',
                         help="internal_answer_key_sentences.csv 경로 (display_id, sentence_id, "
                              "auto_status, auto_entailment 컬럼 필요)")
    parser.add_argument("--answer-key", type=str, default='data/stimuli/internal_answer_key.csv',
                         help="internal_answer_key.csv path (optional; required for --core-only)")
    parser.add_argument("--output-dir", type=str, default="outputs/threshold_retuning")
    parser.add_argument("--core-only", action="store_true",
                         help="core set(전원 공통 평정, n≥3) 문장만 대상으로 분석 — "
                              "coverage set의 단일 평정자 '다수결' 잡음을 배제한 강건성 체크")
    args = parser.parse_args()


    if not args.responses_dir or not args.sentence_answer_key:
        parser.error("--responses-dir and --sentence-answer-key are required")

    run(args.responses_dir, args.sentence_answer_key, args.answer_key,
        args.output_dir, args.core_only)




if __name__ == "__main__":
    main()