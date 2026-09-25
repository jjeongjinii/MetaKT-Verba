


import argparse
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auroc_ci(y_true: np.ndarray, y_score: np.ndarray,
                        n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:



    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    point = roc_auc_score(y_true, y_score)

    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue  
        boot_aucs.append(roc_auc_score(yt, ys))

    alpha = 1 - ci
    lo, hi = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi


def bootstrap_paired_auroc_diff_test(
    y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray,
    n_boot: int = 2000, seed: int = 42,
) -> dict:


    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)
    n = len(y_true)

    point_a = roc_auc_score(y_true, y_score_a)
    point_b = roc_auc_score(y_true, y_score_b)
    point_diff = point_a - point_b

    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_a = roc_auc_score(yt, y_score_a[idx])
        auc_b = roc_auc_score(yt, y_score_b[idx])
        diffs.append(auc_a - auc_b)

    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_value = min(p_value, 1.0)

    return {
        "auroc_a": point_a, "auroc_b": point_b, "diff": point_diff,
        "diff_ci_lo": lo, "diff_ci_hi": hi, "p_value": p_value,
        "significant": not (lo <= 0 <= hi),
    }


def compare_two_judges(y_true, y_score_ministral, y_score_llama, label_a="Ministral-8B",
                        label_b="Llama-3.1-8B", n_boot: int = 2000) -> None:


    print(f"{'=' * 60}\n{label_a} vs {label_b} — AUROC 비교 (n={len(y_true)})\n{'=' * 60}")

    pa, lo_a, hi_a = bootstrap_auroc_ci(y_true, y_score_ministral, n_boot=n_boot)
    pb, lo_b, hi_b = bootstrap_auroc_ci(y_true, y_score_llama, n_boot=n_boot)
    print(f"{label_a}: AUROC = {pa:.4f}  (95% CI [{lo_a:.4f}, {hi_a:.4f}])")
    print(f"{label_b}: AUROC = {pb:.4f}  (95% CI [{lo_b:.4f}, {hi_b:.4f}])")

    result = bootstrap_paired_auroc_diff_test(y_true, y_score_ministral, y_score_llama, n_boot=n_boot)
    verdict = "유의함 (p<0.05)" if result["significant"] else "유의하지 않음 — 표본으로는 구별 안 됨"
    print(f"\n차이(Δ={label_a}-{label_b}) = {result['diff']:+.4f}  "
          f"(95% CI [{result['diff_ci_lo']:+.4f}, {result['diff_ci_hi']:+.4f}], p≈{result['p_value']:.4f})")
    print(f"-> {verdict}")

    if not result["significant"] and len(y_true) < 150:
        print(f"\n[참고] 표본이 n={len(y_true)}로 작아 신뢰구간이 넓을 수 있습니다 — "
              f"판정 실패나 데이터 누락으로 줄어든 표본을 늘리면(예: core-set 밖 문항까지 "
              f"확장, 또는 누락된 기준선 값을 실제로 채워 넣기) 검정력이 올라갈 수 있습니다.")


# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    args = parser.parse_args()


if __name__ == "__main__":
    main()