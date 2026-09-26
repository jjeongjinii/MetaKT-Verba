from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def _bootstrap_indices(n, rng, cluster_ids=None):
    if cluster_ids is None:
        return rng.randint(0, n, size=n)
    cluster_ids = np.asarray(cluster_ids)
    if len(cluster_ids) != n:
        raise ValueError(f'cluster_ids length {len(cluster_ids)} != n {n}')
    clusters = pd.unique(cluster_ids)
    sampled_clusters = clusters[rng.randint(0, len(clusters), size=len(clusters))]
    parts = [np.flatnonzero(cluster_ids == c) for c in sampled_clusters]
    return np.concatenate(parts)

def bootstrap_auroc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int=2000, ci: float=0.95, seed: int=42, cluster_ids=None) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
    point = roc_auc_score(y_true, y_score)
    boot_aucs = []
    for _ in range(n_boot):
        idx = _bootstrap_indices(n, rng, cluster_ids=cluster_ids)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, ys))
    alpha = 1 - ci
    lo, hi = np.percentile(boot_aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, lo, hi)

def bootstrap_paired_auroc_diff_test(y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray, n_boot: int=2000, seed: int=42, cluster_ids=None) -> dict:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
    n = len(y_true)
    point_a = roc_auc_score(y_true, y_score_a)
    point_b = roc_auc_score(y_true, y_score_b)
    point_diff = point_a - point_b
    diffs = []
    for _ in range(n_boot):
        idx = _bootstrap_indices(n, rng, cluster_ids=cluster_ids)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_a = roc_auc_score(yt, y_score_a[idx])
        auc_b = roc_auc_score(yt, y_score_b[idx])
        diffs.append(auc_a - auc_b)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_value = min(p_value, 1.0)
    return {'auroc_a': point_a, 'auroc_b': point_b, 'diff': point_diff, 'diff_ci_lo': lo, 'diff_ci_hi': hi, 'p_value': p_value, 'significant': not lo <= 0 <= hi}

def compare_two_judges(y_true, y_score_ministral, y_score_llama, label_a='Ministral-8B', label_b='Llama-3.1-8B', n_boot: int=2000) -> None:
    print(f"{'=' * 60}\n{label_a} vs {label_b} - AUROC comparison (n={len(y_true)})\n{'=' * 60}")
    pa, lo_a, hi_a = bootstrap_auroc_ci(y_true, y_score_ministral, n_boot=n_boot)
    pb, lo_b, hi_b = bootstrap_auroc_ci(y_true, y_score_llama, n_boot=n_boot)
    print(f'{label_a}: AUROC = {pa:.4f}  (95% CI [{lo_a:.4f}, {hi_a:.4f}])')
    print(f'{label_b}: AUROC = {pb:.4f}  (95% CI [{lo_b:.4f}, {hi_b:.4f}])')
    result = bootstrap_paired_auroc_diff_test(y_true, y_score_ministral, y_score_llama, n_boot=n_boot)
    verdict = 'significant (p<0.05)' if result['significant'] else 'not significant'
    print(f"\nDifference (delta={label_a}-{label_b}) = {result['diff']:+.4f}  (95% CI [{result['diff_ci_lo']:+.4f}, {result['diff_ci_hi']:+.4f}], p≈{result['p_value']:.4f})")
    print(f'-> {verdict}')
    if not result['significant'] and len(y_true) < 150:
        print(f'\n[Note] Sample size n={len(y_true)} is small, so confidence intervals may be wide.')
