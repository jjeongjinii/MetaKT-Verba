"""Item-cluster bootstrap CIs for MetaKT-Verba.

Replaces the sentence-level i.i.d. bootstrap in compare_auroc.py for the analyses below.
Resampling unit:
  - sentence-level AUROC  -> narrative item (display item_id, e.g. Q042); all sentences of a
                             drawn item are kept, an item drawn k times contributes k copies
  - RQ1 condition deltas  -> interaction (ITEST_id, step_idx), or student (ITEST_id)

Subcommands
  nli        Table 3 sentence AUROC (mDeBERTa, fact premise) + Sec. 5.4.3 CA-exclusion
  verifiers  Table 5 (per verifier x premise AUROC) + Table A1 (paired record-fact, Holm)
  rq1        Table 2 Full vs. ablations, paired over the same interactions

Run from the repository root, e.g.
  python src/cluster_bootstrap_ci.py nli
  python src/cluster_bootstrap_ci.py verifiers --scores <verifier_scores.csv>
  python src/cluster_bootstrap_ci.py rq1
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def roc_auc_score(y, s):
    """Mann-Whitney AUROC with average ranks for ties (identical to sklearn's, ~20x faster)."""
    y = np.asarray(y)
    r = rankdata(np.asarray(s, dtype=float))
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

CODINGS = {"strict": {"supported"}, "lenient": {"supported", "partial"}}


# ─────────────────────────────── core bootstrap ───────────────────────────────

class ClusterSampler:
    """Draws n_clusters clusters with replacement and returns the pooled row indices."""

    def __init__(self, cluster_ids, seed: int):
        codes, uniq = pd.factorize(pd.Series(cluster_ids).astype(str).to_numpy())
        self.members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
        self.n_clusters = len(uniq)
        self.rng = np.random.RandomState(seed)

    def draw_clusters(self) -> np.ndarray:
        return self.rng.randint(0, self.n_clusters, size=self.n_clusters)

    def draw(self) -> np.ndarray:
        return np.concatenate([self.members[k] for k in self.draw_clusters()])


def _pct(vals, ci):
    a = (1 - ci) / 2 * 100
    lo, hi = np.percentile(vals, [a, 100 - a])
    return float(lo), float(hi)


def _two_sided_p(diffs):
    """Percentile-bootstrap two-sided p with +1 correction (never exactly 0; floor = 2/(B+1))."""
    diffs = np.asarray(diffs)
    b = len(diffs)
    lo_tail = ((diffs <= 0).sum() + 1) / (b + 1)
    hi_tail = ((diffs >= 0).sum() + 1) / (b + 1)
    return float(min(1.0, 2 * min(lo_tail, hi_tail)))


def cluster_auroc_ci(y, s, clusters, n_boot, seed, ci=0.95) -> dict:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    smp = ClusterSampler(clusters, seed)
    boots, skipped = [], 0
    for _ in range(n_boot):
        idx = smp.draw()
        if np.unique(y[idx]).size < 2:
            skipped += 1
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = _pct(boots, ci)
    return {"auroc": roc_auc_score(y, s), "ci_lo": lo, "ci_hi": hi, "n": len(y),
            "n_clusters": smp.n_clusters, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "boot_skipped": skipped}


def cluster_paired_auroc_diff(y, s_a, s_b, clusters, n_boot, seed, ci=0.95) -> dict:
    """Delta = AUROC(a) - AUROC(b); both scores evaluated on the same resampled clusters."""
    y = np.asarray(y, dtype=int)
    s_a = np.asarray(s_a, dtype=float)
    s_b = np.asarray(s_b, dtype=float)
    smp = ClusterSampler(clusters, seed)
    diffs, skipped = [], 0
    for _ in range(n_boot):
        idx = smp.draw()
        if np.unique(y[idx]).size < 2:
            skipped += 1
            continue
        diffs.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    lo, hi = _pct(diffs, ci)
    a, b = roc_auc_score(y, s_a), roc_auc_score(y, s_b)
    return {"auroc_a": a, "auroc_b": b, "delta": a - b, "ci_lo": lo, "ci_hi": hi,
            "p_value": _two_sided_p(diffs), "n": len(y), "n_clusters": smp.n_clusters,
            "boot_skipped": skipped}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# ─────────────────────────────── shared filtering ───────────────────────────────

def attach_answer_key(df: pd.DataFrame, answer_key: str) -> pd.DataFrame:
    """Adds in_core_set / is_negative_control using aggregate_ratings.load_answer_key."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import aggregate_ratings as ar
    key = ar.load_answer_key(answer_key)
    cols = [c for c in ("in_core_set", "is_negative_control") if c in key.columns]
    df = df.drop(columns=[c for c in cols if c in df.columns])
    return df.merge(key[cols], left_on="item_id", right_index=True, how="left")


def is_tied(dist) -> bool:
    """True if the top two label counts in aggregate_ratings' tag_distribution ('a=2;b=2;c=1') are equal."""
    if not isinstance(dist, str) or "=" not in dist:
        return False
    counts = sorted((int(kv.split("=")[1]) for kv in dist.split(";") if "=" in kv), reverse=True)
    return len(counts) > 1 and counts[0] == counts[1]


TAG_TYPOS = {"unsuppported": "unsupported", "unsuported": "unsupported", "suported": "supported"}


def filter_sentences(df, answer_key, keep_nc, core_only, exclude_ties=False):
    df = df.dropna(subset=["majority_tag"]).copy()
    df["majority_tag"] = df["majority_tag"].astype(str).str.strip().str.lower().replace(TAG_TYPOS)
    if "in_core_set" not in df.columns and "is_core" in df.columns:
        df["in_core_set"] = df["is_core"]
    if "tag_distribution" in df.columns:
        df["is_tied"] = df["tag_distribution"].map(is_tied)
    if exclude_ties:
        if "is_tied" not in df.columns:
            raise ValueError("--exclude-ties needs a tag_distribution column.")
        df = df[~df["is_tied"]]
    if answer_key and os.path.exists(answer_key):
        df = attach_answer_key(df, answer_key)
    if not keep_nc and "is_negative_control" in df.columns:
        df = df[df["is_negative_control"] != True]
    if core_only:
        if "in_core_set" not in df.columns:
            raise ValueError("core-set filtering needs in_core_set (pass --answer-key).")
        df = df[df["in_core_set"] == True]
    return df[df["majority_tag"] != "unknown"].copy()


def _fmt_ci(r, key="auroc"):
    return f"{r[key]:+.3f}" if key == "delta" else f"{r[key]:.3f}"


# ─────────────────────────────── nli: Table 3 + 5.4.3 ───────────────────────────────

def cmd_nli(args):
    df = pd.read_csv(args.sentence_level_csv)
    df = df.dropna(subset=["auto_status"])
    df = filter_sentences(df, args.answer_key, args.keep_negative_controls, args.core_only,
                          args.exclude_ties)
    # same eligibility as threshold_retuning_analysis.convergence_stats
    df = df[df["auto_entailment"].notna() & (df["auto_status"] != "FAIL_HEDGE")]

    cited = df["cited_indicators"].astype(str)
    subsets = {"all": df}
    for ind in args.exclude_cited:
        subsets[f"excl_{ind}"] = df[~cited.str.contains(ind, regex=False)]

    if "is_tied" in df.columns:
        print(f"[info] sentences with a tied modal tag in this set: {int(df['is_tied'].sum())}")
    rows = []
    for name, sub in subsets.items():
        for coding, pos in CODINGS.items():
            y = sub["majority_tag"].isin(pos).astype(int)
            r = cluster_auroc_ci(y, sub["auto_entailment"], sub["item_id"], args.n_boot, args.seed)
            rows.append({"subset": name, "coding": coding, **r})
    out = pd.DataFrame(rows)
    _save(out, args.out_dir, "nli_auroc_cluster" + ("_core" if args.core_only else "") + ".csv")
    print(out[["subset", "coding", "n", "n_clusters", "auroc", "ci_lo", "ci_hi", "boot_skipped"]]
          .round(3).to_string(index=False))


# ─────────────────────────────── verifiers: Table 5 + A1 ───────────────────────────────

def _to_wide(df, args):
    """Accepts wide '<verifier>__<premise>' columns, a custom --col-template, or long format
    (one row per sentence x verifier x premise with columns verifier / premise / score)."""
    names = dict(kv.split("=", 1) for kv in args.premise_names.split(",")) if args.premise_names else {}
    long_cols = [args.verifier_col, args.premise_col, args.score_col]
    if all(c in df.columns for c in long_cols):
        keys = [c for c in ("item_id", "sentence_id") if c in df.columns]
        meta = [c for c in df.columns if c not in long_cols + keys]
        inv = {v: k for k, v in names.items()}
        d = df.copy()
        d["_col"] = d[args.verifier_col].astype(str) + "__" + d[args.premise_col].astype(str).map(
            lambda x: inv.get(x, x))
        wide = d.pivot_table(index=keys, columns="_col", values=args.score_col, aggfunc="first")
        wide.columns = list(wide.columns)
        base = d.drop_duplicates(keys).set_index(keys)[meta]
        return base.join(wide).reset_index()
    if args.col_template != "{v}__{p}":
        rename = {}
        for v in args.verifiers.split(","):
            for p in ("fact", "record"):
                src = args.col_template.format(v=v.strip(), p=names.get(p, p))
                if src in df.columns:
                    rename[src] = f"{v.strip()}__{p}"
        df = df.rename(columns=rename)
    return df


def cmd_verifiers(args):
    raw = pd.read_pickle(args.scores) if args.scores.endswith(".pkl") else pd.read_csv(args.scores)
    need = {"item_id", "majority_tag"}
    if not need <= set(raw.columns):
        raise KeyError(f"{args.scores} lacks {sorted(need - set(raw.columns))}. Pass the per-sentence "
                       f"score rows (e.g. outputs/verifier_family/scores.pkl), not a summary table. "
                       f"Columns found: {list(raw.columns)[:15]}")
    df = _to_wide(raw.drop(columns=["steps"], errors="ignore"), args)
    df = filter_sentences(df, args.answer_key, args.keep_negative_controls, core_only=False,
                          exclude_ties=args.exclude_ties)
    if "in_core_set" not in df.columns:
        raise ValueError("in_core_set missing: pass --answer-key or include the column in --scores.")
    verifiers = [v.strip() for v in args.verifiers.split(",") if v.strip()]
    missing = [f"{v}__{p}" for v in verifiers for p in ("fact", "record") if f"{v}__{p}" not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing} in {args.scores}.\n"
                       f"Available columns: {list(df.columns)}\n"
                       "Use --verifiers / --col-template / --premise-names, or long format "
                       "(--verifier-col / --premise-col / --score-col).")

    scopes = {"core": df[df["in_core_set"] == True], "all": df}

    # Table 5: sentence set "shared" = both premises scored (paper's paired n=83 / 287);
    #          "own" = each premise on every sentence where it is available (record n=87 / 299)
    t5 = []
    for scope, sub in scopes.items():
        for v in verifiers:
            shared = sub.dropna(subset=[f"{v}__fact", f"{v}__record"])
            for p in ("fact", "record"):
                for sset, s in (("shared", shared), ("own", sub.dropna(subset=[f"{v}__{p}"]))):
                    if sset == "own" and len(s) == len(shared):
                        continue
                    for coding, pos in CODINGS.items():
                        y = s["majority_tag"].isin(pos).astype(int)
                        r = cluster_auroc_ci(y, s[f"{v}__{p}"], s["item_id"], args.n_boot, args.seed)
                        t5.append({"scope": scope, "set": sset, "verifier": v, "premise": p,
                                   "coding": coding, **r})
    t5 = pd.DataFrame(t5)
    _save(t5, args.out_dir, "table5_verifier_auroc_cluster.csv")
    print("\n[Table 5] paper table = scope core, set shared; set own = all available sentences")
    print(t5[["scope", "set", "verifier", "premise", "coding", "n", "n_clusters", "auroc", "ci_lo", "ci_hi"]]
          .round(3).to_string(index=False))

    # Table A1: paired record - fact on the shared sentence set, Holm within each panel
    a1 = []
    for scope, sub in scopes.items():
        panel = []
        for v in verifiers:
            s = sub.dropna(subset=[f"{v}__fact", f"{v}__record"])
            for coding, pos in CODINGS.items():
                y = s["majority_tag"].isin(pos).astype(int)
                r = cluster_paired_auroc_diff(y, s[f"{v}__record"], s[f"{v}__fact"], s["item_id"],
                                              args.n_boot, args.seed)
                panel.append({"scope": scope, "verifier": v, "coding": coding, **r})
        if args.holm_family == "panel":
            for r, ph in zip(panel, holm([r["p_value"] for r in panel])):
                r["p_holm"] = ph
        a1.extend(panel)
    if args.holm_family == "all":      # paper: one family over both analysis sets
        for r, ph in zip(a1, holm([r["p_value"] for r in a1])):
            r["p_holm"] = ph
    a1 = pd.DataFrame(a1)
    _save(a1, args.out_dir, "tableA1_paired_record_minus_fact_cluster.csv")
    print(f"\n[Table A1] delta = record - fact; Holm family = {args.holm_family}")
    print(a1[["scope", "verifier", "coding", "n", "n_clusters", "delta", "ci_lo", "ci_hi",
              "p_value", "p_holm"]].round(3).to_string(index=False))


# ─────────────────────────────── rq1: Table 2 ───────────────────────────────

def _rq1_parts(g: pd.DataFrame, metric: str, agg: str):
    """Per-interaction numerator/denominator so that any resample reproduces the aggregate."""
    if metric == "F":
        rate, w = g["faithfulness_rate"], g["n_sentences"]
    elif metric == "cF":
        rate, w = g["conditional_faithfulness_rate"], g["n_content_eligible"]
    else:  # SVR
        rate, w = g["structural_violation_rate"], g["n_sentences"]
    valid = rate.notna() & w.notna()
    if agg == "pooled":   # ratio of sums over sentences
        num = (rate.fillna(0) * w.fillna(0)).where(valid, 0.0)
        den = w.fillna(0).where(valid, 0.0)
    else:                 # mean of per-interaction rates
        num = rate.fillna(0).where(valid, 0.0)
        den = valid.astype(float)
    return num.to_numpy(float), den.to_numpy(float)


def cmd_rq1(args):
    paths = sorted(glob.glob(args.summaries))
    if not paths:
        raise FileNotFoundError(args.summaries)
    comps = [c.strip() for c in args.compare.split(",") if c.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    all_rows = []

    for path in paths:
        model = os.path.basename(path).replace("pipeline_summary_", "").replace(".csv", "")
        d = pd.read_csv(path)
        d["_key"] = d["ITEST_id"].astype(str) + "_" + d["step_idx"].astype(str)
        conds = [args.reference] + comps
        by_cond = {c: d[d["condition"] == c].drop_duplicates("_key").set_index("_key") for c in conds}
        keys = sorted(set.intersection(*[set(g.index) for g in by_cond.values()]))
        for c in conds:
            dropped = len(by_cond[c]) - len(keys)
            if dropped:
                print(f"[{model}] {c}: {dropped} interaction(s) not present in all conditions; dropped")
        by_cond = {c: g.loc[keys] for c, g in by_cond.items()}
        clusters = (by_cond[args.reference]["ITEST_id"].astype(str).to_numpy()
                    if args.cluster == "student" else np.array(keys))

        for metric in metrics:
            parts = {c: _rq1_parts(by_cond[c], metric, args.agg) for c in conds}

            def stat(c, idx):
                num, den = parts[c]
                return num[idx].sum() / den[idx].sum() if den[idx].sum() > 0 else np.nan

            full_idx = np.arange(len(keys))
            ref_point = stat(args.reference, full_idx)
            if np.isnan(ref_point):
                continue
            rows = []
            for c in comps:
                pt = stat(c, full_idx)
                if np.isnan(pt):
                    continue
                smp = ClusterSampler(clusters, args.seed)
                diffs = []
                for _ in range(args.n_boot):
                    idx = smp.draw()
                    dlt = stat(args.reference, idx) - stat(c, idx)
                    if not np.isnan(dlt):
                        diffs.append(dlt)
                lo, hi = _pct(diffs, 0.95)
                rows.append({"model": model, "metric": metric, "agg": args.agg,
                             "reference": args.reference, "compare": c,
                             "ref_pct": 100 * ref_point, "cmp_pct": 100 * pt,
                             "delta_pp": 100 * (ref_point - pt), "ci_lo_pp": 100 * lo,
                             "ci_hi_pp": 100 * hi, "p_value": _two_sided_p(diffs),
                             "n_interactions": len(keys), "n_clusters": smp.n_clusters})
            if rows:
                ph = holm([r["p_value"] for r in rows])
                for r, p in zip(rows, ph):
                    r["p_holm"] = p
            all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    _save(out, args.out_dir, f"rq1_paired_{args.agg}_{args.cluster}.csv")
    print(out.round(3).to_string(index=False))


# ─────────────────────────────── cli ───────────────────────────────

def _save(df, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, name), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/elicer/project/metaKT-verba/outputs/cluster_bootstrap")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("nli", help="Table 3 sentence AUROC + Sec. 5.4.3 CA exclusion")
    p.add_argument("--sentence-level-csv", default="/home/elicer/project/metaKT-verba/calibration_stimuli_out1/threshold_retuning_out/sentence_level_with_scores.csv")
    p.add_argument("--answer-key", default="/home/elicer/project/metaKT-verba/calibration_stimuli_out1/internal_answer_key.csv")
    p.add_argument("--keep-negative-controls", action="store_true")
    p.add_argument("--core-only", action="store_true")
    p.add_argument("--exclude-cited", nargs="*", default=["cognitive_avoidance"])
    p.add_argument("--exclude-ties", action="store_true", help="sensitivity: drop tied modal tags")
    p.set_defaults(func=cmd_nli)

    p = sub.add_parser("verifiers", help="Table 5 + Table A1")
    p.add_argument("--scores", default='/home/elicer/project/metaKT-verba/outputs/verifier_family/table5_within_indicator_by_verifier_premise.csv',
                   help="CSV: item_id, sentence_id, majority_tag, and <verifier>__fact / <verifier>__record")
    p.add_argument("--verifiers", default="mdeberta_base,deberta_large,minicheck_7b,gemma_27b")
    p.add_argument("--holm-family", choices=["all", "panel"], default="all",
                   help="all = one Holm family over both analysis sets (paper); panel = per set")
    p.add_argument("--col-template", default="{v}__{p}",
                   help="wide-format column pattern, e.g. '{v}_{p}' or 'score_{v}_{p}'")
    p.add_argument("--premise-names", default="",
                   help="map internal names to file names, e.g. 'fact=symbolic,record=record'")
    p.add_argument("--verifier-col", default="verifier")
    p.add_argument("--premise-col", default="premise")
    p.add_argument("--score-col", default="score")
    p.add_argument("--exclude-ties", action="store_true", help="sensitivity: drop tied modal tags")
    p.add_argument("--answer-key", default="/home/elicer/project/metaKT-verba/calibration_stimuli_out1/internal_answer_key.csv")
    p.add_argument("--keep-negative-controls", action="store_true")
    p.set_defaults(func=cmd_verifiers)

    p = sub.add_parser("rq1", help="Table 2 Full vs. ablations (paired interaction bootstrap)")
    p.add_argument("--summaries", default="/home/elicer/project/metaKT-verba/outputs/pipeline_summary_*.csv")
    p.add_argument("--reference", default="full")
    p.add_argument("--compare", default="ablate_stage1,ablate_citation,naive")
    p.add_argument("--metrics", default="F,cF")
    p.add_argument("--agg", choices=["pooled", "mean"], default="mean",
                   help="mean = mean of per-interaction rates (what run_pipeline.py reports for Table 2); "
                        "pooled = sentence-weighted ratio of sums")
    p.add_argument("--cluster", choices=["interaction", "student"], default="interaction")
    p.set_defaults(func=cmd_rq1)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()