"""TEMPORARY: recompute RQ1 (Table 2) on CPU from pipeline_results_<model>_en.json.

No model is loaded. Every number is rebuilt from what run_pipeline.py stored per interaction:
  - 'verification' : per-sentence Stage 3 status and NLI scores
  - 'sentences'    : per-sentence text and parsed citation tags
  - 'ground_truth_facts' / 'facts_shown_to_generator' : "[indicator] STATUS: note" strings

Outputs (out-dir):
  per_interaction.csv                         one row per model x condition x interaction
  consistency_check.csv                       recomputed vs. stored summary (should be all zero)
  table2_recomputed.csv                       Table 2 (+ pooled, citable-stratum, both SVR definitions)
  citation_audit.csv                          malformed / invalid / uncited tags
  rq1_paired_bootstrap_<agg>_<cluster>.csv    Full - ablation, paired bootstrap, Holm per model x metric

Run from the repository root (needs src/cluster_bootstrap_ci.py next to this file):
  python src/rq1_recompute_from_json.py --inputs "data/results/rq1/pipeline_results_*_en.json"
  python src/rq1_recompute_from_json.py --agg pooled          # sentence-weighted sensitivity
  python src/rq1_recompute_from_json.py --citable-only        # drop interactions without a HIGH/AMBIGUOUS fact
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_bootstrap_ci import ClusterSampler, _pct, _two_sided_p, holm  # noqa: E402

PASS_STATUSES = {"PASS", "PASS_INFERRED", "PASS_NO_SIGNAL"}
CITED_EVAL_CONDITIONS = {"full", "template"}   # scored by verify_narrative, not the best-match path
FACT_RE = re.compile(r"^\[([^\]]+)\]\s+([A-Z_]+)")
CLEAN_TAG_RE = re.compile(r"^[A-Za-z_]+$")
UNPARSED_FACTS = {}
WEIGHT_COL = {"F": "n_sentences", "cF": "n_content_eligible"}


def parse_facts(fact_strings):
    """'[overconfidence] HIGH: note' -> {'overconfidence': 'HIGH'}; unmatched strings are logged."""
    out = {}
    for s in fact_strings or []:
        m = FACT_RE.match(str(s))
        if m:
            out[m.group(1)] = m.group(2)
        else:
            key = str(s)[:90]
            UNPARSED_FACTS[key] = UNPARSED_FACTS.get(key, 0) + 1
    return out


def interaction_row(model, r):
    base = {"model": model, "condition": r["condition"], "ITEST_id": r["ITEST_id"],
            "step_idx": r["step_idx"], "error": "error" in r,
            "truncated": bool(r.get("truncated_in_thinking", False))}
    if base["error"] or base["truncated"] or "summary" not in r:
        return base

    stored = r["summary"]
    verif = r.get("verification") or []
    sents = r.get("sentences") or []
    gt = parse_facts(r.get("ground_truth_facts"))
    shown = parse_facts(r.get("facts_shown_to_generator")) or gt
    no_citable = not any(st in ("HIGH", "AMBIGUOUS") for st in gt.values())
    base["has_citable"] = not no_citable

    # Stage 3 metrics rebuilt from sentence statuses (mirrors stage3_verification.summarize_results)
    n_invref_stage3 = None
    if verif:
        st = [v["status"] for v in verif]
        n = len(st)
        n_pass = sum(s in PASS_STATUSES for s in st)
        n_nocit = sum(s == "FAIL_NO_CITATION" for s in st)
        n_invref_stage3 = sum(s == "FAIL_INVALID_REF" for s in st)
        elig = n - n_nocit
        base.update({"n_sentences": n, "F": n_pass / n,
                     "cF": n_pass / elig if elig > 0 else np.nan,
                     "SVR_stage3": n_nocit / n, "n_content_eligible": elig})

    # citation audit from parsed tags
    if r.get("citation_required") and sents:
        n = len(sents)
        tags = [s.get("cited") or [] for s in sents]
        n_uncited = sum(not t for t in tags)
        n_malformed = sum(any(not CLEAN_TAG_RE.match(x) for x in t) for t in tags)
        if r["condition"] in CITED_EVAL_CONDITIONS and n_invref_stage3 is not None and len(verif) == n:
            n_invalid = n_invref_stage3                      # Stage 3's own FAIL_INVALID_REF verdicts
        else:
            n_invalid = sum(bool(t) and not set(t) <= set(shown) for t in tags)
        base.update({"n_parsed_sentences": n, "n_malformed_tag": n_malformed,
                     "SVR_uncited": 0.0 if no_citable else n_uncited / n,      # Sec. 3.4 definition
                     "SVR_uncited_noexempt": n_uncited / n,                   # no no-signal exemption
                     "SVR_uncited_or_invalid": (n_uncited + n_invalid) / n})   # generator citation_check

    for k_new, k_old in [("F__stored", "faithfulness_rate"), ("cF__stored", "conditional_faithfulness_rate"),
                         ("SVR_stored", "structural_violation_rate")]:
        v = stored.get(k_old)
        base[k_new] = np.nan if v is None else v
    return base


def load_model(path):
    model = re.sub(r"^pipeline_results_|_en\.json$", "", os.path.basename(path))
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    return pd.DataFrame([interaction_row(model, r) for r in results])


def _ok(df):
    return df[~df["error"] & ~df["truncated"]]


def consistency(df):
    rows = []
    for (m, c), g in _ok(df).groupby(["model", "condition"]):
        row = {"model": m, "condition": c, "n": len(g)}
        for new in ("F", "cF"):
            old = f"{new}__stored"
            if new in g and old in g:
                d = (g[new] - g[old]).abs()
                row[f"{new}_mismatch"] = int(((d > 1e-9) | (g[new].isna() ^ g[old].isna())).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _col(g, name):
    return g[name] if name in g else pd.Series(np.nan, index=g.index)


def table2(df, conditions):
    out = []
    for (m, c), g in _ok(df).groupby(["model", "condition"]):
        if c not in conditions:
            continue
        cit = _col(g, "has_citable").fillna(False).astype(bool)
        w = g["n_sentences"]
        out.append({"model": m, "condition": c, "n_interactions": int(g["F"].notna().sum()),
                    "F_mean": 100 * g["F"].mean(), "cF_mean": 100 * g["cF"].mean(),
                    "n_cF_defined": int(g["cF"].notna().sum()),
                    "F_pooled": 100 * (g["F"] * w).sum() / w[g["F"].notna()].sum(),
                    "SVR_stored": 100 * g["SVR_stored"].mean(),
                    "SVR_uncited": 100 * _col(g, "SVR_uncited").mean(),
                    "SVR_uncited_noexempt": 100 * _col(g, "SVR_uncited_noexempt").mean(),
                    "SVR_uncited_or_invalid": 100 * _col(g, "SVR_uncited_or_invalid").mean(),
                    "n_citable": int(cit.sum()),
                    "F_mean_citable": 100 * g.loc[cit, "F"].mean(),
                    "F_mean_no_citable": 100 * g.loc[~cit, "F"].mean()})
    return pd.DataFrame(out)


def _parts(frame, metric, agg):
    v = frame[metric].to_numpy(float)
    ok = np.isfinite(v)
    if agg == "mean":
        return np.where(ok, v, 0.0), ok.astype(float)
    w = frame[WEIGHT_COL[metric]].to_numpy(float)
    ok &= np.isfinite(w)
    return np.where(ok, v * w, 0.0), np.where(ok, w, 0.0)


def paired_bootstrap(df, reference, comps, metrics, n_boot, seed, cluster, agg, citable_only=False):
    ok = _ok(df).copy()
    if citable_only:   # has_citable depends only on the interaction, so all conditions keep the same set
        ok = ok[ok["has_citable"] == True]
    ok["_key"] = ok["ITEST_id"].astype(str) + "_" + ok["step_idx"].astype(str)
    rows = []
    for model, gm in ok.groupby("model"):
        keys = sorted(gm["_key"].unique())          # union: point estimates equal Table 2 exactly
        wide = {c: gm[gm["condition"] == c].drop_duplicates("_key").set_index("_key").reindex(keys)
                for c in [reference] + comps}
        cl = (gm.drop_duplicates("_key").set_index("_key").reindex(keys)["ITEST_id"].astype(str).to_numpy()
              if cluster == "student" else np.array(keys))
        for metric in metrics:
            if metric not in gm:
                continue
            rn, rd = _parts(wide[reference], metric, agg)
            block = []
            for c in comps:
                cn, cd = _parts(wide[c], metric, agg)
                if rd.sum() == 0 or cd.sum() == 0:
                    continue
                smp = ClusterSampler(cl, seed)
                diffs = []
                for _ in range(n_boot):
                    idx = smp.draw()
                    if rd[idx].sum() == 0 or cd[idx].sum() == 0:
                        continue
                    diffs.append(rn[idx].sum() / rd[idx].sum() - cn[idx].sum() / cd[idx].sum())
                lo, hi = _pct(diffs, 0.95)
                ref_pt, cmp_pt = rn.sum() / rd.sum(), cn.sum() / cd.sum()
                block.append({"model": model, "metric": metric, "agg": agg, "reference": reference,
                              "compare": c, "ref_pct": 100 * ref_pt, "cmp_pct": 100 * cmp_pt,
                              "delta_pp": 100 * (ref_pt - cmp_pt), "ci_lo_pp": 100 * lo,
                              "ci_hi_pp": 100 * hi, "p_value": _two_sided_p(diffs),
                              "n_ref": int(np.isfinite(wide[reference][metric].to_numpy(float)).sum()),
                              "n_cmp": int(np.isfinite(wide[c][metric].to_numpy(float)).sum()),
                              "n_clusters": smp.n_clusters})
            for r, p in zip(block, holm([b["p_value"] for b in block])):
                r["p_holm"] = p
            rows.extend(block)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="/home/jin/project/metaKT-verba/pipeline_results_Qwen3-8B_en.json")
    ap.add_argument("--out-dir", default="outputs/rq1_recompute")
    ap.add_argument("--reference", default="full")
    ap.add_argument("--compare", default="ablate_stage1,ablate_citation,naive")
    ap.add_argument("--metrics", default="F,cF")
    ap.add_argument("--cluster", choices=["interaction", "student"], default="interaction")
    ap.add_argument("--agg", choices=["mean", "pooled"], default="mean",
                    help="mean = per-interaction rates averaged (Table 2); pooled = sentence-weighted")
    ap.add_argument("--citable-only", action="store_true",
                    help="bootstrap only interactions with at least one HIGH/AMBIGUOUS Stage 1 fact")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise FileNotFoundError(args.inputs)
    df = pd.concat([load_model(p) for p in paths], ignore_index=True)
    os.makedirs(args.out_dir, exist_ok=True)
    comps = [c.strip() for c in args.compare.split(",") if c.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    pd.set_option("display.width", 250)

    print("\n[0] Rows per model x condition")
    print(df.groupby(["model", "condition"]).agg(rows=("error", "size"), errors=("error", "sum"),
                                                  truncated=("truncated", "sum")).to_string())

    cons = consistency(df)
    cons.to_csv(os.path.join(args.out_dir, "consistency_check.csv"), index=False)
    print("\n[1] Recomputed vs stored per-interaction metrics (all mismatch columns should be 0)")
    print(cons.to_string(index=False))

    t2 = table2(df, [args.reference] + comps + ["template"])
    t2.to_csv(os.path.join(args.out_dir, "table2_recomputed.csv"), index=False)
    print("\n[2] Table 2 recomputed (values in %; *_mean = run_pipeline.py aggregation)")
    print(t2.round(2).to_string(index=False))

    if "n_parsed_sentences" in df:
        audit = (df[df["n_parsed_sentences"].notna()].groupby(["model", "condition"])
                 .agg(interactions=("n_parsed_sentences", "size"), sentences=("n_parsed_sentences", "sum"),
                      malformed_tag_sentences=("n_malformed_tag", "sum"))
                 .reset_index())
        audit.to_csv(os.path.join(args.out_dir, "citation_audit.csv"), index=False)
        print("\n[3] Citation audit (a malformed tag can swallow the next sentence in the parser)")
        print(audit.to_string(index=False))

    if UNPARSED_FACTS:
        print("\n[3b] WARNING: fact strings not matched by FACT_RE (has_citable / invalid may be wrong):")
        for k, v in sorted(UNPARSED_FACTS.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {v:5d}x  {k}")

    boot = paired_bootstrap(df, args.reference, comps, metrics, args.n_boot, args.seed, args.cluster, args.agg,
                            args.citable_only)
    tag = "_citable" if args.citable_only else ""
    boot.to_csv(os.path.join(args.out_dir, f"rq1_paired_bootstrap_{args.agg}_{args.cluster}{tag}.csv"),
                index=False)
    print(f"\n[4] Paired bootstrap ({args.agg}{tag}) over {args.cluster}s: {args.reference} - comparison (pp)")
    print(boot.round(3).to_string(index=False))

    df.to_csv(os.path.join(args.out_dir, "per_interaction.csv"), index=False)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()