import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_bootstrap_ci import cluster_auroc_ci, cluster_paired_auroc_diff  # noqa: E402

VERIFIERS = ["mDeBERTa-v3-base-mnli-xnli", "DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
             "Bespoke-MiniCheck-7B", "Gemma-3-27B-it-logprob"]
# Reproduces n = 257 and strict AUROC 0.466 for mDeBERTa (fact premise) on the 287-sentence set.
AFFECT_LEXICON = r"frustrat|confus|bored|concentrat|emotion|feel|interest"
# Internal indicator names as they appear in sentence text (natural-language or snake_case forms).
INDICATOR_NAMES = (r"overconfiden|underconfiden|slipping|lucky[ _]guess|confidence[ _]triad|cognitive[ _]avoidance|"
                   r"productive[ _]struggle|unproductive[ _]frustration|strategic[ _]help|boredom[ _]offtask")
TYPOS = {"unsuppported": "unsupported", "unsuported": "unsupported", "suported": "supported"}
TRUE_VALUES = {"1", "true", "yes", "y", "t", "o", "counter", "contradicted", "1.0"}


def load_scores(path):
    s = pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)
    s["majority_tag"] = s["majority_tag"].astype(str).str.strip().str.lower().replace(
        {"unsuppported": "unsupported"})
    return s[(s["is_negative_control"] != True) & s["fact_premise"].notna()
             & (s["majority_tag"] != "unknown") & s["score"].notna()].copy()


def _ci(y, score, items, args):
    r = cluster_auroc_ci(y.astype(int), score, items, args.n_boot, args.seed)
    return {k: r[k] for k in ("n", "n_clusters", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi")}


def cmd_nonaffective(args):
    s = load_scores(args.scores)
    rows = []
    for (v, p), g in s.groupby(["verifier", "premise"]):
        if v not in args.verifiers:
            continue
        sub = g[~g["sentence_text"].str.contains(args.lexicon, case=False, regex=True)]
        for coding, pos in [("strict", {"supported"}), ("lenient", {"supported", "partial"})]:
            rows.append({"verifier": v, "premise": p, "coding": coding,
                         **_ci(sub["majority_tag"].isin(pos), sub["score"], sub["item_id"], args)})
    out = pd.DataFrame(rows)
    print(f"lexicon: {args.lexicon}")
    print(out.round(3).to_string(index=False))
    out.to_csv(os.path.join(args.out_dir, "nonaffective_auroc.csv"), index=False)


def cmd_controls(args):
    true = load_scores(args.scores)
    true = true[true["premise"] == "record"][["item_id", "sentence_id", "verifier", "score", "majority_tag", "is_core"]]
    ctl = []
    for f in sorted(glob.glob(args.controls)):
        m = re.search(r"(shuffle\d+|hyponly)", os.path.basename(f))
        if not m:
            continue
        d = pd.read_pickle(f) if f.endswith(".pkl") else pd.read_csv(f)
        d = d[d["premise"] == "record"][["item_id", "sentence_id", "verifier", "score"]].copy()
        d["verifier"] = d["verifier"].str.replace(r"-(shuffle\d+|hyponly)$", "", regex=True)
        d["variant"] = m.group(1)
        ctl.append(d)
    if not ctl:
        raise FileNotFoundError(args.controls)
    ctl = pd.concat(ctl, ignore_index=True)
    ctl["kind"] = np.where(ctl["variant"].str.startswith("shuffle"), "shuffled", "hyponly")
    wide = (ctl.groupby(["item_id", "sentence_id", "verifier", "kind"])["score"].mean()
            .unstack("kind").reset_index())
    nshuf = ctl[ctl.kind == "shuffled"].groupby("verifier")["variant"].nunique().to_dict()
    df = true.merge(wide, on=["item_id", "sentence_id", "verifier"], how="inner")

    rows = []
    for (v, scope), g0 in [((v, sc), g[g.is_core == True] if sc == "core" else g)
                           for v, g in df.groupby("verifier") for sc in ("core", "all")]:
        for coding, pos in [("strict", {"supported"}), ("lenient", {"supported", "partial"})]:
            for kind in ("shuffled", "hyponly"):
                if kind not in g0 or g0[kind].isna().all():
                    continue
                g = g0.dropna(subset=[kind])
                yy = g["majority_tag"].isin(pos).astype(int)
                r = cluster_paired_auroc_diff(yy, g["score"], g[kind], g["item_id"], args.n_boot, args.seed)
                c = cluster_auroc_ci(yy, g[kind], g["item_id"], args.n_boot, args.seed)
                rows.append({"verifier": v, "scope": scope, "coding": coding, "control": kind,
                             "n": r["n"], "auroc_true": r["auroc_a"], "auroc_control": r["auroc_b"],
                             "control_lo": c["ci_lo"], "control_hi": c["ci_hi"],
                             "delta": r["delta"], "delta_lo": r["ci_lo"], "delta_hi": r["ci_hi"],
                             "p": r["p_value"],
                             "n_shuffles": nshuf.get(v, 0) if kind == "shuffled" else None})
    out = pd.DataFrame(rows)
    print("auroc_control for 'shuffled' uses each sentence's mean score over the shuffles")
    print(out.round(3).to_string(index=False))
    out.to_csv(os.path.join(args.out_dir, "record_controls.csv"), index=False)


def cmd_restatement(args):
    from scipy.stats import fisher_exact
    s = load_scores(args.scores)
    d = s[(s["verifier"] == "mDeBERTa-v3-base-mnli-xnli") & (s["premise"] == "fact")].copy()
    d["names_label"] = d["sentence_text"].str.contains(args.pattern, case=False, regex=True)
    d["passed"] = d["score"] >= args.threshold          # Stage 3 PASS at the paper's threshold
    d["supported"] = d["majority_tag"] == "supported"
    g, o = d[d.names_label], d[~d.names_label]
    print(f"sentences naming an internal indicator: {len(g)}/{len(d)} ({100 * len(g) / len(d):.1f}%)")
    for col in ("passed", "supported"):
        a, b = int(g[col].sum()), int(o[col].sum())
        orr, p = fisher_exact([[a, len(g) - a], [b, len(o) - b]])
        print(f"{col}: {a}/{len(g)} ({100 * a / len(g):.1f}%) vs {b}/{len(o)} ({100 * b / len(o):.1f}%); "
              f"OR = {orr:.2f}, Fisher p = {p:.3f}")
    d[d.names_label][["item_id", "sentence_id", "sentence_text", "passed", "majority_tag"]].to_csv(
        os.path.join(args.out_dir, "restatement_sentences.csv"), index=False)


def load_rater_tags(pattern):
    """metakt_verba_ratings_R*.csv -> one row per (item_id, sentence_id, rater, tag)."""
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f, encoding="utf-8-sig")
        for _, r in d.iterrows():
            for kv in str(r["level1_tags"]).split(";"):
                if ":" in kv:
                    sid, tag = kv.split(":", 1)
                    tag = tag.strip().lower()
                    rows.append((r["item_id"], sid.strip().upper(), r["rater_id"], TYPOS.get(tag, tag)))
    return pd.DataFrame(rows, columns=["item_id", "sentence_id", "rater", "tag"]).drop_duplicates(
        ["item_id", "sentence_id", "rater"], keep="last")


def cmd_loo(args):
    """Each core rater's binary tag vs the majority binary tag of the other raters.
    'unknown' tags count as missing; equal splits among the others (2-2) are excluded.
    For a binary predictor, AUROC equals balanced accuracy."""
    s = load_scores(args.scores)
    core = s[s["is_core"] == True].drop_duplicates(["item_id", "sentence_id"])[["item_id", "sentence_id"]]
    tags = load_rater_tags(args.ratings).merge(core, on=["item_id", "sentence_id"])
    w = tags.pivot_table(index=["item_id", "sentence_id"], columns="rater", values="tag", aggfunc="first")
    print(f"core sentences: {len(w)} (paper: 83); raters: {list(w.columns)}")
    rows = []
    for coding, pos in [("strict", {"supported"}), ("lenient", {"supported", "partial"})]:
        b = w.apply(lambda c: c.isin(pos)).astype(float).where(w != "unknown")
        for r in w.columns:
            others = b.drop(columns=r)
            npos, n = others.sum(axis=1), others.notna().sum(axis=1)
            ref = np.where(npos > n / 2, 1, np.where(npos < n / 2, 0, -1))
            x = b[r].to_numpy()
            keep = (ref >= 0) & ~np.isnan(x)
            y, xx = ref[keep], x[keep]
            bal = ((xx[y == 1] == 1).mean() + (xx[y == 0] == 0).mean()) / 2
            rows.append({"coding": coding, "rater": r, "n": int(keep.sum()),
                         "ties_excluded": int((ref < 0).sum()), "auroc": bal})
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    for coding, g in out.groupby("coding"):
        print(f"{coding}: {g.auroc.min():.2f}-{g.auroc.max():.2f} (mean {g.auroc.mean():.2f}); "
              f"ties excluded {g.ties_excluded.min()}-{g.ties_excluded.max()}")
    out.to_csv(os.path.join(args.out_dir, "loo_human_auroc.csv"), index=False)


def cmd_split(args):
    s = load_scores(args.scores)
    c = (pd.read_excel(args.coding, sheet_name="Coding") if args.coding.endswith(".xlsx")
         else pd.read_csv(args.coding))
    flag = c[args.flag_col].astype(str).str.strip().str.lower().isin(TRUE_VALUES)
    c = c.assign(counter_evidence=flag)[["item_id", "sentence_id", "counter_evidence"]]
    base = s.drop_duplicates(["item_id", "sentence_id"])
    uns = base[base["majority_tag"] == "unsupported"].merge(c, on=["item_id", "sentence_id"], how="left")
    print(f"unsupported in analysis set: {len(uns)} (paper: 130); coded: {int(uns.counter_evidence.notna().sum())}; "
          f"contradicted: {int(uns.counter_evidence.fillna(False).sum())} (paper: 71)")
    # Section 5.4.2 check: counter-evidence rate among unsupported sentences Stage 3 passed vs failed
    md = s[(s["verifier"] == "mDeBERTa-v3-base-mnli-xnli") & (s["premise"] == "fact")][["item_id", "sentence_id", "score"]]
    chk = uns.drop(columns=["score"], errors="ignore").merge(md, on=["item_id", "sentence_id"]).dropna(subset=["counter_evidence"])
    if len(chk):
        from scipy.stats import fisher_exact
        ps, fl = chk[chk.score >= .5], chk[chk.score < .5]
        a, b = int(ps.counter_evidence.sum()), int(fl.counter_evidence.sum())
        p = fisher_exact([[a, len(ps) - a], [b, len(fl) - b]])[1]
        print(f"5.4.2 check: passed {a}/{len(ps)}, failed {b}/{len(fl)}, Fisher p = {p:.2f} (paper: 23/40, 48/90, p = .71)")
    if uns["counter_evidence"].isna().any():
        print("WARNING: some unsupported sentences have no coding; they are dropped from both splits")
    s = s.merge(c, on=["item_id", "sentence_id"], how="left")
    s["cls"] = s["majority_tag"]
    s.loc[(s.majority_tag == "unsupported") & (s.counter_evidence == True), "cls"] = "contradicted"
    s.loc[(s.majority_tag == "unsupported") & (s.counter_evidence == False), "cls"] = "unrelated"

    rows = []
    for (v, p), g in s.groupby(["verifier", "premise"]):
        if v not in args.verifiers:
            continue
        for coding, pos in [("strict", {"supported"}), ("lenient", {"supported", "partial"})]:
            for neg in ("contradicted", "unrelated"):
                sub = g[g["cls"].isin(pos | {neg})]
                rows.append({"verifier": v, "premise": p, "coding": coding, "negatives": neg,
                             **_ci(sub["cls"].isin(pos), sub["score"], sub["item_id"], args)})
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    out.to_csv(os.path.join(args.out_dir, "unsupported_split_auroc.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/data/results/review")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("nonaffective")
    p.add_argument("--scores", default='/data/results/verifier_family/scores_with_logprob.pkl')
    p.add_argument("--lexicon", default=AFFECT_LEXICON)
    p.add_argument("--verifiers", nargs="+", default=VERIFIERS)
    p.set_defaults(func=cmd_nonaffective)


    p = sub.add_parser("record-controls")
    p.add_argument("--scores", default='/data/results/verifier_family/scores_with_logprob.pkl', help="true-record scores (scores_with_logprob.pkl)")
    p.add_argument("--controls", default="data/results/record_controls/scores_*.pkl")
    p.set_defaults(func=cmd_controls)

    p = sub.add_parser("restatement")
    p.add_argument("--scores", default='/data/results/verifier_family/scores_with_logprob.pkl')
    p.add_argument("--pattern", default=INDICATOR_NAMES)
    p.add_argument("--threshold", type=float, default=0.5)
    p.set_defaults(func=cmd_restatement)


    p = sub.add_parser("loo-human")
    p.add_argument("--scores", default='/data/results/verifier_family/scores_with_logprob.pkl')
    p.add_argument("--ratings", default="/data/ratings/metakt_verba_ratings_R*.csv")
    p.set_defaults(func=cmd_loo)

    p = sub.add_parser("unsupported-split")
    p.add_argument("--scores", default='/data/results/verifier_family/scores_with_logprob.pkl')
    p.add_argument("--coding", default='/data/annotations/counter_evidence_coding.csv', help="CSV with item_id, sentence_id and a counter-evidence flag")
    p.add_argument("--flag-col", default="counter_evidence")
    p.add_argument("--verifiers", nargs="+", default=VERIFIERS)
    p.set_defaults(func=cmd_split)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
