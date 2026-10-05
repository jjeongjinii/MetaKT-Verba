import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_bootstrap_ci import cluster_auroc_ci, roc_auc_score  # noqa: E402

VERIFIERS = ["mDeBERTa-v3-base-mnli-xnli", "DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
             "Bespoke-MiniCheck-7B", "Gemma-3-27B-it-logprob"]
CATS = ["unfaithful", "partial", "faithful"]


def resolve(path):
    """Accept paths relative to the current directory or to this script's directory."""
    if os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"{path} not found (looked in {os.getcwd()} and next to this script)")


def read_labels(path, id_name):
    path = resolve(path)
    with open(path, "rb") as fh:
        head = fh.read(4)
    if path.lower().endswith(".csv"):
        x = pd.read_csv(path)
    elif head[:2] == b"PK":                     # real .xlsx (zip container)
        x = pd.read_excel(path, sheet_name="Labels", engine="openpyxl")
    else:
        raise ValueError(
            f"{path} is not a valid .xlsx file ({os.path.getsize(path)} bytes, starts with {head!r}). "
            "It may be an Excel lock file (~$...), an empty download, or a sheet re-saved in another "
            "format. Re-save it from Excel as .xlsx, or export the Labels sheet as CSV and pass that.")
    x = x.rename(columns={"ID": id_name, "Label": "label", "Note": "note"})
    x = x[x[id_name].notna()]                       # ignore stray rows below the data
    x["label"] = x["label"].astype(str).str.strip().str.lower().replace({"nan": np.nan, "": np.nan})
    bad = x["label"].notna() & ~x["label"].isin(CATS + ["not_assessable"])
    if bad.any():
        raise ValueError(f"unexpected labels: {x.loc[bad, 'label'].unique()}")
    return x[[id_name, "label", "note"]]


def kappa(a, b, quadratic=False):
    k = len(CATS)
    idx = {c: i for i, c in enumerate(CATS)}
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    w = (np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
         if quadratic else 1 - np.eye(k))
    exp = np.outer(m.sum(1), m.sum(0)) / m.sum()
    return 1 - (w * m).sum() / (w * exp).sum()


def fragility(y, score, items, n_boot, seed, max_flips=40):
    """Fewest adversarial label flips that bring the AUROC CI lower bound to <= 0.5.
    Each step flips the single label (faithful <-> not) that lowers the AUROC most."""
    y = np.asarray(y, int).copy()
    score = np.asarray(score, float)
    for k in range(0, max_flips + 1):
        r = cluster_auroc_ci(y, score, items, n_boot, seed)
        if r["ci_lo"] <= 0.5:
            return k, r
        best, best_auc = None, np.inf
        for i in range(len(y)):
            y[i] = 1 - y[i]
            if 0 < y.sum() < len(y):
                a = roc_auc_score(y, score)
                if a < best_auc:
                    best, best_auc = i, a
            y[i] = 1 - y[i]
        y[best] = 1 - y[best]
    return None, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="/data/annotations/fidelity_labels.xlsx")
    ap.add_argument("--key", default="/data/annotations/fidelity_key.csv")
    ap.add_argument("--scores", default='/data/results/verifier_family/scores_with_logprob.pkl', help="scores_with_logprob.pkl (long format)")
    ap.add_argument("--retest", default='/data/annotations/fidelity_retest.xlsx')
    ap.add_argument("--verifiers", default=",".join(VERIFIERS))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/data/results/fidelity_analysis")
    ap.add_argument("--fragility", action="store_true",
                    help="fewest adversarial label flips that make each fact-premise fidelity CI include 0.5")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    key = pd.read_csv(resolve(args.key))
    lab = read_labels(args.labels, "label_id").merge(key, on="label_id", how="left")
    if lab["label"].isna().any():
        ids = lab.loc[lab["label"].isna(), "label_id"].tolist()
        raise ValueError(f"{len(ids)} row(s) still unlabeled in {args.labels}: {ids}")
    n_na = int((lab["label"] == "not_assessable").sum())
    lab = lab[lab["label"] != "not_assessable"].copy()
    print(f"labeled {len(lab) + n_na}; not_assessable excluded {n_na}; analysed {len(lab)}")
    print("fidelity labels:", lab["label"].value_counts().reindex(CATS[::-1]).to_dict())

    sp = resolve(args.scores)
    s = pd.read_pickle(sp) if sp.endswith(".pkl") else pd.read_csv(sp)
    s["majority_tag"] = s["majority_tag"].astype(str).str.lower().replace({"unsuppported": "unsupported"})
    tags = s.drop_duplicates(["item_id", "sentence_id"])[["item_id", "sentence_id", "majority_tag"]]
    lab = lab.merge(tags, on=["item_id", "sentence_id"], how="left")

    # [1] fidelity x teacher-visible support
    ct = pd.crosstab(lab["label"], lab["majority_tag"]).reindex(
        index=CATS[::-1], columns=["supported", "partial", "unsupported"], fill_value=0)
    ct.to_csv(os.path.join(args.out_dir, "fidelity_x_support.csv"))
    print("\n[1] fidelity (rows) x teacher-visible support (columns)\n" + ct.to_string())
    f, uns = lab["label"].eq("faithful"), lab["majority_tag"].eq("unsupported")
    print(f"faithful but judged unsupported: {(f & uns).sum()}/{f.sum()} ({100 * (f & uns).sum() / max(f.sum(), 1):.1f}%)")

    # [2] verifier AUROC against fidelity
    rows = []
    for v in [x.strip() for x in args.verifiers.split(",") if x.strip()]:
        for premise in ("fact", "record"):
            sc = s[(s["verifier"] == v) & (s["premise"] == premise)][["item_id", "sentence_id", "score"]]
            m = lab.merge(sc, on=["item_id", "sentence_id"]).dropna(subset=["score"])
            for coding, pos in [("strict", {"faithful"}), ("lenient", {"faithful", "partial"})]:
                y = m["label"].isin(pos).astype(int)
                if len(m) == 0 or y.nunique() < 2:
                    continue
                r = cluster_auroc_ci(y, m["score"], m["item_id"], args.n_boot, args.seed)
                rows.append({"verifier": v, "premise": premise, "coding": coding, **r,
                             "tracks_fidelity": r["ci_lo"] > 0.5})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "fidelity_auroc.csv"), index=False)
    print("\n[2] verifier AUROC against FIDELITY labels (item-cluster 95% CI)")
    print(res[["verifier", "premise", "coding", "n", "n_pos", "auroc", "ci_lo", "ci_hi", "tracks_fidelity"]]
          .round(3).to_string(index=False))

    # [3] intra-rater reliability (retest)
    if args.retest:
        rt = read_labels(args.retest, "retest_id").merge(key[["retest_id", "label_id"]].dropna(), on="retest_id")
        both = rt.merge(read_labels(args.labels, "label_id"), on="label_id", suffixes=("_retest", "_main"))
        both = both[both["label_retest"].isin(CATS) & both["label_main"].isin(CATS)]
        print(f"\n[3] intra-rater, n = {len(both)}: agreement {(both.label_retest == both.label_main).mean():.2f}; "
              f"Cohen's kappa {kappa(both.label_main, both.label_retest):.2f}; "
              f"quadratic-weighted kappa {kappa(both.label_main, both.label_retest, True):.2f}")

    # [4] fragility of the fact-premise fidelity result to label errors
    if args.fragility:
        print("\n[4] fragility: fewest adversarial flips of the author's labels until the CI includes 0.5")
        for v in [x.strip() for x in args.verifiers.split(",") if x.strip()]:
            sc = s[(s["verifier"] == v) & (s["premise"] == "fact")][["item_id", "sentence_id", "score"]]
            m = lab.merge(sc, on=["item_id", "sentence_id"]).dropna(subset=["score"])
            for coding, pos in [("strict", {"faithful"}), ("lenient", {"faithful", "partial"})]:
                k, r = fragility(m["label"].isin(pos), m["score"], m["item_id"], 2000, args.seed)
                kk = f">{40}" if k is None else str(k)
                print(f"  {v[:28]:28s} {coding:7s} flips needed: {kk:>3s} of {len(m)} "
                      f"({'n/a' if k is None else f'{100 * k / len(m):.0f}%'}); AUROC then {r['auroc']:.3f} "
                      f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
