"""W4: rescore Full narratives with the same best-match procedure as the dagger conditions.

Full is normally scored by verify_narrative (trusting each sentence's citation tag), while
w/o Stage 1, w/o Citation and Naive are scored by verify_narrative_uncited (best-matching
HIGH/AMBIGUOUS/RAW fact, citation tags ignored). This script applies verify_narrative_uncited
to the stored Full sentences so all four conditions are compared under one procedure.

No generation is needed: sentences and Stage 1 facts are read from pipeline_results_<model>_en.json.
mDeBERTa-base only, so it runs on one GPU in minutes (CPU also works, slower).

Run from src/ (needs stage1_grounding, stage2_generation, stage3_verification,
rq1_recompute_from_json, cluster_bootstrap_ci on the path):
  python rq1_full_bestmatch.py --inputs "../data/results/rq1/pipeline_results_*_en.json"
"""

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage3_verification import NLIVerifier, summarize_results  # noqa: E402
from rq1_recompute_from_json import load_model, paired_bootstrap, _ok  # noqa: E402

FACT_RE = re.compile(r"^\[([^\]]+)\]\s+([A-Z_]+):?\s*(.*)$", re.S)


@dataclass
class Fact:                      # duck-types stage1_grounding.SymbolicFact for _fact_to_premise_text
    indicator: str
    status: str
    value: Optional[float] = None
    note: str = ""


@dataclass
class Sent:                      # duck-types stage2_generation.CitedSentence for verify_narrative_uncited
    text: str
    cited_indicators: list


def parse_fact_strings(strings):
    out = []
    for s in strings or []:
        m = FACT_RE.match(str(s))
        if not m:
            raise ValueError(f"unparsable fact string: {s[:120]!r}")
        note = m.group(3).strip()
        out.append(Fact(m.group(1), m.group(2), None, "" if note in ("", "None") else note))
    return out


def rescore(path, verifier, condition="full"):
    model = re.sub(r"^pipeline_results_|_en\.json$", "", os.path.basename(path))
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    rows = []
    todo = [r for r in results if r.get("condition") == condition and "summary" in r
            and "error" not in r and not r.get("truncated_in_thinking")]
    for k, r in enumerate(todo, 1):
        facts = parse_fact_strings(r.get("ground_truth_facts"))
        sents = [Sent(s["text"], s.get("cited") or []) for s in (r.get("sentences") or [])]
        if not sents:
            continue
        ver = verifier.verify_narrative_uncited(sents, facts, lang="en")
        summ = summarize_results(ver)
        rate = summ.get("faithfulness_rate")
        has_citable = any(f.status in ("HIGH", "AMBIGUOUS") for f in facts)
        rows.append({"model": model, "condition": f"{condition}_bestmatch",
                     "ITEST_id": r["ITEST_id"], "step_idx": r["step_idx"],
                     "error": False, "truncated": False, "has_citable": has_citable,
                     "n_sentences": len(ver), "n_content_eligible": len(ver),
                     "F": rate, "cF": rate})
        if k % 50 == 0:
            print(f"  [{model}] {k}/{len(todo)}")
    return model, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="/home/elicer/project/metaKT-verba/pipeline_results_Qwen3-8B_en.json")
    ap.add_argument("--out-dir", default="/home/elicer/project/metaKT-verba/outputs/rq1_full_bestmatch")
    ap.add_argument("--nli-model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise FileNotFoundError(args.inputs)
    os.makedirs(args.out_dir, exist_ok=True)
    verifier = NLIVerifier(model_name=args.nli_model, entailment_threshold=args.threshold)
    comps = ["full", "ablate_stage1", "ablate_citation", "naive"]
    pd.set_option("display.width", 250)

    for path in paths:
        model, bm = rescore(path, verifier)
        bm.to_csv(os.path.join(args.out_dir, f"full_bestmatch_{model}.csv"), index=False)
        df = pd.concat([load_model(path), bm], ignore_index=True)

        ok = _ok(df)
        summary = []
        for c in ["full_bestmatch"] + comps:
            g = ok[ok["condition"] == c]
            cit = g["has_citable"].fillna(False).astype(bool)
            w = g["n_sentences"]
            summary.append({"condition": c, "n": int(g["F"].notna().sum()),
                            "F_mean_400": 100 * g["F"].mean(),
                            "F_pooled_citable": 100 * (g.loc[cit, "F"] * w[cit]).sum() / w[cit & g["F"].notna()].sum()})
        print(f"\n=== {model}: F under each condition (full_bestmatch = Full rescored like the dagger rows)")
        print(pd.DataFrame(summary).round(2).to_string(index=False))

        for agg, cit in [("mean", False), ("pooled", True)]:
            boot = paired_bootstrap(df, "full_bestmatch", comps, ["F"], args.n_boot, args.seed,
                                    "interaction", agg, cit)
            tag = f"{agg}{'_citable' if cit else ''}"
            boot.to_csv(os.path.join(args.out_dir, f"paired_{model}_{tag}.csv"), index=False)
            print(f"\n[{model}] Full(best-match) - comparison, {tag} (pp)")
            print(boot[["compare", "ref_pct", "cmp_pct", "delta_pp", "ci_lo_pp", "ci_hi_pp", "p_holm"]]
                  .round(2).to_string(index=False))


if __name__ == "__main__":
    main()