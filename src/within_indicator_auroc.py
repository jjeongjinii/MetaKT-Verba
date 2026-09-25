
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["majority_tag"].notna() & (df["majority_tag"] != "unknown")]
    d = d[d["auto_entailment"].notna() & (d["auto_status"] != "FAIL_HEDGE")]
    cited = d["cited_indicators"].astype(str).str.strip()
    d = d[~cited.str.contains(r"[,;|]")].assign(indicator=cited)

    def row(name, g):
        su = g[g["majority_tag"].isin(["supported", "unsupported"])]
        y = (su["majority_tag"] == "supported").astype(int)
        auc = roc_auc_score(y, su["auto_entailment"]) if y.nunique() == 2 else np.nan
        return {"indicator": name, "n": len(g), "mean_entailment": g["auto_entailment"].mean(),
                "auroc": auc, "n_auroc": len(su), "n_supported": int(y.sum()),
                "n_unsupported": int((1 - y).sum())}

    rows = [row(ind, g) for ind, g in d.groupby("indicator")]
    table = pd.DataFrame(rows).sort_values("mean_entailment").reset_index(drop=True)
    pooled = row("pooled", d)
    pooled["mean_entailment"] = np.nan
    return pd.concat([table, pd.DataFrame([pooled])], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description='Run the analysis.')
    parser.add_argument("--sentence-level-csv", default="outputs/threshold_retuning/sentence_level_with_scores.csv")
    parser.add_argument("--output", default="outputs/table5_within_indicator.csv")
    args = parser.parse_args()
    table = build_table(pd.read_csv(args.sentence_level_csv))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.round(3).to_string(index=False))
    inds = table[table["indicator"] != "pooled"]
    print(f"\nmean entailment range: {inds['mean_entailment'].min():.3f}-{inds['mean_entailment'].max():.3f} "
          f"({inds['mean_entailment'].max() / inds['mean_entailment'].min():.1f}x); "
          f"within-indicator AUROC: {inds['auroc'].min():.3f}-{inds['auroc'].max():.3f}")




if __name__ == "__main__":
    main()
