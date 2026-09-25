


import argparse

import numpy as np
import pandas as pd

from metakt_indicators import load_and_merge_data, preprocess_meta_kt

HIGH_THRESHOLD = 0.5
DISPUTED_HINT_RANGE = (1, 8)


def audit(raw_df: pd.DataFrame) -> dict:
    raw_df = raw_df.copy()
    raw_df["_hint_raw"] = pd.to_numeric(raw_df["hintCount"], errors="coerce").fillna(0)
    df = preprocess_meta_kt(raw_df)          # scales hintCount in place, keeps _hint_raw

    conf, h = df["RES_CONFUSED"], df["hintCount"]
    high = df[df["m_cognitive_avoidance"] > HIGH_THRESHOLD]
    intended = high[high["RES_CONFUSED"] > high["hintCount"]]
    reversed_ = high[high["RES_CONFUSED"] < high["hintCount"]]

    lo, hi = DISPUTED_HINT_RANGE
    disputed = df[df["_hint_raw"].between(lo, hi) & (conf == 0)]

    out = {
        "n_interactions": len(df),
        "n_high": len(high),
        "share_intended": len(intended) / len(high) if len(high) else float("nan"),
        "n_reversed": len(reversed_),
        "reversed_raw_hints_min": float(reversed_["_hint_raw"].min()) if len(reversed_) else None,
        "reversed_raw_hints_max": float(reversed_["_hint_raw"].max()) if len(reversed_) else None,
        "reversed_raw_hints_mean": float(reversed_["_hint_raw"].mean()) if len(reversed_) else None,
        "n_disputed_range_zero_confusion": len(disputed),
        "n_disputed_range_zero_confusion_high": int((disputed["m_cognitive_avoidance"] > HIGH_THRESHOLD).sum()),
    }
    return out


def format_report(r: dict) -> str:
    lines = ["=" * 64, "cognitive_avoidance full-log audit", "=" * 64,
             f"interactions:                    {r['n_interactions']:,}",
             f"HIGH (> {HIGH_THRESHOLD}):                     {r['n_high']:,}",
             f"  intended direction (conf > h): {r['share_intended']:.2%}",
             f"  reversed (h > conf):           {r['n_reversed']:,}"]
    if r["n_reversed"]:
        lines.append(f"    raw hints: {r['reversed_raw_hints_min']:.0f}-{r['reversed_raw_hints_max']:.0f} "
                     f"(mean {r['reversed_raw_hints_mean']:.1f})")
    lo, hi = DISPUTED_HINT_RANGE
    lines += [f"{lo}-{hi} raw hints & zero confusion:  {r['n_disputed_range_zero_confusion']:,} "
              f"(HIGH: {r['n_disputed_range_zero_confusion_high']})", "=" * 64]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=str, default="data/raw/assistments2017",
                        help="folder with the ASSISTments 2017 student_log_*.csv files")
    parser.add_argument("--label-path", type=str, default="data/raw/assistments2017/training_label.csv",
                        help="label file whose ITEST_id list defines the Meta-KT student set")

    args = parser.parse_args()



    print(format_report(audit(load_and_merge_data(args.log_dir, args.label_path))))


if __name__ == "__main__":
    main()
