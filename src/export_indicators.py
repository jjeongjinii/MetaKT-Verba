import argparse
from pathlib import Path

import pandas as pd

from metakt_indicators import load_and_merge_data, preprocess_meta_kt


META_COLS = [
    "m_overconfidence",
    "m_underconfidence",
    "m_strategic_help",
    "m_cognitive_avoidance",
    "m_productive_struggle",
    "m_unproductive_frustration",
    "m_lucky_guess",
    "m_slipping",
    "m_boredom_offtask",
]

CONTEXT_COLS = [
    "step_idx",
    "response_time_ratio",
    "skill_cum_accuracy",
    "hintCount_raw",
    "SC",
]

OUTPUT_COLS = [
    "ITEST_id",
    "skill",
    "correct",
    *CONTEXT_COLS,
    *META_COLS,
]


def build_indicator_export(log_dir: Path, train_labels: Path, test_labels: Path) -> pd.DataFrame:
    train_raw = load_and_merge_data(str(log_dir), str(train_labels))
    test_raw = load_and_merge_data(str(log_dir), str(test_labels))

    overlap = set(train_raw["ITEST_id"].unique()) & set(test_raw["ITEST_id"].unique())
    if overlap:
        train_raw = train_raw[~train_raw["ITEST_id"].isin(overlap)].copy()

    raw = pd.concat([train_raw, test_raw], ignore_index=True).drop_duplicates().copy()
    raw["hintCount_raw"] = raw["hintCount"]

    df = preprocess_meta_kt(raw)
    df["hintCount_raw"] = raw.loc[df.index, "hintCount_raw"]
    df["step_idx"] = df.groupby("ITEST_id").cumcount()
    df["response_time_ratio"] = (
        df["elapsed_time"]
        / df.groupby("ITEST_id")["elapsed_time"].transform("mean").replace(0, pd.NA)
    )
    df["skill_cum_accuracy"] = (
        df.groupby(["ITEST_id", "skill"])["correct"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    missing = [col for col in OUTPUT_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after preprocessing: {missing}")

    return df[OUTPUT_COLS].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Meta-KT indicators and interaction context without loading a prediction model."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/raw/assistments2017"),
        help="Directory containing student_log_*.csv files.",
    )
    parser.add_argument(
        "--train-labels",
        type=Path,
        default=Path("data/raw/assistments2017/training_label.csv"),
    )
    parser.add_argument(
        "--test-labels",
        type=Path,
        default=Path("data/raw/assistments2017/validation_test_label.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/full_predictions2.csv"),
    )
    args = parser.parse_args()

    result = build_indicator_export(args.log_dir, args.train_labels, args.test_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
