import os
import re
import pickle
import argparse

import numpy as np
import pandas as pd

import aggregate_ratings as ar

SENTENCE_TEXT_COLUMN_CANDIDATES = ["sentence_en", "sentence_text", "sentence", "hypothesis", "text"]

PASS_LIKE_STATUSES = {"PASS", "PASS_NO_SIGNAL", "PASS_INFERRED"}
FAIL_CONTENT_STATUSES = {"FAIL_NLI", "FAIL_INVALID_REF", "FAIL_NLI_UNCITED"}

def _detect_sentence_text_column(sentence_key_df: pd.DataFrame) -> str:
    for c in SENTENCE_TEXT_COLUMN_CANDIDATES:
        if c in sentence_key_df.columns:
            return c
    return None

def _collect_item_free_text(resp_df: pd.DataFrame) -> pd.DataFrame:
    ft = resp_df.dropna(subset=["free_text"]).copy()
    ft = ft[ft["free_text"].astype(str).str.strip() != ""]
    if ft.empty:
        return pd.DataFrame(columns=["item_id", "free_text_combined"])
    ft = ft.drop_duplicates(subset=["item_id", "rater_id", "free_text"])
    grouped = ft.groupby("item_id").apply(
        lambda g: " | ".join(f"{r.rater_id}: {r.free_text}" for r in g.itertuples())
    ).reset_index(name="free_text_combined")
    return grouped

def load_case_table(responses_dir: str, sentence_answer_key_path: str,
                     answer_key_path: str = None, verifier_scores_path: str = None) -> pd.DataFrame:
    resp_df = ar.load_responses(responses_dir)
    sentence_key_df = ar.load_sentence_answer_key(sentence_answer_key_path)
    sentence_summary = ar.build_sentence_level_summary(resp_df, sentence_key_df)
    if sentence_summary.empty:
        raise RuntimeError("Sentence-level summary is empty after joining ratings and keys.")

    text_col = _detect_sentence_text_column(sentence_key_df)
    if text_col and text_col != "sentence_text":
        key_text = sentence_key_df.rename(columns={"display_id": "item_id", text_col: "sentence_text"})
        sentence_summary = sentence_summary.merge(
            key_text[["item_id", "sentence_id", "sentence_text"]], on=["item_id", "sentence_id"], how="left"
        )
    elif not text_col:
        sentence_summary["sentence_text"] = None
        print("[warning] Sentence text column not found; exporting metadata only")

    free_text_df = _collect_item_free_text(resp_df)
    sentence_summary = sentence_summary.merge(free_text_df, on="item_id", how="left")

    if answer_key_path:
        key_df = ar.load_answer_key(answer_key_path)
        cols = [c for c in ["category", "is_negative_control", "nli_tertile", "in_core_set"]
                if c in key_df.columns]
        sentence_summary = sentence_summary.merge(
            key_df[cols], left_on="item_id", right_index=True, how="left"
        )

    if verifier_scores_path:
        suffix = str(verifier_scores_path).lower()
        if suffix.endswith((".pkl", ".pickle")):
            scores = pd.read_pickle(verifier_scores_path)
        elif suffix.endswith(".csv"):
            scores = pd.read_csv(verifier_scores_path)
        else:
            raise ValueError(f"Unsupported verifier-score format: {verifier_scores_path}")

        required = {
            "item_id", "sentence_id", "majority_tag", "is_negative_control",
            "score", "verifier", "premise",
        }
        missing = required - set(scores.columns)
        if missing:
            raise ValueError(f"Missing verifier-score columns: {sorted(missing)}")

        if pd.api.types.is_bool_dtype(scores["is_negative_control"]):
            is_control = scores["is_negative_control"].fillna(False)
        else:
            is_control = (
                scores["is_negative_control"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes"})
            )

        canonical = scores[
            (scores["verifier"] == "mDeBERTa-v3-base-mnli-xnli")
            & (scores["premise"] == "fact")
            & (~is_control)
            & scores["majority_tag"].ne("unknown")
            & scores["score"].notna()
        ][["item_id", "sentence_id", "score"]].copy()

        if canonical.duplicated(["item_id", "sentence_id"]).any():
            raise RuntimeError("Duplicate canonical mDeBERTa/fact scores found.")

        sentence_summary = sentence_summary.merge(
            canonical.rename(columns={"score": "canonical_entailment"}),
            on=["item_id", "sentence_id"],
            how="inner",
            validate="one_to_one",
        )
        sentence_summary["auto_entailment"] = sentence_summary["canonical_entailment"]
        sentence_summary["auto_status"] = np.where(
            sentence_summary["canonical_entailment"] >= 0.5,
            "PASS",
            "FAIL_NLI",
        )
        print(f"[canonical disagreement set] sentences={len(sentence_summary)}")

    return sentence_summary

def classify_and_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["majority_tag", "auto_status"]).copy()

    auto_pass_like = df["auto_status"].isin(PASS_LIKE_STATUSES)
    auto_fail_content = df["auto_status"].isin(FAIL_CONTENT_STATUSES)

    conditions = [
        auto_pass_like & (df["majority_tag"] == "unsupported"),
        auto_fail_content & (df["majority_tag"] == "supported"),
        auto_pass_like & (df["majority_tag"] == "partial"),
        auto_fail_content & (df["majority_tag"] == "partial"),
        auto_pass_like & (df["majority_tag"] == "supported"),
        auto_fail_content & (df["majority_tag"] == "unsupported"),
    ]
    choices = ["under_detection", "over_flag", "partial_near_miss_pass",
               "partial_near_miss_fail", "agree_pass", "agree_fail"]
    df["case_type"] = np.select(conditions, choices, default="other")

    ent = df["auto_entailment"].astype(float)
    df["severity"] = np.nan
    df.loc[df["case_type"] == "under_detection", "severity"] = ent.fillna(0.5)
    df.loc[df["case_type"] == "over_flag", "severity"] = 1 - ent.fillna(0.5)
    df.loc[df["case_type"] == "partial_near_miss_pass", "severity"] = ent.fillna(0.5) * 0.5
    df.loc[df["case_type"] == "partial_near_miss_fail", "severity"] = (1 - ent.fillna(0.5)) * 0.5

    return df

def _load_premise_reconstructor(lang: str = "en"):
    try:
        from stage3_verification import _fact_to_premise_text
        return lambda fact: _fact_to_premise_text(fact, lang=lang)
    except Exception as e:
        print(f"[warning] Premise reconstruction is unavailable: {e}")
        return None

def _build_internal_id_maps(answer_key_path: str):
    key_df = pd.read_csv(answer_key_path, dtype={"display_id": str})
    display_to_internal = dict(zip(key_df["display_id"], key_df["internal_item_id"]))
    return display_to_internal

def _load_facts_by_internal_id(facts_pickle_path: str) -> dict:
    with open(facts_pickle_path, "rb") as f:
        items = pickle.load(f)
    return {it["item_id"]: it for it in items}

def _resolve_facts_for_internal_id(internal_id: str, items_by_id: dict):
    if internal_id in items_by_id:
        return items_by_id[internal_id].get("facts")
    m = re.match(r"^NC-(IT-\d+)-(IT-\d+)$", str(internal_id))
    if m:
        host_id = m.group(1)
        if host_id in items_by_id:
            return items_by_id[host_id].get("facts")
    return None

def attach_premise_text(df: pd.DataFrame, answer_key_path: str, facts_pickle_path: str,
                         lang: str = "en") -> pd.DataFrame:
    reconstructor = _load_premise_reconstructor(lang)
    if reconstructor is None:
        df["premise_text"] = None
        return df

    display_to_internal = _build_internal_id_maps(answer_key_path)
    items_by_id = _load_facts_by_internal_id(facts_pickle_path)

    premises = []
    n_ok, n_fail = 0, 0
    for _, row in df.iterrows():
        premise = None
        try:
            internal_id = display_to_internal.get(row["item_id"])
            facts = _resolve_facts_for_internal_id(internal_id, items_by_id) if internal_id else None
            cited = str(row.get("cited_indicators") or "")
            first_indicator = cited.split(";")[0].split(",")[0].strip() if cited else None
            if facts and first_indicator:
                match = next((f for f in facts if f.indicator == first_indicator), None)
                if match is not None:
                    premise = reconstructor(match)
        except Exception:
            premise = None
        if premise:
            n_ok += 1
        else:
            n_fail += 1
        premises.append(premise)

    df["premise_text"] = premises
    print(f"[premise reconstruction] success={n_ok}, unavailable={n_fail}")
    return df

CASE_TYPE_LABELS = {
    "under_detection": "Under-detection",
    "over_flag": "Over-flagging",
    "partial_near_miss_pass": "Partial near-miss (automatic pass)",
    "partial_near_miss_fail": "Partial near-miss (automatic fail)",
}

def _fmt_case_block(row: pd.Series) -> str:
    lines = [
        f"### {row['item_id']} / {row['sentence_id']} (severity={row['severity']:.3f})"
    ]
    if "category" in row and pd.notna(row.get("category")):
        suffix = " [negative control]" if row.get("is_negative_control") else ""
        lines.append(f"- Category: {row['category']}{suffix}")
    lines.append(f"- Cited indicators: {row.get('cited_indicators')}")
    if pd.notna(row.get("sentence_text")):
        lines.append(f"- Sentence: \"{row['sentence_text']}\"")
    if pd.notna(row.get("premise_text")):
        lines.append(f"- NLI premise: \"{row['premise_text']}\"")
    entailment = row.get("auto_entailment")
    entailment_text = f"{entailment:.3f}" if pd.notna(entailment) else "N/A"
    lines.append(f"- auto_status={row['auto_status']}, auto_entailment={entailment_text}")
    lines.append(
        f"- Expert majority={row['majority_tag']} "
        f"(distribution={row['tag_distribution']}, n={row['n_raters']})"
    )
    if pd.notna(row.get("free_text_combined")):
        lines.append(f"- Rater comments: {row['free_text_combined']}")
    return "\n".join(lines)

def write_report(df: pd.DataFrame, output_dir: str, top_n: int):
    os.makedirs(output_dir, exist_ok=True)
    full = df.sort_values(["case_type", "severity"], ascending=[True, False])
    csv_path = os.path.join(output_dir, "all_sentence_cases.csv")
    full.to_csv(csv_path, index=False)

    counts = df["case_type"].value_counts()
    md = ["# MetaKT-Verba disagreement cases", "", "## Counts"]
    for case_type, label in CASE_TYPE_LABELS.items():
        md.append(f"- {label}: {int(counts.get(case_type, 0))}")
    md.append(f"- Agreement: {int(counts.get('agree_pass', 0) + counts.get('agree_fail', 0))}")
    md.append(f"- Other: {int(counts.get('other', 0))}")
    md.append("")

    for case_type, label in CASE_TYPE_LABELS.items():
        subset = (
            df[df["case_type"] == case_type]
            .sort_values("severity", ascending=False)
            .head(top_n)
        )
        md.extend([f"## {label}: top {len(subset)}", ""])
        if subset.empty:
            md.append("_No cases._")
        for _, row in subset.iterrows():
            md.extend([_fmt_case_block(row), ""])

    report_path = os.path.join(output_dir, "qualitative_review.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved qualitative review to {report_path}")
    print(f"Saved sentence-level cases to {csv_path}")

def run(responses_dir, sentence_answer_key, answer_key, output_dir, top_n,
        facts_pickle=None, lang="en", verifier_scores_path=None):
    df = load_case_table(responses_dir, sentence_answer_key, answer_key, verifier_scores_path=verifier_scores_path)
    df = classify_and_score(df)
    if facts_pickle:
        if not answer_key:
            raise ValueError("--facts-pickle requires --answer-key.")
        df = attach_premise_text(df, answer_key, facts_pickle, lang=lang)
    else:
        df["premise_text"] = None
    write_report(df, output_dir, top_n)

def main():
    parser = argparse.ArgumentParser(
        description="Export sentence-level expert/Stage-3 disagreement cases."
    )
    parser.add_argument("--responses-dir", default="data/ratings")
    parser.add_argument(
        "--sentence-answer-key",
        default="data/stimuli/internal_answer_key_sentences.csv",
    )
    parser.add_argument("--answer-key", default="data/stimuli/internal_answer_key.csv")
    parser.add_argument(
        "--facts-pickle",
        default=None,
        help="Optional Phase-B cache for best-effort premise reconstruction.",
    )
    parser.add_argument("--lang", default="en", choices=["ko", "en"])
    parser.add_argument(
        "--verifier-scores",
        default="data/results/verifier_family/scores_all_verifiers.csv",
        help="Canonical verifier-family scores in CSV or PKL format.",
    )
    parser.add_argument("--output-dir", default="outputs/disagreement_review")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    run(
        args.responses_dir,
        args.sentence_answer_key,
        args.answer_key,
        args.output_dir,
        args.top_n,
        facts_pickle=args.facts_pickle,
        lang=args.lang,
        verifier_scores_path=args.verifier_scores,
    )

if __name__ == "__main__":
    main()
