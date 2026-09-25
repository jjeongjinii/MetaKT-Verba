


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


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

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
                     answer_key_path: str = None) -> pd.DataFrame:
    resp_df = ar.load_responses(responses_dir)
    sentence_key_df = ar.load_sentence_answer_key(sentence_answer_key_path)
    sentence_summary = ar.build_sentence_level_summary(resp_df, sentence_key_df)
    if sentence_summary.empty:
        raise RuntimeError("sentence_level_summary가 비어 있습니다 (응답/키 조인 결과 없음).")

    text_col = _detect_sentence_text_column(sentence_key_df)
    if text_col and text_col != "sentence_text":
        key_text = sentence_key_df.rename(columns={"display_id": "item_id", text_col: "sentence_text"})
        sentence_summary = sentence_summary.merge(
            key_text[["item_id", "sentence_id", "sentence_text"]], on=["item_id", "sentence_id"], how="left"
        )
    elif not text_col:
        sentence_summary["sentence_text"] = None
        print("[경고] sentence-answer-key에서 문장 텍스트 컬럼을 찾지 못했습니다 "
              f"(찾아본 이름: {SENTENCE_TEXT_COLUMN_CANDIDATES}). "
              "생성 문장 원문 없이 cited_indicators/auto_status만 표시됩니다.")

    free_text_df = _collect_item_free_text(resp_df)
    sentence_summary = sentence_summary.merge(free_text_df, on="item_id", how="left")

    if answer_key_path:
        key_df = ar.load_answer_key(answer_key_path)
        cols = [c for c in ["category", "is_negative_control", "nli_tertile", "in_core_set"]
                if c in key_df.columns]
        sentence_summary = sentence_summary.merge(
            key_df[cols], left_on="item_id", right_index=True, how="left"
        )

    return sentence_summary


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

def _load_premise_reconstructor(lang: str = "ko"):



    try:
        from stage3_verification import _fact_to_premise_text
        return lambda fact: _fact_to_premise_text(fact, lang=lang)
    except Exception as e:
        print(f"[안내] stage3_verification.py를 import하지 못해 premise 원문 재구성을 건너뜁니다: {e}")
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
                         lang: str = "ko") -> pd.DataFrame:
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
    print(f"[상세 모드] premise 재구성 성공 {n_ok}건 / 실패(또는 해당없음) {n_fail}건")
    return df


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

CASE_TYPE_LABELS = {
    "under_detection": "자동지표 미탐지 — auto=PASS(계열)인데 전문가 다수결=근거없음",
    "over_flag": "자동지표 과잉플래그 — auto=FAIL(내용)인데 전문가 다수결=지지됨",
    "partial_near_miss_pass": "근접사례 — auto=PASS(계열)인데 전문가 다수결=부분지지",
    "partial_near_miss_fail": "근접사례 — auto=FAIL(내용)인데 전문가 다수결=부분지지",
}


def _fmt_case_block(row: pd.Series) -> str:
    lines = [f"### {row['item_id']} / {row['sentence_id']}  (severity={row['severity']:.3f})"]
    if "category" in row and pd.notna(row.get("category")):
        lines.append(f"- 범주: {row['category']}" +
                      ("  ⚠ 부정 대조 문항" if row.get("is_negative_control") else ""))
    lines.append(f"- 인용 지표(cited_indicators): {row.get('cited_indicators')}")
    if pd.notna(row.get("sentence_text")):
        lines.append(f"- 생성된 문장: \"{row['sentence_text']}\"")
    if pd.notna(row.get("premise_text")):
        lines.append(f"- NLI premise 원문: \"{row['premise_text']}\"")
    ent = row.get("auto_entailment")
    lines.append(f"- auto_status={row['auto_status']}" +
                 (f", auto_entailment={ent:.3f}" if pd.notna(ent) else ", auto_entailment=N/A"))
    lines.append(f"- 전문가 다수결: {row['majority_tag']}  (분포: {row['tag_distribution']}, n={row['n_raters']})")
    if pd.notna(row.get("free_text_combined")):
        lines.append(f"- 평정자 자유서술(문항 전체 기준): {row['free_text_combined']}")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, output_dir: str, top_n: int):
    os.makedirs(output_dir, exist_ok=True)
    df_sorted_full = df.sort_values(["case_type", "severity"], ascending=[True, False])
    df_sorted_full.to_csv(os.path.join(output_dir, "all_sentence_cases.csv"), index=False)

    counts = df["case_type"].value_counts()
    md = ["# MetaKT-Verba 자동-전문가 불일치 정성분석용 사례집", ""]
    md.append("## 사례 유형별 건수")
    for ct, label in CASE_TYPE_LABELS.items():
        md.append(f"- {label}: {int(counts.get(ct, 0))}건")
    md.append(f"- 일치(agree_pass/agree_fail): {int(counts.get('agree_pass', 0) + counts.get('agree_fail', 0))}건")
    md.append(f"- 기타(other): {int(counts.get('other', 0))}건")
    md.append("")

    for ct, label in CASE_TYPE_LABELS.items():
        subset = df[df["case_type"] == ct].sort_values("severity", ascending=False).head(top_n)
        md.append(f"## {label} — 상위 {len(subset)}건")
        md.append("")
        if subset.empty:
            md.append("_해당 사례 없음_")
        for _, row in subset.iterrows():
            md.append(_fmt_case_block(row))
            md.append("")
        md.append("")

    report_path = os.path.join(output_dir, "qualitative_review.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"✅ 정성분석용 사례집 작성 완료: {report_path}")
    print(f"✅ 전체 사례 CSV: {os.path.join(output_dir, 'all_sentence_cases.csv')}")


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

def run(responses_dir, sentence_answer_key, answer_key, output_dir, top_n,
        facts_pickle=None, lang="ko"):
    df = load_case_table(responses_dir, sentence_answer_key, answer_key)
    df = classify_and_score(df)
    if facts_pickle:
        if not answer_key:
            raise ValueError("--facts-pickle을 쓰려면 --answer-key도 함께 지정해야 합니다 "
                              "(display_id -> internal_item_id 매핑이 필요합니다).")
        df = attach_premise_text(df, answer_key, facts_pickle, lang=lang)
    else:
        df["premise_text"] = None
    write_report(df, output_dir, top_n)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses-dir", type=str, default='data/ratings')
    parser.add_argument("--sentence-answer-key", type=str, default='data/stimuli/internal_answer_key_sentences.csv')
    parser.add_argument("--answer-key", type=str, default='data/stimuli/internal_answer_key.csv',
                         help="internal_answer_key.csv 경로 (category/negative-control 표시 및 "
                              "--facts-pickle 사용 시 필수)")
    parser.add_argument("--facts-pickle", type=str, default=None,
                         help="prepare_calibration_stimuli.py --save-phase-b로 저장한 캐시 경로. "
                              "있으면 premise 원문을 재구성해서 보여줍니다 (상세 모드).")
    parser.add_argument("--lang", type=str, default="en", choices=["ko", "en"])
    parser.add_argument("--output-dir", type=str, default="outputs/disagreement_review")
    parser.add_argument("--top-n", type=int, default=20)

    args = parser.parse_args()


    if not args.responses_dir or not args.sentence_answer_key:
        parser.error("--responses-dir와 --sentence-answer-key가 필요합니다")

    run(args.responses_dir, args.sentence_answer_key, args.answer_key,
        args.output_dir, args.top_n, facts_pickle=args.facts_pickle, lang=args.lang)


if __name__ == "__main__":
    main()