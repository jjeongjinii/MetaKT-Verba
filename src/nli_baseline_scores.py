


import argparse
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core_set_io import (determine_core_set, infer_original_citations, load_item_context,
                         load_r1_r5_csvs, load_stimuli_sentences)
from stage1_grounding import build_symbolic_facts


_VALID_INDICATOR_TOKEN = re.compile(r"^([a-zA-Z_]+)")


def _clean_indicator_token(raw: str) -> Optional[str]:


    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    m = _VALID_INDICATOR_TOKEN.match(raw)
    return m.group(1) if m else None


def load_true_citations(sentence_answer_key_csv: str) -> Dict[tuple, List[str]]:


    df = pd.read_csv(sentence_answer_key_csv)
    id_col = next((c for c in ["display_id", "item_id"] if c in df.columns), None)
    sid_col = next((c for c in ["sentence_id", "sid"] if c in df.columns), None)
    cite_col = next((c for c in ["cited_indicators", "cited_indicator", "indicator"] if c in df.columns), None)
    if id_col is None or sid_col is None or cite_col is None:
        raise KeyError(
            f"필요한 컬럼을 못 찾았습니다. 실제 컬럼: {list(df.columns)} — "
            f"load_true_citations()의 후보 목록에 실제 이름을 추가하세요."
        )
    out: Dict[tuple, List[str]] = {}
    n_messy = 0
    for _, row in df.iterrows():
        cite = row[cite_col]
        if isinstance(cite, str) and cite.startswith("["):
            import ast
            tokens = ast.literal_eval(cite)
        elif isinstance(cite, str):
            tokens = cite.split(";")  
        else:
            tokens = []

        cleaned = []
        for t in tokens:
            c = _clean_indicator_token(t)
            if c is None:
                continue
            if c != t.strip():
                n_messy += 1
            cleaned.append(c)
        out[(row[id_col], row[sid_col])] = cleaned

    if n_messy:
        print(f"[load_true_citations] 인용 태그 파싱 잔여물이 있던 항목 {n_messy}건을 정리했습니다 "
              f"(원본 Stage2 생성 시 태그 스트리핑 정규식이 일부 실패한 흔적).")
    return out


def extract_baseline_from_sentence_answer_key(
    sentence_answer_key_csv: str, target_keys: Optional[List[tuple]] = None,
) -> pd.DataFrame:


    df = pd.read_csv(sentence_answer_key_csv)
    id_col = next((c for c in ["display_id", "item_id"] if c in df.columns), None)
    sid_col = next((c for c in ["sentence_id", "sid"] if c in df.columns), None)
    if id_col is None or sid_col is None or "auto_entailment" not in df.columns:
        raise KeyError(
            f"필요한 컬럼을 못 찾았습니다. 실제 컬럼: {list(df.columns)} — "
            f"'auto_entailment'이 있는 파일이어야 합니다."
        )

    out = df[[id_col, sid_col, "auto_entailment", "auto_status"]].rename(
        columns={id_col: "item_id", sid_col: "sentence_id"}).copy()

    if "auto_status" in out.columns:
        invalid_ref_mask = out["auto_status"] == "FAIL_INVALID_REF"
        n_invalid_ref = int(invalid_ref_mask.sum())
        if n_invalid_ref:
            out.loc[invalid_ref_mask, "auto_entailment"] = 0.0
            print(f"[extract_baseline_from_sentence_answer_key] FAIL_INVALID_REF {n_invalid_ref}건을 "
                  f"entailment=0(내용 검증 실패)으로 채점했습니다 — 제외하면 NLI에 유리하게 "
                  f"편향되므로 stage3_verification.py의 hallucination_rate 계산과 일치시킴.")

        n_no_citation = int((out["auto_status"] == "FAIL_NO_CITATION").sum())
        if n_no_citation:
            print(f"[extract_baseline_from_sentence_answer_key] FAIL_NO_CITATION {n_no_citation}건은 "
                  f"제외합니다(인용 형식 자체를 안 지킨 구조 위반 — 내용 충실도와는 다른 축).")

    out["confidence_supported"] = out["auto_entailment"] * 100

    if target_keys is not None:
        key_set = set(target_keys)
        before = len(out)
        out = out[out.apply(lambda r: (r["item_id"], r["sentence_id"]) in key_set, axis=1)].reset_index(drop=True)
        missing = key_set - set(zip(out["item_id"], out["sentence_id"]))
        print(f"[extract_baseline_from_sentence_answer_key] {before}행 중 target_keys {len(key_set)}개와 "
              f"일치하는 {len(out)}행만 남김.")
        if missing:
            print(f"  ⚠ target_keys에는 있는데 이 CSV엔 없는 키 {len(missing)}개(참고): "
                  f"{list(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
    return out


def combine_baseline_with_recovered(
    sentence_answer_key_csv: str, recovered_csv: Optional[str] = None,
    target_keys: Optional[List[tuple]] = None,
    include_recovery_methods: Optional[List[str]] = ("inferred_negative_control",),
) -> pd.DataFrame:


    baseline = extract_baseline_from_sentence_answer_key(sentence_answer_key_csv)

    if recovered_csv:
        rec = pd.read_csv(recovered_csv)
        if include_recovery_methods is not None:
            rec = rec[rec["recovery_method"].isin(include_recovery_methods)].copy()
        rec["auto_entailment"] = rec["recovered_entailment"]
        rec["auto_status"] = "RECOVERED:" + rec["recovery_method"]
        rec["confidence_supported"] = rec["auto_entailment"] * 100
        rec = rec[["item_id", "sentence_id", "auto_entailment", "auto_status", "confidence_supported"]]

        
        
        overlap_keys = set(zip(rec["item_id"], rec["sentence_id"]))
        n_overlap = baseline.apply(lambda r: (r["item_id"], r["sentence_id"]) in overlap_keys, axis=1).sum()
        if n_overlap:
            print(f"[combine_baseline_with_recovered] 기존 값 {n_overlap}건을 recovered 값으로 교체합니다.")
        baseline = baseline[~baseline.apply(lambda r: (r["item_id"], r["sentence_id"]) in overlap_keys, axis=1)]
        baseline = pd.concat([baseline, rec], ignore_index=True)
        print(f"[combine_baseline_with_recovered] {recovered_csv}에서 {len(rec)}개 문장을 추가했습니다 "
              f"(recovery_method={list(include_recovery_methods) if include_recovery_methods else '전체'}).")

    if target_keys is not None:
        key_set = set(target_keys)
        before = len(baseline)
        baseline = baseline[baseline.apply(lambda r: (r["item_id"], r["sentence_id"]) in key_set, axis=1)].reset_index(drop=True)
        missing = key_set - set(zip(baseline["item_id"], baseline["sentence_id"]))
        print(f"[combine_baseline_with_recovered] {before}행 중 target_keys {len(key_set)}개와 "
              f"일치하는 {len(baseline)}행만 남김.")
        if missing:
            print(f"  ⚠ target_keys에는 있는데 여전히 없는 키 {len(missing)}개(참고): "
                  f"{list(missing)[:5]}{' ...' if len(missing) > 5 else ''}")

    return baseline


def compute_nli_baseline_for_core_set(
    source_csv: str, answer_key_csv: str, stimuli_master_path: str, responses_dir: str,
    output_csv: str, sentence_answer_key_csv: Optional[str] = None, lang: str = "en",
) -> pd.DataFrame:
    from prepare_calibration_stimuli import load_source_data
    from stage2_generation import CitedSentence
    from stage3_verification import NLIVerifier

    print("소스 데이터 로딩 중...")
    df = load_source_data(source_csv)
    key_df = pd.read_csv(answer_key_csv)
    sentences_by_display_id = load_stimuli_sentences(stimuli_master_path, lang=lang)

    print("R1~R5 평정으로 core-set 판별 중...")
    _, level2_df = load_r1_r5_csvs(responses_dir)
    core_items = determine_core_set(level2_df)
    print(f"core-set 문항 수: {len(core_items)}")

    true_citations = None
    if sentence_answer_key_csv:
        true_citations = load_true_citations(sentence_answer_key_csv)
        print(f"문장별 실제 인용 {len(true_citations)}개를 불러왔습니다 — 추정 대신 이걸 우선 사용합니다.")

    print("NLI 모델(mDeBERTa) 로딩 중... (vLLM/Ministral/Llama 불필요)")
    verifier = NLIVerifier()

    rows: List[Dict[str, object]] = []
    n_skipped_nc = 0
    n_skipped_not_core = 0
    for i, key_row in key_df.iterrows():
        display_id = key_row["display_id"]
        if display_id not in core_items:
            n_skipped_not_core += 1
            continue
        try:
            ctx = load_item_context(display_id, key_row["internal_item_id"], sentences_by_display_id, df)
        except (ValueError, KeyError) as e:
            n_skipped_nc += 1  
            print(f"[경고] {display_id} 건너뜀: {type(e).__name__}: {e}")
            continue

        original_facts = build_symbolic_facts(ctx.meta_vector, hint_count=ctx.hint_count, sc=ctx.sc, lang=lang)
        sentence_texts = ctx.sentence_texts
        if true_citations is not None:
            citation_lists = [true_citations.get((display_id, f"S{i+1}"), []) for i in range(len(sentence_texts))]
        else:
            
            
            inferred = infer_original_citations(sentence_texts, original_facts, verifier, lang=lang)
            citation_lists = [[c] if c else [] for c in inferred]

        for idx, (text, citation_list) in enumerate(zip(sentence_texts, citation_lists)):
            sentence_id = f"S{idx + 1}"
            if not citation_list:
                continue
            result = verifier.verify_narrative(
                [CitedSentence(text=text, cited_indicators=citation_list)], original_facts, lang=lang,
            )[0]
            entailment = result.nli_scores["entailment"] if result.nli_scores is not None else np.nan
            rows.append({
                "item_id": display_id, "sentence_id": sentence_id, "sentence_text": text,
                "citation": ";".join(citation_list), "citation_is_true": true_citations is not None,
                "nli_entailment_score": entailment,
                "confidence_supported": entailment * 100 if not np.isnan(entailment) else np.nan,
            })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(key_df)}개 문항 확인 완료 (지금까지 {len(rows)}개 문장 채점됨)")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f"\n완료: core-set {len(core_items)}개 문항에서 {len(out_df)}개 문장 채점 "
          f"(core-set 아님 {n_skipped_not_core}개, 형식 불일치 {n_skipped_nc}개 제외). 저장: {output_csv}")
    print(f"\n다음 단계: 이 CSV를 Ministral/Llama의 predictions.csv와 (item_id, sentence_id) "
          f"기준으로 merge하세요 — true_label은 그쪽 파일 걸 그대로 쓰고, confidence_supported만 "
          f"세 번째 열로 추가하면 compare_auroc.py의 3자 비교에 바로 넣을 수 있습니다.")
    return out_df


# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--source-csv", type=str, default='data/interim/full_predictions2.csv')
    parser.add_argument("--answer-key", type=str, default='data/stimuli/internal_answer_key.csv')
    parser.add_argument("--stimuli-master", type=str, default='data/stimuli/stimuli_master.json')
    parser.add_argument("--responses-dir", type=str, default='data/ratings')
    parser.add_argument("--sentence-answer-key", type=str, default='data/stimuli/internal_answer_key_sentences.csv')
    parser.add_argument("--output", type=str, default="outputs/nli_baseline_scores.csv")

    parser.add_argument("--lang", type=str, default="en", choices=["en", "ko"])
    args = parser.parse_args()


    required = [args.source_csv, args.answer_key, args.stimuli_master, args.responses_dir]
    if not all(required):
        parser.error("--source-csv, --answer-key, --stimuli-master, --responses-dir가 모두 필요합니다.")

    compute_nli_baseline_for_core_set(
        source_csv=args.source_csv, answer_key_csv=args.answer_key,
        stimuli_master_path=args.stimuli_master, responses_dir=args.responses_dir,
        output_csv=args.output, sentence_answer_key_csv=args.sentence_answer_key, lang=args.lang,
    )


if __name__ == "__main__":
    main()