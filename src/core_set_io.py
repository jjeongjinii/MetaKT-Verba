


import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aggregate_ratings import _parse_level1_tags_field
from prepare_calibration_stimuli import COLUMN_MAP


def load_r1_r5_csvs(responses_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:


    required_cols = {"item_id", "rater_id", "level1_tags", "faithfulness", "actionability", "clarity"}
    paths = sorted(glob.glob(os.path.join(responses_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"'{responses_dir}'에서 CSV를 찾지 못했습니다.")

    level1_rows, level2_rows = [], []
    n_skipped = 0
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
        if not required_cols.issubset(df.columns):
            n_skipped += 1
            continue  
        for _, r in df.iterrows():
            tags = _parse_level1_tags_field(r["level1_tags"])
            for sid, tag in tags.items():
                level1_rows.append({"item_id": r["item_id"], "rater_id": r["rater_id"],
                                     "sentence_id": sid, "level1_tag": tag})
            level2_rows.append({
                "item_id": r["item_id"], "rater_id": r["rater_id"],
                "faithfulness": float(r["faithfulness"]), "actionability": float(r["actionability"]),
                "clarity": float(r["clarity"]),
            })

    if n_skipped:
        print(f"[load_r1_r5_csvs] 평정 CSV 형식이 아닌 파일 {n_skipped}개는 건너뛰었습니다.")
    if not level2_rows:
        raise ValueError(f"'{responses_dir}'에 평정 CSV 형식(item_id,rater_id,level1_tags,...)에 맞는 "
                          f"파일이 하나도 없습니다.")
    return pd.DataFrame(level1_rows), pd.DataFrame(level2_rows)


def determine_core_set(level2_df: pd.DataFrame) -> set:


    n_raters_total = level2_df["rater_id"].nunique()
    counts = level2_df.groupby("item_id")["rater_id"].nunique()
    return set(counts[counts == n_raters_total].index)


def _extract_row_idx(internal_item_id: str) -> int:


    if not str(internal_item_id).startswith("IT-"):
        raise ValueError(f"internal_item_id 형식이 예상과 다릅니다('IT-<row_idx>' 형식이어야 함): {internal_item_id!r}")
    return int(str(internal_item_id)[3:])


@dataclass
class _PlainSentence:


    text: str


_SENTENCE_TEXT_KEYS = ("text", "sentence", "content", "narrative", "body")


def _extract_sentence_text(s, lang: str = "en") -> str:


    if isinstance(s, str):
        return s
    if isinstance(s, dict):
        if lang in s:
            return s[lang]
        for key in _SENTENCE_TEXT_KEYS:
            if key in s:
                return s[key]
        raise KeyError(
            f"문장 dict에서 텍스트 키를 찾지 못했습니다. lang='{lang}'과 시도한 키 "
            f"{_SENTENCE_TEXT_KEYS} 모두 없음. 실제로 있는 키: {list(s.keys())} — "
            f"core_set_io.py의 _SENTENCE_TEXT_KEYS에 실제 키 이름을 추가하세요."
        )
    raise TypeError(f"문장 원소의 타입을 처리할 수 없습니다: {type(s)} ({s!r})")


def load_stimuli_sentences(stimuli_master_path: str, lang: str = "en") -> Dict[str, List[str]]:


    with open(stimuli_master_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    return {it["item_id"]: [_extract_sentence_text(s, lang=lang) for s in it["sentences"]] for it in items}


def infer_original_citations(sentence_texts: List[str], original_facts: list, verifier, lang: str) -> List[Optional[str]]:



    plain = [_PlainSentence(text=t) for t in sentence_texts]
    results = verifier.verify_narrative_uncited(plain, original_facts, lang=lang)
    return [r.cited_indicators[0] if r.cited_indicators else None for r in results]


@dataclass
class ItemContext:


    display_id: str
    row_idx: int
    meta_vector: Dict[str, float]
    hint_count: float
    sc: Optional[float]
    sentence_texts: List[str]


def load_item_context(
    display_id: str, internal_item_id: str, sentences_by_display_id: Dict[str, List[str]], df: pd.DataFrame,
) -> ItemContext:


    if display_id not in sentences_by_display_id:
        raise KeyError(f"stimuli_master.json에 display_id={display_id!r}가 없습니다.")
    sentence_texts = sentences_by_display_id[display_id]

    row_idx = _extract_row_idx(internal_item_id)
    src_row = df.loc[row_idx]

    meta_vector = {mk: float(src_row[COLUMN_MAP[mk]]) for mk in COLUMN_MAP if mk.startswith("m_")}
    hint_count = float(src_row[COLUMN_MAP["hint_count"]])
    sc = None
    if COLUMN_MAP.get("sc") and COLUMN_MAP["sc"] in src_row.index and pd.notna(src_row[COLUMN_MAP["sc"]]):
        sc = float(src_row[COLUMN_MAP["sc"]])

    return ItemContext(
        display_id=display_id, row_idx=row_idx, meta_vector=meta_vector, hint_count=hint_count,
        sc=sc, sentence_texts=sentence_texts,
    )
