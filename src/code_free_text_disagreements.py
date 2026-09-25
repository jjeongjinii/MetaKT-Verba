


import os
import re
import argparse

import pandas as pd

# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

RATER_SPLIT_RE = re.compile(r'\s\|\s')
RATER_PREFIX_RE = re.compile(r'^(R\d+):\s*(.*)$', re.DOTALL)


SENTENCE_MARKER_RE = re.compile(
    r'\[?[sS](\d+)((?:\s*,?\s*[sS]\d+|\s*,\s*\d+)*)\]?\s*[:\-\.,]?\s*'
)


def _split_rater_segments(free_text_combined: str):
    if not isinstance(free_text_combined, str) or not free_text_combined.strip():
        return []
    chunks = RATER_SPLIT_RE.split(free_text_combined.strip())
    segments = []
    for c in chunks:
        m = RATER_PREFIX_RE.match(c.strip())
        if m:
            segments.append((m.group(1), m.group(2)))
        else:
            segments.append((None, c.strip()))  
    return segments


def _split_by_sentence_markers(body: str):


    matches = list(SENTENCE_MARKER_RE.finditer(body))
    if not matches:
        return [(None, body.strip())] if body.strip() else []

    pieces = []
    
    lead = body[:matches[0].start()].strip(" \n\t.,;-")
    if len(lead) >= 4:
        pieces.append((None, lead))

    for i, m in enumerate(matches):
        nums = {int(n) for n in re.findall(r'\d+', m.group(0)) }
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip(" \n\t.,;-")
        if text:
            pieces.append((nums, text))
    return pieces


def extract_targeted_comments(free_text_combined: str, sentence_num: int):


    targeted, general = [], []
    for rid, body in _split_rater_segments(free_text_combined):
        rid_label = rid or "?"
        for nums, text in _split_by_sentence_markers(body):
            if not text:
                continue
            if nums is None:
                general.append(f"{rid_label}: {text}")
            elif sentence_num in nums:
                targeted.append(f"{rid_label}: {text}")
    targeted_text = " / ".join(targeted)
    general_text = " / ".join(general)
    return targeted_text, general_text, bool(targeted)


def _sentence_id_to_num(sentence_id: str):
    m = re.search(r'(\d+)', str(sentence_id))
    return int(m.group(1)) if m else None


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

TYPE1_PATTERNS = [
    (r'오히려', 2.0, '오히려'),
    (r'정반대', 2.0, '정반대'),
    (r'반대(로|되)', 1.5, '반대(로/되)'),
    (r'기록과\s*맞지\s*않', 2.0, '기록과 맞지 않음'),
    (r'전제가\s*(잘못|성립하지\s*않)', 2.0, '전제가 잘못/불성립'),
    (r'근거\s*없이', 2.0, '근거 없이 (주장)'),
    (r'[라다]고\s*보[기긴]\s*(어렵|어려|힘들|힘든)', 2.0, "'~라고 보기 어렵다'"),
    (r'로\s*보[기긴]\s*(어렵|어려|힘들|힘든)', 2.0, "'~로 보기 어렵다'"),
    (r'[라다]고\s*(하기|하긴)\s*(어렵|어려|힘들|힘든)', 1.5, "'~하기 어렵다'"),
    (r'[라다]고\s*(할|하기)\s*수\s*없', 1.5, "'~할 수 없다'"),
    (r'단정(지을)?\s*수(는)?\s*없', 1.0, '단정할 수 없음'),
    (r'뒷받침(하기\s*(어렵|어려)|되지\s*않|하지\s*못|하기\s*(힘들|힘든))', 2.0, '뒷받침되지 않음'),
    (r'성립하지\s*않', 1.5, '성립하지 않음'),
    (r'예측할\s*수\s*있', 1.0, '(반대 방향을) 예측할 수 있음'),
    (r'(확인되지|관찰되지|나타나지|기록되지)\s*않', 1.5, "'확인/관찰/기록되지 않음'"),
]

TYPE2_PATTERNS = [
    (r'방향은?\s*맞', 2.0, '방향은 맞음'),
    (r'부분\s*(적으로)?\s*지지', 2.0, '부분(적으로) 지지'),
    (r'일부(는|만)?\s*(지지|맞|일치)', 1.5, '일부 지지/일치'),
    (r'다소\s*과장', 2.0, '다소 과장'),
    (r'과장된', 1.5, '과장된 표현'),
    (r'어느\s*정도\s*(일치|맞|가능)', 1.5, '어느 정도 일치'),
    (r'~?에\s*가깝다고\s*(판단|보임)', 1.0, '~에 가깝다고 판단'),
    (r'대체적으로\s*(맞|일치)', 1.5, '대체적으로 맞음'),
    (r'서술될\s*필요가\s*있', 1.0, '보완 서술 필요'),
]

TYPE3_PATTERNS = [
    (r'판단\s*(하기)?\s*(불가능|어렵|어려|할\s*수\s*없)', 2.0, '판단 불가/어려움'),
    (r'알\s*수\s*없', 2.0, '알 수 없음'),
    (r'확인할\s*수\s*없', 2.0, '확인할 수 없음'),
    (r'파악할\s*수\s*없', 2.0, '파악할 수 없음'),
    (r'기록에\s*없', 2.0, '기록에 없음'),
    (r'정보가\s*(없|부족|제한적)', 1.5, '정보 부족/제한적'),
    (r'판단할\s*근거[가를]?\s*(부족|없)', 1.5, '판단할 근거 부족'),
    (r'제시된\s*정보만으로는', 2.0, '제시된 정보만으로는'),
    (r'본문자료만으로는', 2.0, '본문자료만으로는'),
    (r'무슨\s*말[인을].*모르겠', 1.0, '모호함(무슨 말인지 모르겠음)'),
]

CATEGORY_PATTERNS = {
    'type1_dispute_fact': TYPE1_PATTERNS,
    'type2_overstated_wording': TYPE2_PATTERNS,
    'type3_insufficient_info': TYPE3_PATTERNS,
}


def classify_text(text: str) -> dict:

    scores = {}
    matched = {}
    for cat, patterns in CATEGORY_PATTERNS.items():
        score = 0.0
        hits = []
        for pattern, weight, label in patterns:
            if re.search(pattern, text):
                score += weight
                hits.append(label)
        scores[cat] = score
        matched[cat] = hits

    if not text.strip():
        return {'primary': 'no_comment', 'scores': scores, 'matched': matched}

    max_score = max(scores.values())
    if max_score == 0:
        return {'primary': 'unclassified', 'scores': scores, 'matched': matched}

    top = [c for c, s in scores.items() if s == max_score]
    if len(top) > 1:
        primary = 'mixed(' + '/'.join(sorted(top)) + ')'
    else:
        primary = top[0]
    return {'primary': primary, 'scores': scores, 'matched': matched}


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

DEFAULT_CASE_TYPES = ['under_detection', 'over_flag']


def code_dataframe(df: pd.DataFrame, case_types=None) -> pd.DataFrame:
    df = df.copy()
    if case_types:
        df = df[df['case_type'].isin(case_types)].copy()

    rows = []
    for _, row in df.iterrows():
        sent_num = _sentence_id_to_num(row.get('sentence_id'))
        free_text = row.get('free_text_combined')
        if sent_num is not None and isinstance(free_text, str) and free_text.strip():
            targeted, general, is_specific = extract_targeted_comments(free_text, sent_num)
            text_used = targeted if is_specific else general
        else:
            text_used, is_specific = "", False

        result = classify_text(text_used)
        rows.append({
            'item_id': row.get('item_id'), 'sentence_id': row.get('sentence_id'),
            'case_type': row.get('case_type'), 'cited_indicators': row.get('cited_indicators'),
            'category': row.get('category'), 'majority_tag': row.get('majority_tag'),
            'auto_status': row.get('auto_status'), 'auto_entailment': row.get('auto_entailment'),
            'coding_text_used': text_used,
            'coding_is_sentence_specific': is_specific,
            'coding_primary': result['primary'],
            'coding_score_type1': result['scores'].get('type1_dispute_fact', 0.0),
            'coding_score_type2': result['scores'].get('type2_overstated_wording', 0.0),
            'coding_score_type3': result['scores'].get('type3_insufficient_info', 0.0),
            'coding_matched_type1': '; '.join(result['matched'].get('type1_dispute_fact', [])),
            'coding_matched_type2': '; '.join(result['matched'].get('type2_overstated_wording', [])),
            'coding_matched_type3': '; '.join(result['matched'].get('type3_insufficient_info', [])),
        })
    return pd.DataFrame(rows)


def summarize(coded_df: pd.DataFrame) -> str:
    lines = ["=" * 70, "free_text 초벌 코딩 요약", "=" * 70, ""]
    lines.append("[1] case_type x coding_primary 교차표")
    lines.append(pd.crosstab(coded_df['case_type'], coded_df['coding_primary']).to_string())
    lines.append("")
    lines.append("[2] cited_indicators x coding_primary 교차표 (지표별로 이의제기 유형이 다른지 확인)")
    single = coded_df[~coded_df['cited_indicators'].astype(str).str.contains(';|,', regex=True, na=False)]
    lines.append(pd.crosstab(single['cited_indicators'], single['coding_primary']).to_string())
    lines.append("")
    n_specific = coded_df['coding_is_sentence_specific'].sum()
    lines.append(f"[3] 문장 특정 코멘트를 찾은 행: {n_specific}/{len(coded_df)} "
                 f"(나머지는 문항 전체 코멘트로 대체하거나 코멘트 없음)")
    n_no_comment = (coded_df['coding_primary'] == 'no_comment').sum()
    n_unclassified = (coded_df['coding_primary'] == 'unclassified').sum()
    lines.append(f"[4] 코멘트 없음: {n_no_comment}건 / 코멘트는 있으나 규칙에 안 걸림(수동 검토 필요): {n_unclassified}건")
    return "\n".join(lines)


def run(input_csv: str, output_dir: str, case_types=None, all_cases: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    if 'case_type' not in df.columns or 'free_text_combined' not in df.columns:
        raise ValueError("입력 CSV에 case_type / free_text_combined 컬럼이 없습니다. "
                          "export_disagreement_cases.py가 만든 all_sentence_cases.csv를 사용하세요.")
    types = None if all_cases else (case_types or DEFAULT_CASE_TYPES)
    coded = code_dataframe(df, case_types=types)
    coded.to_csv(os.path.join(output_dir, "free_text_coded.csv"), index=False)

    report = summarize(coded)
    with open(os.path.join(output_dir, "coding_summary.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n✅ 코딩 완료: {os.path.join(output_dir, 'free_text_coded.csv')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default='outputs/disagreement_review/all_sentence_cases.csv', help="export_disagreement_cases.py가 만든 all_sentence_cases.csv 경로")
    parser.add_argument("--output-dir", type=str, default="outputs/free_text_coding")
    parser.add_argument("--case-types", nargs="+", default=None,
                         help=f"코딩할 case_type 목록 (기본값: {DEFAULT_CASE_TYPES})")
    parser.add_argument("--all-cases", action="store_true", help="case_type 필터 없이 전체 행 코딩")

    args = parser.parse_args()


    if not args.input:
        parser.error("--input이 필요합니다")

    run(args.input, args.output_dir, case_types=args.case_types, all_cases=args.all_cases)


if __name__ == "__main__":
    main()