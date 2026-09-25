


import os
import re
import json
import time
import random
import argparse
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

import aggregate_ratings as ar
from code_free_text_disagreements import extract_targeted_comments, _sentence_id_to_num

CORE_SET_FLEISS_KAPPA = 0.444  
GROUND_TRUTH_DEFS = {'strict': {'supported'}, 'lenient': {'supported', 'partial'}}

DISALLOWED_JUDGE_SUBSTRINGS = ['qwen']  


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

def identify_items_by_min_raters(resp_df: pd.DataFrame, min_raters: int = 5) -> set:


    counts = resp_df.groupby('item_id')['rater_id'].nunique()
    return set(counts[counts >= min_raters].index)


def identify_core_set(resp_df: pd.DataFrame) -> set:


    return identify_items_by_min_raters(resp_df, min_raters=5)


def load_id_map(path: str) -> dict:

    key_df = pd.read_csv(path, dtype=str)
    if 'display_id' not in key_df.columns or 'internal_item_id' not in key_df.columns:
        raise ValueError(f"--id-map 파일에 display_id/internal_item_id 컬럼이 없습니다 "
                          f"(있는 컬럼: {list(key_df.columns)}).")
    return dict(zip(key_df['display_id'], key_df['internal_item_id']))


def _unwrap_items_list(raw):

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for wrapper_key in ('items', 'data', 'stimuli', 'results', 'calibration_items'):
            if wrapper_key in raw and isinstance(raw[wrapper_key], list):
                return raw[wrapper_key]
        return raw  
    raise ValueError(f"알 수 없는 stimuli-master 최상위 구조: {type(raw)}")


def load_stimuli_master(path: str, id_map_path: str = None) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    unwrapped = _unwrap_items_list(raw)

    if isinstance(unwrapped, dict):
        items = dict(unwrapped)
    else:
        if not unwrapped:
            raise ValueError("stimuli-master를 풀었더니 항목이 0개입니다.")
        sample = unwrapped[0]
        id_key = next((k for k in ('display_id', 'item_id', 'internal_item_id') if k in sample), None)
        if id_key is None:
            raise ValueError(f"stimuli-master 항목에서 id 필드를 못 찾았습니다. "
                              f"사용 가능한 키: {list(sample.keys())}")
        items = {it[id_key]: it for it in unwrapped}

    if id_map_path:
        
        
        
        internal_to_display = {v: k for k, v in load_id_map(id_map_path).items()}
        items = {internal_to_display.get(k, k): v for k, v in items.items()}

    return items


def _diagnose_id_mismatch(core_items: list, stimuli_keys: list) -> str:


    lines = [
        "core set의 item_id와 stimuli-master의 키가 거의/전혀 안 맞습니다.",
        f"  - 응답 파일(R1~R5) 쪽 item_id 예시: {sorted(core_items)[:5]}",
        f"  - stimuli-master 쪽 키 예시:        {sorted(stimuli_keys)[:5]}",
        "위 두 줄의 형식이 다르다면(예: 'Q133' vs 'IT-481'), stimuli-master가 display_id가 "
        "아니라 내부 item_id로 저장돼 있거나, 두 파일을 이어주는 매핑 파일"
        "(보통 internal_answer_key.csv의 display_id↔internal_item_id 컬럼)이 따로 필요하다는 뜻입니다. "
        "그 매핑 파일 경로를 알려주시면 --id-map 옵션으로 연결하는 코드를 추가하겠습니다.",
    ]
    return '\n'.join(lines)


SENTENCE_TEXT_KEY_CANDIDATES = ['en', 'text', 'sentence_text', 'sentence_en', 'sentence', 'hypothesis', 'ko']


def _get_sentence_text(s: dict) -> str:
    for k in SENTENCE_TEXT_KEY_CANDIDATES:
        if k in s and s[k]:
            return s[k]
    raise KeyError(f"문장 딕셔너리에서 텍스트 필드를 못 찾았습니다 (찾아본 키: "
                    f"{SENTENCE_TEXT_KEY_CANDIDATES}). 실제 키: {list(s.keys())}")


def majority_tag_from_counts(tags: list) -> tuple:
    tags = [t for t in tags if t]
    if not tags:
        return None, ''
    counts = pd.Series(tags).value_counts()
    top = counts.index[0]
    dist = ';'.join(f'{k}={v}' for k, v in counts.items())
    return top, dist


def build_case_table(responses_dir: str, stimuli_master_path: str, id_map_path: str = None,
                      min_raters: int = 5) -> pd.DataFrame:


    resp_df = ar.load_responses(responses_dir)
    target_items = identify_items_by_min_raters(resp_df, min_raters=min_raters)
    if not target_items:
        raise RuntimeError(f"min_raters={min_raters}인 문항을 찾지 못했습니다 — 응답 파일을 확인하세요.")
    print(f"[로딩] min_raters>={min_raters} 문항 {len(target_items)}개 식별됨"
          + (" (=core set)" if min_raters >= 5 else ""))

    stimuli = load_stimuli_master(stimuli_master_path, id_map_path=id_map_path)
    missing = [iid for iid in target_items if iid not in stimuli]
    if len(missing) == len(target_items):
        raise RuntimeError(_diagnose_id_mismatch(target_items, list(stimuli.keys())))
    if missing:
        print(f"[경고] stimuli-master에서 못 찾은 문항 {len(missing)}개(예: {missing[:3]}) — 제외합니다.")
    target_items = [iid for iid in target_items if iid in stimuli]

    target_resp = resp_df[resp_df['item_id'].isin(target_items)].copy()
    free_text_by_item = {}
    for iid, g in target_resp.groupby('item_id'):
        
        
        
        ft = g.dropna(subset=['free_text']).drop_duplicates(subset=['rater_id', 'free_text'])
        ft = ft[ft['free_text'].astype(str).str.strip() != '']
        free_text_by_item[iid] = ' | '.join(f"{r.rater_id}: {r.free_text}" for r in ft.itertuples())

    rows = []
    for iid in target_items:
        item = stimuli[iid]
        sent_lookup = {s.get('sentence_id') or f"S{i+1}": _get_sentence_text(s)
                       for i, s in enumerate(item.get('sentences', []))}
        for sent_id, text in sent_lookup.items():
            sent_num = _sentence_id_to_num(sent_id)
            tags = target_resp[(target_resp['item_id'] == iid) & (target_resp['sentence_id'] == sent_id)]['level1_tag'].tolist()
            majority_tag, dist = majority_tag_from_counts(tags)
            if majority_tag is None:
                continue
            targeted, general, is_specific = extract_targeted_comments(
                free_text_by_item.get(iid, ''), sent_num if sent_num is not None else -1)
            rationale = targeted if is_specific else general
            rows.append({
                'item_id': iid, 'sentence_id': sent_id, 'sentence_text': text,
                'skill': item.get('skill'), 'steps': item.get('steps'),
                'skill_cum_accuracy': item.get('skill_cum_accuracy'),
                'majority_tag': majority_tag, 'tag_distribution': dist, 'n_raters': len(tags),
                'rationale': rationale,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "case 테이블이 비었습니다 — 문항 ID는 매칭됐지만 문장 단위(sentence_id) "
            "매칭이나 라벨 집계에서 전부 걸러진 것으로 보입니다. stimuli-master의 sentences[].sentence_id "
            "형식(S1/S2/S3)이 응답 파일의 sentence_id와 같은지 확인하세요."
        )
    print(f"[로딩] 문장 {len(df)}개, 문항 {df['item_id'].nunique()}개 "
          f"(n_raters 분포: {df['n_raters'].value_counts().sort_index().to_dict()})")
    return df


def build_core_set_cases(responses_dir: str, stimuli_master_path: str, id_map_path: str = None) -> pd.DataFrame:

    return build_case_table(responses_dir, stimuli_master_path, id_map_path, min_raters=5)


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

TAG_DEFINITIONS = """- 지지됨(supported): 위 행동 기록으로 명확히 뒷받침됨
- 부분지지(partial): 방향은 맞으나 과장되었거나 일부만 맞음
- 근거없음(unsupported): 행동 기록과 무관하거나 모순됨
- 판단불가(unknown): 제시된 정보만으로는 판단하기 어려움"""


def format_steps(steps: list) -> str:
    lines = ["| step | correct | hint_count | response_time_ratio |", "|---|---|---|---|"]
    for s in steps:
        lines.append(f"| {s.get('step')} | {s.get('correct')} | {s.get('hint_count')} | "
                      f"{s.get('response_time_ratio'):.3f} |" if isinstance(s.get('response_time_ratio'), (int, float))
                      else f"| {s.get('step')} | {s.get('correct')} | {s.get('hint_count')} | {s.get('response_time_ratio')} |")
    return '\n'.join(lines)


def format_case_block(row: pd.Series, reveal_answer: bool) -> str:
    block = [
        f"스킬: {row['skill']} / 직전까지 이 스킬 누적 정답률: {row['skill_cum_accuracy']}",
        format_steps(row['steps']),
        f'서술: "{row["sentence_text"]}"',
    ]
    if reveal_answer:
        block.append(f"전문가 판정: {row['majority_tag']}")
        if row.get('rationale'):
            block.append(f"전문가 이유: {row['rationale']}")
    return '\n'.join(block)


def build_prompt(few_shot_rows: pd.DataFrame, target_row: pd.Series) -> str:
    parts = [
        "당신은 학생 행동 로그와 그것을 요약한 서술 문장을 비교해, 그 문장이 행동 기록으로 "
        "뒷받침되는지 판정하는 전문가입니다. 아래 네 가지 카테고리 중 하나로 판정하세요:",
        TAG_DEFINITIONS,
        "",
        "다음은 실제 전문가 평정자가 판정한 예시들입니다. 각 예시에는 전문가가 그렇게 "
        "판정한 이유도 함께 있습니다 — 이유를 참고해서 판단 기준을 이해하세요.",
        "",
    ]
    for i, (_, row) in enumerate(few_shot_rows.iterrows()):
        parts.append(f"### 예시 {i + 1}")
        parts.append(format_case_block(row, reveal_answer=True))
        parts.append("")

    parts.append("### 판정 대상")
    parts.append(format_case_block(target_row, reveal_answer=False))
    parts.append("")
    parts.append(
        '위 예시들의 판정 기준을 참고해서 판정 대상을 평가하세요. 다음 JSON 형식으로만 '
        '답하세요 (다른 텍스트 없이): {"label": "supported|partial|unsupported|unknown", '
        '"confidence_supported": 0-100 사이 정수 (100=확실히 지지됨, 0=확실히 근거없음), '
        '"reasoning": "한두 문장 이유"}'
    )
    return '\n'.join(parts)


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

def call_judge_llm_anthropic(prompt: str, model: str, api_key: str, max_retries: int = 3) -> dict:
    body = json.dumps({
        "model": model, "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }).encode('utf-8')
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"Content-Type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            text = ''.join(b['text'] for b in data['content'] if b['type'] == 'text')
            return parse_judge_output(text)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return {'label': None, 'confidence_supported': None, 'reasoning': f'호출 실패: {last_err}'}


def call_judge_llm_openai_compatible(prompt: str, model: str, api_base: str, api_key: str,
                                      max_retries: int = 3) -> dict:



    body = json.dumps({
        "model": model, "max_tokens": 300, "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode('utf-8')
    req = urllib.request.Request(
        api_base.rstrip('/') + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            text = data['choices'][0]['message']['content']
            return parse_judge_output(text)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return {'label': None, 'confidence_supported': None, 'reasoning': f'호출 실패: {last_err}'}


def call_judge_llm(prompt: str, model: str, api_key: str, backend: str = 'anthropic',
                    api_base: str = None, max_retries: int = 3) -> dict:
    if any(bad in model.lower() for bad in DISALLOWED_JUDGE_SUBSTRINGS):
        raise ValueError(f"판정 모델('{model}')이 생성기(Qwen)와 겹칩니다 — self-preference bias 위험. "
                          "다른 계열 모델을 쓰세요 (예: claude-sonnet-4-6, 또는 로컬 Llama/Ministral).")
    if backend == 'anthropic':
        return call_judge_llm_anthropic(prompt, model, api_key, max_retries)
    elif backend == 'openai_compatible':
        if not api_base:
            raise ValueError("backend='openai_compatible'에는 --api-base(로컬 서버 URL)가 필요합니다.")
        return call_judge_llm_openai_compatible(prompt, model, api_base, api_key or 'none', max_retries)
    else:
        raise ValueError(f"알 수 없는 backend: {backend} (anthropic | openai_compatible)")


def parse_judge_output(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {'label': None, 'confidence_supported': None, 'reasoning': f'JSON 파싱 실패: {text[:200]}'}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {'label': None, 'confidence_supported': None, 'reasoning': f'JSON 파싱 실패: {text[:200]}'}
    label = str(obj.get('label', '')).strip().lower()
    conf = obj.get('confidence_supported')
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = None
    return {'label': label if label in {'supported', 'partial', 'unsupported', 'unknown'} else None,
            'confidence_supported': conf, 'reasoning': obj.get('reasoning', '')}


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

def stratified_few_shot_sample(pool: pd.DataFrame, n_shots: int, rng: random.Random) -> pd.DataFrame:

    groups = {tag: g.index.tolist() for tag, g in pool.groupby('majority_tag')}
    for idxs in groups.values():
        rng.shuffle(idxs)
    picked = []
    tags_cycle = list(groups.keys())
    i = 0
    while len(picked) < min(n_shots, len(pool)):
        tag = tags_cycle[i % len(tags_cycle)]
        if groups[tag]:
            picked.append(groups[tag].pop())
        i += 1
        if all(not v for v in groups.values()):
            break
    return pool.loc[picked]


def run_nested_cv(df: pd.DataFrame, n_folds: int, n_shots: int, model: str, api_key: str,
                   dry_run: bool, mock_fn=None, seed: int = 42, backend: str = 'anthropic',
                   api_base: str = None) -> pd.DataFrame:
    rng = random.Random(seed)
    items = sorted(df['item_id'].unique())
    rng.shuffle(items)
    folds = np.array_split(items, n_folds)

    predictions = []
    n_calls, n_early_failures = 0, 0
    for fold_i, test_items in enumerate(folds):
        test_items = set(test_items)
        train_df = df[~df['item_id'].isin(test_items)]
        test_df = df[df['item_id'].isin(test_items)]
        few_shot = stratified_few_shot_sample(train_df, n_shots, rng)

        print(f"[fold {fold_i + 1}/{n_folds}] train 문항 {train_df['item_id'].nunique()}개 "
              f"(few-shot {len(few_shot)}개 사용) / test 문장 {len(test_df)}개")

        for _, row in test_df.iterrows():
            prompt = build_prompt(few_shot, row)
            if dry_run:
                result = mock_fn(row, prompt) if mock_fn else {
                    'label': None, 'confidence_supported': None, 'reasoning': 'dry-run'}
            else:
                result = call_judge_llm(prompt, model, api_key, backend=backend, api_base=api_base)

            n_calls += 1
            if result['confidence_supported'] is None:
                n_early_failures += 1
                if n_calls == n_early_failures:  
                    print(f"  ⚠ {n_calls}번째 호출까지 전부 실패. 마지막 오류: {result['reasoning'][:300]}")
                if not dry_run and n_calls == n_early_failures and n_calls >= 3:
                    raise RuntimeError(
                        f"처음 {n_calls}건이 전부 실패했습니다 — 모델명/API 키/backend 설정을 먼저 "
                        f"확인하세요 (위 오류 메시지 참고). 나머지 문장을 계속 호출해서 API 비용을 "
                        f"낭비하지 않도록 여기서 중단합니다. 문제를 고친 뒤 다시 실행하세요."
                    )

            predictions.append({
                'fold': fold_i, 'item_id': row['item_id'], 'sentence_id': row['sentence_id'],
                'majority_tag': row['majority_tag'],
                'pred_label': result['label'], 'pred_confidence_supported': result['confidence_supported'],
                'pred_reasoning': result['reasoning'],
            })
    return pd.DataFrame(predictions)


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

def _auc_mann_whitney(scores: np.ndarray, truth_positive: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    truth_positive = np.asarray(truth_positive, dtype=bool)
    n_pos, n_neg = int(truth_positive.sum()), int((~truth_positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = pd.Series(scores).rank(method='average').to_numpy()
    u = ranks[truth_positive].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def evaluate(preds: pd.DataFrame) -> dict:
    scored = preds.dropna(subset=['pred_confidence_supported'])
    n_failed = len(preds) - len(scored)
    if n_failed:
        print(f"[경고] {n_failed}건은 판정 실패(파싱 실패/API 오류)로 평가에서 제외됩니다.")

    results = {'n_total': len(preds), 'n_scored': len(scored), 'n_failed': n_failed}
    if n_failed:
        failed = preds[preds['pred_confidence_supported'].isna()]
        results['failure_reason_counts'] = (
            failed['pred_reasoning'].value_counts().head(5).to_dict()
        )
    for gt_key, positive_tags in GROUND_TRUTH_DEFS.items():
        sub = scored[scored['majority_tag'] != 'unknown']
        truth = sub['majority_tag'].isin(positive_tags).to_numpy()
        auc = _auc_mann_whitney(sub['pred_confidence_supported'].to_numpy(), truth)
        results[f'auc_{gt_key}'] = auc

    valid_label = scored.dropna(subset=['pred_label'])
    if len(valid_label):
        results['accuracy_4way'] = (valid_label['pred_label'] == valid_label['majority_tag']).mean()
    return results


def format_report(results: dict) -> str:
    min_raters = results.get('min_raters', 5)
    title = "Few-shot LLM 판정관 실험 결과 (nested CV, core set)" if min_raters >= 5 else \
        f"Few-shot LLM 판정관 실험 결과 (nested CV, min_raters>={min_raters} 확장셋)"
    lines = ["=" * 70, title, "=" * 70, ""]
    lines.append(f"평가된 문장: {results['n_scored']}/{results['n_total']} "
                 f"(판정 실패 {results['n_failed']}건 제외)")
    if results.get('failure_reason_counts'):
        lines.append("")
        lines.append("[실패 사유 상위 5건 — predictions.csv의 pred_reasoning 컬럼에 전체 있음]")
        for reason, cnt in results['failure_reason_counts'].items():
            lines.append(f"  ({cnt}건) {str(reason)[:200]}")
    lines.append("")
    lines.append(f"AUROC (strict, supported만 정답)  = {results.get('auc_strict', float('nan')):.3f}")
    lines.append(f"AUROC (lenient, partial도 정답)   = {results.get('auc_lenient', float('nan')):.3f}")
    if 'accuracy_4way' in results:
        lines.append(f"4-way 정확도 (참고용)             = {results['accuracy_4way']:.3f}")
    lines.append("")
    if min_raters >= 5:
        lines.append(f"⚠ 참고: core set Fleiss κ = {CORE_SET_FLEISS_KAPPA} (moderate) — 전문가 5인끼리도 "
                     f"이 정도만 일치합니다. AUROC가 1.0에 크게 못 미쳐도, 이 상한 근처면 '판정관이 "
                     f"인간 수준에 도달했다'는 뜻이지 '판정관이 나쁘다'는 뜻이 아닙니다.")
    else:
        lines.append(f"⚠ 주의: 이번 실행은 min_raters={min_raters}로 core-set(5인 전원) 밖의 문항까지 "
                     f"포함합니다 — core set Fleiss κ={CORE_SET_FLEISS_KAPPA}는 5인 전원이 평정한 40개 "
                     f"문항에서만 계산된 값이라 이 결과 전체의 신뢰도 상한으로 그대로 갖다 쓰면 안 됩니다. "
                     f"평정자가 적은 문항일수록 majority_tag 자체가 더 불안정하다는 점을 감안하세요 — "
                     f"predictions.csv의 n_raters 컬럼으로 원하는 만큼만 다시 걸러 재계산할 수 있습니다.")
    lines.append(f"⚠ 비교 기준선: Stage 3 NLI의 AUROC ≈ 0.46~0.48 (본 논문 4.5절, core-set 기준).")
    lines.append("=" * 70)
    return '\n'.join(lines)


# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════

def run(responses_dir, stimuli_master, n_folds, n_shots, model, api_key, output_dir,
        dry_run=False, mock_fn=None, backend='anthropic', api_base=None, id_map=None,
        min_raters=5):
    os.makedirs(output_dir, exist_ok=True)
    df = build_case_table(responses_dir, stimuli_master, id_map_path=id_map, min_raters=min_raters)
    df.to_csv(os.path.join(output_dir, 'core_set_cases.csv'), index=False)

    preds = run_nested_cv(df, n_folds, n_shots, model, api_key, dry_run, mock_fn,
                           backend=backend, api_base=api_base)
    preds.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    results = evaluate(preds)
    results['min_raters'] = min_raters
    report = format_report(results)
    print('\n' + report)
    with open(os.path.join(output_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--responses-dir', type=str, default='data/ratings')
    parser.add_argument('--stimuli-master', type=str, default='data/stimuli/stimuli_master.json')
    parser.add_argument('--id-map', type=str, default='data/stimuli/internal_answer_key.csv',
                         help="internal_answer_key.csv 경로 (display_id, internal_item_id 컬럼 필요). "
                              "stimuli-master의 항목이 display_id(Qxxx)가 아니라 내부 ID(IT-xxx)로 "
                              "키가 잡혀 있을 때 이걸로 연결한다.")
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-shots', type=int, default=8)
    parser.add_argument('--min-raters', type=int, default=5,
                         help="문항을 포함시키는 최소 평정자 수. 기본값 5(=core set, 5인 전원 평정 "
                              "40개 문항만). 표본을 늘리려면 낮추세요(예: 1이면 응답이 하나라도 있는 "
                              "문항을 전부 사용). 낮출수록 문항별 정답(majority_tag) 신뢰도가 "
                              "평정자 수에 따라 들쭉날쭉해진다는 점을 report.txt가 경고로 알려줍니다.")
    parser.add_argument('--model', type=str, default="mistralai/Ministral-8B-Instruct-2410", #'meta-llama/Meta-Llama-3.1-8B-Instruct',
                         help='판정 모델. 생성기(Qwen) 계열은 self-preference bias 위험으로 거부됩니다. '
                              'Anthropic 모델 문자열은 바뀔 수 있으니 https://docs.claude.com 에서 '
                              '최신 값을 확인하세요')
    parser.add_argument('--backend', type=str, default='openai_compatible', choices=['anthropic', 'openai_compatible'], #['TRITON_ATTN', 'FLEX_ATTENTION']
                         help="'anthropic'은 api.anthropic.com 호출. 'openai_compatible'은 vLLM/TGI 등 "
                              "로컬 서빙이 노출하는 OpenAI 호환 엔드포인트 호출 (Llama/Ministral 로컬 서빙용).")
    parser.add_argument('--api-base', type=str, default='http://localhost:8000/v1',
                         help="backend='openai_compatible'일 때 로컬 서버 URL (예: http://localhost:8000/v1)")
    parser.add_argument('--api-key', type=str, default=os.environ.get('ANTHROPIC_API_KEY'))
    parser.add_argument('--output-dir', type=str, default='outputs/few_shot_judge')
    parser.add_argument('--dry-run', action='store_true', help='실제 API 호출 없이 목(mock) 판정관으로 실행')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()


    if not args.responses_dir or not args.stimuli_master:
        parser.error('--responses-dir와 --stimuli-master가 필요합니다')
    if not args.dry_run and args.backend == 'anthropic' and not args.api_key:
        parser.error('backend=anthropic 실제 실행에는 --api-key 또는 ANTHROPIC_API_KEY 환경변수가 필요합니다 (또는 --dry-run)')
    if not args.dry_run and args.backend == 'openai_compatible' and not args.api_base:
        parser.error('backend=openai_compatible 실행에는 --api-base(로컬 서버 URL)가 필요합니다')

    run(args.responses_dir, args.stimuli_master, args.n_folds, args.n_shots,
        args.model, args.api_key, args.output_dir, dry_run=args.dry_run,
        backend=args.backend, api_base=args.api_base, id_map=args.id_map,
        min_raters=args.min_raters)


def _make_synthetic_stimuli(item_ids_by_sentence: dict, rng: np.random.RandomState) -> dict:


    skills = ['area', 'percent-of', 'inequality-solving', 'discount', 'multiplication']
    stimuli = {}
    for item_id, sent_ids in item_ids_by_sentence.items():
        base = int(rng.randint(100, 900))
        stimuli[item_id] = {
            'display_id': item_id, 'skill': rng.choice(skills),
            'skill_cum_accuracy': round(float(rng.uniform(0.1, 0.6)), 2),
            'steps': [{'step': base + i, 'correct': bool(rng.rand() < 0.4),
                       'hint_count': int(rng.choice([0, 0, 1, 2, 3])),
                       'response_time_ratio': round(float(rng.uniform(0.1, 2.2)), 3)} for i in range(4)],
            'sentences': [{'sentence_id': sid, 'text': f'({item_id}/{sid}) synthetic sentence for testing.'}
                          for sid in sent_ids],
        }
    return stimuli


if __name__ == '__main__':
    main()