import os

import argparse
import json
import os
import pickle
import random
import sys
from collections import defaultdict
 
import numpy as np
import pandas as pd
 
from stage1_grounding import SymbolicFact, build_symbolic_facts
from stage2_generation import build_prompt, parse_generated_narrative, validate_citations, CitedSentence
from stage3_verification import summarize_results, VerificationResult, _fact_to_premise_text
 
try:
    from stage1_grounding import HIGH_THRESHOLD
except ImportError:
    HIGH_THRESHOLD = 0.5
 
INDICATOR_TO_CATEGORY = {
    'overconfidence': 'Monitoring',
    'underconfidence': 'Monitoring',
    'strategic_help': 'Regulation',
    'cognitive_avoidance': 'Regulation',
    'productive_struggle': 'Struggle',
    'unproductive_frustration': 'Affective',
    'boredom_offtask': 'Affective',
    'lucky_guess': 'Reliability',
    'slipping': 'Reliability',
}
 
 
COLUMN_MAP = {
    'student_id': 'ITEST_id',
    'skill': 'skill',
    'step': 'step_idx',
    'hint_count': 'hintCount',
    'correct': 'correct',
    'sc': 'SC',
    'response_time_ratio': 'response_time_ratio',
    'skill_cum_accuracy': 'skill_cum_accuracy',
    'hint_count_display': 'hintCount_raw',
    'm_overconfidence': 'm_overconfidence',
    'm_underconfidence': 'm_underconfidence',
    'm_strategic_help': 'm_strategic_help',
    'm_cognitive_avoidance': 'm_cognitive_avoidance',
    'm_productive_struggle': 'm_productive_struggle',
    'm_unproductive_frustration': 'm_unproductive_frustration',
    'm_lucky_guess': 'm_lucky_guess',
    'm_slipping': 'm_slipping',
    'm_boredom_offtask': 'm_boredom_offtask',
}
REQUIRED_FIELDS = ['student_id', 'skill', 'step', 'hint_count', 'correct'] + \
                  [k for k in COLUMN_MAP if k.startswith('m_')]
 
META_KEYS = [k for k in COLUMN_MAP if k.startswith('m_')]
CATEGORIES = ['Monitoring', 'Regulation', 'Struggle', 'Affective', 'Reliability']
 
 
 
def load_source_data(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
 
    missing = [COLUMN_MAP[k] for k in REQUIRED_FIELDS if COLUMN_MAP[k] not in df.columns]
    if missing:
        raise ValueError(
            "Required columns are missing from the input file.\n"
            f"  Missing columns: {missing}\n"
            f"  Available columns: {list(df.columns)}\n"
            "  Update COLUMN_MAP at the top of the script to match the input columns."
        )
    return df
 
 
def categorize_fact(fact: SymbolicFact, sc: float = None) -> str:
    if fact.indicator == 'confidence_triad':
        if fact.status == 'AMBIGUOUS':
            if sc is not None and sc < 0.5 - 0.15:
                return 'Reliability'
            elif sc is not None and sc > 0.5 + 0.15:
                return 'Struggle'
            return 'Reliability'
        return INDICATOR_TO_CATEGORY.get(fact.indicator, 'Reliability')
    return INDICATOR_TO_CATEGORY.get(fact.indicator, None)
 
 
def compute_row_facts(row: pd.Series) -> dict:
    meta_vector = {mk: float(row[COLUMN_MAP[mk]]) for mk in META_KEYS}
    hint_count = float(row[COLUMN_MAP['hint_count']])
    sc = None
    if COLUMN_MAP.get('sc') and COLUMN_MAP['sc'] in row.index and pd.notna(row[COLUMN_MAP['sc']]):
        sc = float(row[COLUMN_MAP['sc']])
 
    facts = build_symbolic_facts(meta_vector, hint_count, sc, lang='en')
 
    strong_facts = [f for f in facts if f.status in ('HIGH', 'AMBIGUOUS')]
    if not strong_facts:
        return {'has_signal': False, 'category': None, 'dominant_indicator': None, 'facts': facts}
 
    def sort_key(f):
        return f.value if f.value is not None else HIGH_THRESHOLD
    dominant = max(strong_facts, key=sort_key)
    category = categorize_fact(dominant, sc)
 
    return {
        'has_signal': category is not None,
        'category': category,
        'dominant_indicator': dominant.indicator,
        'dominant_status': dominant.status,
        'facts': facts,
    }
 
 
def run_phase_a(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[Phase A] Running Stage 1 symbolization for {len(df):,} interactions...")
    records = []
    for idx, row in df.iterrows():
        r = compute_row_facts(row)
        records.append({
            'row_idx': idx,
            'student_id': row[COLUMN_MAP['student_id']],
            'skill': row[COLUMN_MAP['skill']],
            'step': row[COLUMN_MAP['step']],
            'has_signal': r['has_signal'],
            'category': r['category'],
            'dominant_indicator': r['dominant_indicator'],
            'dominant_status': r.get('dominant_status'),
        })
    out = pd.DataFrame.from_records(records)
    n_signal = out['has_signal'].sum()
    print(f"[Phase A] Complete. Interactions with signals: {n_signal:,} / {len(out):,} "
          f"({n_signal/len(out)*100:.1f}%)")
    print("[Phase A] Category distribution:")
    print(out[out['has_signal']]['category'].value_counts())
    return out
 
 
 
FULL_POOL_THRESHOLD = 2000


def select_candidate_pool(phase_a_df: pd.DataFrame, pool_size: int, seed: int) -> pd.DataFrame:
    signal_df = phase_a_df[phase_a_df['has_signal']].copy()
    per_cat = max(10, pool_size // len(CATEGORIES))
    rng = np.random.RandomState(seed)
    parts = []
    for cat in CATEGORIES:
        cat_df = signal_df[signal_df['category'] == cat]
        if len(cat_df) <= FULL_POOL_THRESHOLD:
            n = len(cat_df)
            sampled = cat_df
            print(f"[Info] Category '{cat}' has only {n} items (<{FULL_POOL_THRESHOLD}); "
                  "using the full population instead of equal allocation.")
            if n < 3 * 6:
                print(f"       Warning: even the full population ({n} items) may be below "
                      f"the minimum of {3*6} items across three tertiles.")
        else:
            n = min(per_cat, len(cat_df))
            if n < per_cat:
                print(f"[Warning] Category '{cat}' has only {len(cat_df)} signal-bearing interactions, "
                      f"below the target of {per_cat}; consider adjusting HIGH_THRESHOLD.")
            sampled = cat_df.sample(n=n, random_state=seed) if n > 0 else cat_df
        parts.append(sampled)
    pool = pd.concat(parts, ignore_index=True)
    print(f"[Phase B] Candidate-pool size: {len(pool)} (target {per_cat} per large category; "
          "full population for sparse categories)")
    return pool
 
 
def build_stimulus_context(df: pd.DataFrame, student_id, step, skill,
                            window: int = 4) -> dict:
    student_rows = df[
        (df[COLUMN_MAP['student_id']] == student_id) & (df[COLUMN_MAP['skill']] == skill)
    ].sort_values(COLUMN_MAP['step'])
    upto = student_rows[student_rows[COLUMN_MAP['step']] <= step].tail(window)
 
    steps_out = []
    for _, r in upto.iterrows():
        hc_display_col = COLUMN_MAP.get('hint_count_display')
        if hc_display_col and hc_display_col in r.index and pd.notna(r[hc_display_col]):
            hint_count_val = int(round(float(r[hc_display_col])))
        else:
            hint_count_val = float(r[COLUMN_MAP['hint_count']])
        entry = {
            'step': int(r[COLUMN_MAP['step']]),
            'correct': bool(r[COLUMN_MAP['correct']]),
            'hint_count': hint_count_val,
        }
        rt_col = COLUMN_MAP.get('response_time_ratio')
        if rt_col and rt_col in r.index and pd.notna(r[rt_col]):
            entry['response_time_ratio'] = float(r[rt_col])
        steps_out.append(entry)
 
    cum_acc = None
    acc_col = COLUMN_MAP.get('skill_cum_accuracy')
    if acc_col and acc_col in upto.columns and len(upto) > 0 and pd.notna(upto.iloc[-1].get(acc_col, np.nan)):
        cum_acc = float(upto.iloc[-1][acc_col])
 
    return {'skill': skill, 'steps': steps_out, 'skill_cum_accuracy': cum_acc}
 
 
def generate_and_verify_item(item_row, df, engine, generator, verifier, debug_dump_fh=None):
    src_row = df.loc[item_row['row_idx']]
    meta_vector = {mk: float(src_row[COLUMN_MAP[mk]]) for mk in META_KEYS}
    hint_count = float(src_row[COLUMN_MAP['hint_count']])
    sc = None
    if COLUMN_MAP.get('sc') and COLUMN_MAP['sc'] in src_row.index and pd.notna(src_row[COLUMN_MAP['sc']]):
        sc = float(src_row[COLUMN_MAP['sc']])
    facts = build_symbolic_facts(meta_vector, hint_count, sc, lang='en')
 
    context = build_stimulus_context(
        df, item_row['student_id'], item_row['step'], item_row['skill']
    )
 
    if engine == 'template':
        sentences = []
        for f in facts:
            if f.status not in ('HIGH', 'AMBIGUOUS'):
                continue
            if f.status == 'AMBIGUOUS':
                text = f"{f.indicator} 관련 신호가 있으나 원인은 확실하지 않을 수 있습니다."
            else:
                text = f"{f.indicator} 관련 패턴이 뚜렷하게 관찰되었습니다."
            sentences.append(CitedSentence(text=text, cited_indicators=[f.indicator]))
        citation_check = validate_citations(sentences, {f.indicator for f in facts})
        results = []
        for s in sentences:
            fact = next(f for f in facts if f.indicator == s.cited_indicators[0])
            if fact.status == 'AMBIGUOUS':
                results.append(VerificationResult(s.text, s.cited_indicators, 'PASS',
                                                    nli_scores={'entailment': 0.6}))
            else:
                results.append(VerificationResult(s.text, s.cited_indicators, 'PASS',
                                                    nli_scores={'entailment': 0.9}))
        had_repetition = False
        bilingual_sentences = [
            {'en': s.text, 'ko': s.text, 'cited_indicators': s.cited_indicators, 'translation_ok': True}
            for s in sentences
        ]
    else:
        gen_result = generator.generate(
            facts, context={'skill': item_row['skill'], 'step': item_row['step']}, language='en'
        )
        sentences = gen_result['sentences']
        citation_check = gen_result['citation_check']
        had_repetition = gen_result.get('had_repetition', False)
        results = verifier.verify_narrative(sentences, facts, lang='en')
        bilingual_sentences = generator.translate_to_korean(sentences)
 
    summary = summarize_results(results)
    mean_entailment = np.mean([
        r.nli_scores['entailment'] for r in results if r.nli_scores is not None
    ]) if results else np.nan
 
    if debug_dump_fh is not None:
        facts_by_indicator = {f.indicator: f for f in facts}
        for r in results:
            premises = [
                _fact_to_premise_text(facts_by_indicator[ind])
                for ind in r.cited_indicators if ind in facts_by_indicator
            ]
            debug_dump_fh.write(json.dumps({
                'item_row_idx': int(item_row['row_idx']),
                'category': item_row['category'],
                'sentence_text': r.sentence_text,
                'cited_indicators': r.cited_indicators,
                'premises': premises,
                'status': r.status,
                'nli_scores': r.nli_scores,
                'detail': r.detail,
            }, ensure_ascii=False) + '\n')
 
    if had_repetition:
        mean_entailment = np.nan
 
    return {
        'item_id': f"IT-{item_row['row_idx']}",
        'student_id': item_row['student_id'],
        'skill': item_row['skill'],
        'step': item_row['step'],
        'category': item_row['category'],
        'dominant_indicator': item_row['dominant_indicator'],
        'context': context,
        'facts': facts,
        'sentences': sentences,
        'bilingual_sentences': bilingual_sentences,
        'verification': results,
        'faithfulness_rate': summary.get('faithfulness_rate', np.nan),
        'mean_entailment': mean_entailment,
        'structural_violation_rate': citation_check.get('structural_violation_rate'),
    }
 
 
def run_phase_b(candidate_pool: pd.DataFrame, df: pd.DataFrame, engine: str, model_name: str = None,
                 trust_remote_code: bool = False, debug_nli_dump: str = None):
    generator = verifier = None
    if engine == 'llama':
        from stage2_generation import NarrativeGenerator
        from stage3_verification import NLIVerifier
        kwargs = {}
        if model_name:
            kwargs['model_name'] = model_name
        generator = NarrativeGenerator(trust_remote_code=trust_remote_code, **kwargs)
        verifier = NLIVerifier()

    model_desc = (', model=' + generator.model_name) if generator else ''
    print(f"[Phase B] Running Stage 2 (engine={engine}{model_desc}) and Stage 3 for "
          f"{len(candidate_pool)} candidates...")
    debug_dump_fh = None
    if debug_nli_dump:
        debug_dump_fh = open(debug_nli_dump, 'w', encoding='utf-8')
        print(f"[Diagnostic] Writing sentence-level NLI premise/hypothesis/score values to {debug_nli_dump}.")
    try:
        items = []
        for i, (_, row) in enumerate(candidate_pool.iterrows()):
            item = generate_and_verify_item(row, df, engine, generator, verifier, debug_dump_fh)
            items.append(item)
            if (i + 1) % 20 == 0:
                print(f"  ... {i+1}/{len(candidate_pool)}")
    finally:
        if debug_dump_fh is not None:
            debug_dump_fh.close()
    print(f"[Phase B] Complete. Items generated and verified: {len(items)}")
    return items

 
 
 
def stratified_sample(items: list, n_target: int, min_per_cell: int, max_per_cell: int, seed: int):
    valid_items = [it for it in items if not np.isnan(it['mean_entailment'])]
    if not valid_items:
        raise RuntimeError("유효한 mean_entailment 값을 가진 문항이 없습니다.")
 
    scores = np.array([it['mean_entailment'] for it in valid_items])
    q1, q2 = np.quantile(scores, [1/3, 2/3])
    for it in valid_items:
        s = it['mean_entailment']
        it['nli_tertile'] = '하' if s <= q1 else ('중' if s <= q2 else '상')
 
    rng = random.Random(seed)
    cells = defaultdict(list)
    for it in valid_items:
        cells[(it['category'], it['nli_tertile'])].append(it)
 
    selected = []
    report_lines = [f"NLI tertile 경계: 하<= {q1:.3f} < 중 <= {q2:.3f} < 상"]
    for cat in CATEGORIES:
        for tertile in ['상', '중', '하']:
            pool = cells.get((cat, tertile), [])
            rng.shuffle(pool)
            target = min(max_per_cell, len(pool))
            take = pool[:max(min(target, max_per_cell), 0)]
            if len(pool) < min_per_cell:
                report_lines.append(f"  ⚠ [{cat} / {tertile}] 후보 {len(pool)}개 (target 최소 {min_per_cell}개 미달)")
            else:
                report_lines.append(f"  [{cat} / {tertile}] 후보 {len(pool)}개 중 {len(take)}개 선택")
            selected.extend(take)
 
    if len(selected) > n_target:
        rng.shuffle(selected)
        selected = selected[:n_target]
    elif len(selected) < n_target:
        report_lines.append(f"  ⚠ 총 선택 {len(selected)}개가 target({n_target})에 못 미칩니다. "
                             f"--candidate-pool-size를 늘리거나 임계값을 재검토하세요.")
 
    return selected, "\n".join(report_lines)
 
 
 
def sanitize_donor_sentences(sentences: list, bilingual_sentences: list, donor_skill: str, host_skill: str):
    import re
    sanitized = []
    sanitized_bilingual = []
    for s, bs in zip(sentences, bilingual_sentences):
        text = s.text
        if donor_skill and donor_skill != host_skill and re.search(re.escape(donor_skill), text):
            text = re.sub(re.escape(donor_skill), 'that skill', text)
        sanitized.append(CitedSentence(text=text, cited_indicators=s.cited_indicators))

        bs = dict(bs)
        if donor_skill and donor_skill != host_skill:
            if bs.get('en') and re.search(re.escape(donor_skill), bs['en']):
                bs['en'] = re.sub(re.escape(donor_skill), 'that skill', bs['en'])
            if bs.get('ko') and re.search(re.escape(donor_skill), bs['ko']):
                bs['ko'] = re.sub(re.escape(donor_skill), '해당 스킬', bs['ko'])
        sanitized_bilingual.append(bs)
    return sanitized, sanitized_bilingual
 
 
def construct_negative_controls(selected_items: list, full_pool: list, n_controls: int, seed: int):
    used_ids = {it['item_id'] for it in selected_items}
    remaining = [it for it in full_pool if it['item_id'] not in used_ids and it['sentences']]
    rng = random.Random(seed)
    rng.shuffle(remaining)
 
    controls = []
    donors = list(remaining)
    hosts = list(remaining)
    rng.shuffle(donors)
 
    used_as_host = set()
    used_as_donor = set()
    for host in hosts:
        if len(controls) >= n_controls:
            break
        if host['item_id'] in used_as_host:
            continue
        candidate_donors = [d for d in donors
                             if d['category'] != host['category']
                             and d['item_id'] not in used_as_donor
                             and d['item_id'] != host['item_id']]
        if not candidate_donors:
            continue
        donor = rng.choice(candidate_donors)
        donor_sentences, donor_bilingual = sanitize_donor_sentences(
            donor['sentences'], donor['bilingual_sentences'], donor['skill'], host['skill']
        )
 
        controls.append({
            'item_id': f"NC-{host['item_id']}-{donor['item_id']}",
            'student_id': host['student_id'],
            'skill': host['skill'],
            'step': host['step'],
            'category': host['category'],
            'context': host['context'],
            'sentences': donor_sentences,
            'bilingual_sentences': donor_bilingual,
            'is_negative_control': True,
            'true_origin_item_id': donor['item_id'],
            'true_origin_category': donor['category'],
            'host_item_id': host['item_id'],
        })
        used_as_host.add(host['item_id'])
        used_as_donor.add(donor['item_id'])
 
    if len(controls) < n_controls:
        print(f"[Warning] Created {len(controls)} of {n_controls} requested negative controls. "
              "Increase --candidate-pool-size to obtain more candidates.")
    return controls
 
 
 
def stratified_shuffle_no_adjacent_controls(items: list, seed: int, min_gap: int = 4):
    rng = random.Random(seed)
    controls = [it for it in items if it.get('is_negative_control')]
    normals = [it for it in items if not it.get('is_negative_control')]
    rng.shuffle(controls)
    rng.shuffle(normals)
 
    result = list(normals)
    for c in controls:
        possible_positions = list(range(0, len(result) + 1))
        rng.shuffle(possible_positions)
        placed = False
        for pos in possible_positions:
            left_ok = all(
                not result[j].get('is_negative_control')
                for j in range(max(0, pos - min_gap), pos)
            )
            right_ok = all(
                not result[j].get('is_negative_control')
                for j in range(pos, min(len(result), pos + min_gap))
            )
            if left_ok and right_ok:
                result.insert(pos, c)
                placed = True
                break
        if not placed:
            result.insert(rng.randrange(len(result) + 1), c)
    return result
 
 
def assign_raters(real_items: list, control_items: list, rater_ids: list, core_n: int, seed: int):
    all_items = real_items + control_items
    rng = random.Random(seed)
 
    by_cat = defaultdict(list)
    for it in all_items:
        by_cat[it['category']].append(it)
    for v in by_cat.values():
        rng.shuffle(v)
 
    core_set = []
    remaining_pool = list(all_items)
    cats = list(by_cat.keys())
    i = 0
    core_ids = set()
    while len(core_set) < min(core_n, len(all_items)):
        cat = cats[i % len(cats)]
        pool = by_cat[cat]
        if pool:
            picked = pool.pop()
            core_set.append(picked)
            core_ids.add(picked['item_id'])
        i += 1
        if i > 10000:
            break
 
    remaining = [it for it in all_items if it['item_id'] not in core_ids]
    rng.shuffle(remaining)
 
    n_raters = len(rater_ids)
    splits = {rid: [] for rid in rater_ids}
    for idx, it in enumerate(remaining):
        splits[rater_ids[idx % n_raters]].append(it)
 
    rater_assignments = {}
    for rid in rater_ids:
        combined = core_set + splits[rid]
        ordered = stratified_shuffle_no_adjacent_controls(combined, seed=hash((seed, rid)) % (2**31))
        rater_assignments[rid] = ordered
 
    return rater_assignments, core_set
 
 
 
def assign_opaque_display_ids(all_items: list, calibration_items: list, seed: int) -> dict:
    everything = list(all_items) + [c for c in calibration_items if c not in all_items]
    ids = [it['item_id'] for it in everything]
    rng = random.Random(seed)
    order = list(range(len(ids)))
    rng.shuffle(order)
    width = max(3, len(str(len(ids))))
    mapping = {}
    for rank, idx in zip(order, range(len(ids))):
        mapping[ids[idx]] = f"Q{rank+1:0{width}d}"
    for i, it in enumerate(calibration_items):
        mapping[it['item_id']] = f"C{i+1}"
    return mapping
 
 
def to_public_item(it: dict, display_id: str) -> dict:
    bilingual = it.get('bilingual_sentences') or []
    sentences_out = []
    for i, s in enumerate(it['sentences']):
        bs = bilingual[i] if i < len(bilingual) else {}
        ko_text = bs.get('ko')
        if not ko_text:
            ko_text = "[번역 실패 — 연구팀에 문의 필요]"
        sentences_out.append({'id': f"S{i+1}", 'en': s.text, 'ko': ko_text})
    return {
        'item_id': display_id,
        'skill': it['context']['skill'],
        'steps': it['context']['steps'],
        'skill_cum_accuracy': it['context'].get('skill_cum_accuracy'),
        'sentences': sentences_out,
    }
 
 
def export_outputs(rater_assignments: dict, core_set: list, real_items: list,
                    control_items: list, calibration_items: list, sampling_report: str,
                    output_dir: str, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    forms_dir = os.path.join(output_dir, 'rater_forms')
    os.makedirs(forms_dir, exist_ok=True)
 
    all_items = real_items + control_items
    display_id_map = assign_opaque_display_ids(all_items, calibration_items, seed)
 
    for rid, items in rater_assignments.items():
        public_items = [to_public_item(it, display_id_map[it['item_id']]) for it in items]
        with open(os.path.join(forms_dir, f"{rid}.json"), 'w', encoding='utf-8') as f:
            json.dump({'rater_id': rid, 'items': public_items}, f, ensure_ascii=False, indent=2)

    n_translation_fail = 0
    for it in all_items:
        for bs in (it.get('bilingual_sentences') or []):
            if not bs.get('ko'):
                n_translation_fail += 1
    if n_translation_fail:
        print(f"\nWarning: {n_translation_fail} sentences failed translation; "
              f"'[번역 실패 — 연구팀에 문의 필요]'가 그대로 노출됩니다. 배포 전에 "
              f"internal_answer_key_sentences.csv에서 translation_ok=False인 행을 찾아 재번역하거나 "
              f"표집에서 제외하세요.")

    key_rows = []
    for it in all_items:
        key_rows.append({
            'display_id': display_id_map[it['item_id']],
            'internal_item_id': it['item_id'],
            'is_negative_control': it.get('is_negative_control', False),
            'true_origin_item_id': it.get('true_origin_item_id'),
            'category': it['category'],
            'nli_tertile': it.get('nli_tertile'),
            'mean_entailment': it.get('mean_entailment'),
            'faithfulness_rate': it.get('faithfulness_rate'),
            'in_core_set': it['item_id'] in {c['item_id'] for c in core_set},
        })
    pd.DataFrame(key_rows).to_csv(os.path.join(output_dir, 'internal_answer_key.csv'), index=False)

    sentence_key_rows = []
    for it in real_items:
        display_id = display_id_map[it['item_id']]
        bilingual = it.get('bilingual_sentences') or []
        for i, (s, verif) in enumerate(zip(it['sentences'], it['verification'])):
            bs = bilingual[i] if i < len(bilingual) else {}
            sentence_key_rows.append({
                'display_id': display_id,
                'sentence_id': f"S{i+1}",
                'sentence_en': s.text,
                'sentence_ko': bs.get('ko'),
                'translation_ok': bs.get('translation_ok', False),
                'cited_indicators': ';'.join(verif.cited_indicators),
                'auto_status': verif.status,
                'auto_entailment': (verif.nli_scores or {}).get('entailment') if verif.nli_scores else None,
                'auto_detail': verif.detail,
            })
    pd.DataFrame(sentence_key_rows).to_csv(
        os.path.join(output_dir, 'internal_answer_key_sentences.csv'), index=False
    )

 
    with open(os.path.join(output_dir, 'calibration_practice.json'), 'w', encoding='utf-8') as f:
        json.dump({'items': [to_public_item(it, display_id_map[it['item_id']]) for it in calibration_items]},
                   f, ensure_ascii=False, indent=2)
 
    with open(os.path.join(output_dir, 'stimuli_master.json'), 'w', encoding='utf-8') as f:
        json.dump({'items': [to_public_item(it, display_id_map[it['item_id']]) for it in all_items]},
                   f, ensure_ascii=False, indent=2)
 
    n_control = len(control_items)
    n_total = len(all_items)
    report = [
        "MetaKT-Verba 평정 프로토콜 — 자극 표집 요약 리포트",
        "=" * 60,
        sampling_report,
        "",
        f"실험 문항(real) 수: {len(real_items)}",
        f"부정 대조 문항 수: {n_control} ({n_control/n_total*100:.1f}% of total, target 12~15%)",
        f"총 문항 수: {n_total}",
        f"Core set(전원 공통 평정) 크기: {len(core_set)}",
        f"평정자 수: {len(rater_assignments)}",
    ]
    for rid, items in rater_assignments.items():
        report.append(f"  - {rid}: {len(items)}문항 배정 (대조 {sum(1 for i in items if i.get('is_negative_control'))}개 포함)")
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'sampling_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("\n" + report_text)
    print(f"\n✅ Output complete: {output_dir}")
 
 
 
def main():
    parser = argparse.ArgumentParser(description='Run the analysis.')
    parser.add_argument('--input', type=str, default='data/interim/full_predictions2.csv', help='Row-level input data file (CSV or Parquet)')
    parser.add_argument('--output-dir', type=str, default='outputs/calibration_stimuli')
    parser.add_argument('--raters', nargs='+', default=['R1', 'R2', 'R3', 'R4', 'R5'])   
    parser.add_argument('--n-target', type=int, default=120,
                         help="Target number of real experimental items. 기본값 120인 이유: 5Categoryx3분위=15개 셀 x "
                              "--max-per-cell(기본 8) = 120이 프로토콜 A3 규칙(각 셀 최대 8문항) 하에서 "
                              "도달 가능한 이론적 상한이기 때문 — 130처럼 이 상한을 넘는 값을 넣으면 "
                              "매번 target 미달 경고만 뜨고 실제로는 120 근방에서 멈춘다. --max-per-cell을 "
                              "올리면(예: 9~10) 그만큼 n-target도 함께 올릴 수 있다.")
    parser.add_argument('--min-per-cell', type=int, default=6)
    parser.add_argument('--max-per-cell', type=int, default=8)
    parser.add_argument('--core-n', type=int, default=40)
    parser.add_argument('--control-frac', type=float, default=0.13, help='Negative-control fraction among all items')
    parser.add_argument('--candidate-pool-size', type=int, default=800,
                         help='Candidate-pool size for Stage 2/3 — 단, 이 값은 "충분히 많은(모집단이 '
                              f'{2000:,}개 초과인) Category"에만 적용되는 Category당 균등 배정 기준(pool_size // 5)이다. '
                              '모집단 자체가 그보다 작은 희소 Category(예: Struggle)는 이 값과 무관하게 항상 '
                              'full population 사용되므로, 최종 후보 풀 크기는 이 값보다 커질 수 있다(예: 800 지정 시 '
                              '실제로는 약 1,175개까지 늘어남 — 풍부한 4개 Category 160개씩 + 희소 Category full population). '
                              'The actual final size is reported in the Phase B candidate-pool log.')
    parser.add_argument('--engine', choices=['llama', 'template'], default='llama',
                         help="llama=실제 Stage2/3 파이프라인(GPU 필요, 백본은 --model로 지정), "
                              "template=Pipeline-only template generator (not for research outputs)")
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B',
                         help="Backbone used to generate rating stimuli. The default is Qwen3-8B because: "
                              "run_pipeline.py --language en 실측 cross-backbone 비교에서 "
                              "Qwen3-8B의 구조 위반률(인용 누락)이 0.70%%인 반면 "
                              "Llama-3.1-8B-Instruct는 72.14%%였음 — 구조 위반이 잦은 백본을 "
                              "쓰면 mean_entailment가 NaN인 문항이 많아져 같은 --candidate-pool-size로 "
                              "표집 target(15개 셀)를 못 채울 위험이 커진다. 다른 백본을 쓰려면 이 값을 "
                              "바꾸면 되고(예: 'meta-llama/Meta-Llama-3.1-8B-Instruct'), "
                              "trust_remote_code가 필요한 모델(EXAONE 등)은 --trust-remote-code를 함께 지정할 것.")
    parser.add_argument('--trust-remote-code', action='store_true',
                         help="--model이 transformers에 네이티브로 없는 커스텀 모델링 코드를 "
                              "요구하는 저장소(EXAONE 등)일 때 사용. 저장소가 제공하는 임의의 "
                              "파이썬 코드를 그대로 실행하게 되므로 신뢰하는 저장소에만 사용할 것.")
    parser.add_argument('--n-calibration', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-phase-b', type=str, default=None,
                         help='Phase B(Stage2 생성 + Stage3 NLI 검증) 결과를 캐시 파일(pickle)로 저장. '
                              '--load-phase-b와 함께 쓰면, --n-target/--control-frac/--core-n 등 표집 파라미터만 '
                              '바꿔가며 재실행할 때 매번 Llama/NLI를 다시 돌리지 않아도 됨 (GPU 시간 절약).')
    parser.add_argument('--load-phase-b', type=str, default=None,
                         help='이전에 --save-phase-b로 저장한 캐시를 불러와 Phase A/B(데이터 로드·Stage1·Stage2·Stage3)를 '
                              '건너뛰고 표집·대조문항·평정자배정만 다시 수행. 이 옵션을 쓰면 --input은 필요 없음.')
    parser.add_argument('--debug-nli-dump', type=str, default=None,
                         help='Diagnostic: save sentence-level (premise, hypothesis, score) values used by Stage 3 NLI to '
                              'a JSONL file. mean_entailment가 예상 밖으로 낮게/높게 나올 때 원인 확인용. '
                              '--engine llama일 때만 의미가 있음(template 엔진은 가짜 NLI 점수를 씀).')
    args = parser.parse_args()

        
 
    if not args.input and not args.load_phase_b:
        parser.error("--input is required unless --load-phase-b is provided")
 
    if args.load_phase_b:
        print(f"[Phase B] Loading Phase B cache: {args.load_phase_b}")
        with open(args.load_phase_b, 'rb') as f:
            items = pickle.load(f)
        print(f"[Phase B] Loaded {len(items)} items from cache (skipping Stage 1/2/3).")
    else:
        df = load_source_data(args.input)
        phase_a = run_phase_a(df)
        candidate_pool = select_candidate_pool(phase_a, args.candidate_pool_size, args.seed)
        items = run_phase_b(candidate_pool, df, args.engine, model_name=args.model,
                             trust_remote_code=args.trust_remote_code, debug_nli_dump=args.debug_nli_dump)
        if args.save_phase_b:
            with open(args.save_phase_b, 'wb') as f:
                pickle.dump(items, f)
            print(f"[Phase B] Saving Phase B results to cache: {args.save_phase_b} "
                  f"(다음부터는 --load-phase-b {args.save_phase_b} 로 표집만 빠르게 재실행 가능)")
 
    n_total_items = len(items)
    n_nan_entailment = sum(1 for it in items if np.isnan(it['mean_entailment']))
    struct_rates = [it['structural_violation_rate'] for it in items
                    if it.get('structural_violation_rate') is not None]
    diag_lines = [
        f"[Stage2 생성 품질 진단] 후보 {n_total_items}개 중 유효 문장이 하나도 만들어지지 않은 항목: "
        f"{n_nan_entailment}개 ({n_nan_entailment/n_total_items*100:.1f}%, 표집 대상에서 자동 제외됨)"
    ]
    if struct_rates:
        diag_lines.append(
            f"[Stage2 구조적 위반율] 평균 structural_violation_rate(인용 누락/근거에 없는 지표 인용): "
            f"{np.mean(struct_rates)*100:.1f}%"
        )
    if n_nan_entailment / n_total_items > 0.1:
        diag_lines.append("  Warning: more than 10% of items produced no valid sentences; check the Stage 2 prompt and decoding parameters.")
    print("\n".join(diag_lines))
 
    rng = random.Random(args.seed)
    shuffled_items = list(items)
    rng.shuffle(shuffled_items)
    calibration_items = shuffled_items[:args.n_calibration]
    pool_for_sampling = shuffled_items[args.n_calibration:]
 
    selected, sampling_report = stratified_sample(
        pool_for_sampling, args.n_target, args.min_per_cell, args.max_per_cell, args.seed
    )
    sampling_report = "\n".join(diag_lines) + "\n\n" + sampling_report
 
    n_controls = max(1, int(round(len(selected) * args.control_frac / (1 - args.control_frac))))
    controls = construct_negative_controls(selected, pool_for_sampling, n_controls, args.seed)
 
    rater_assignments, core_set = assign_raters(selected, controls, args.raters, args.core_n, args.seed)
 
    export_outputs(rater_assignments, core_set, selected, controls, calibration_items,
                    sampling_report, args.output_dir, seed=args.seed)
 
 
 
 
if __name__ == '__main__':
    main()