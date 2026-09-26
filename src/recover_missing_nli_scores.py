import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from nli_baseline_scores import load_true_citations
from prepare_calibration_stimuli import COLUMN_MAP
from core_set_io import infer_original_citations, load_stimuli_sentences
from stage1_grounding import build_symbolic_facts
GPU_NOT_NEEDED_NOTE = 'This script only requires the mDeBERTa NLI model; vLLM and judge LLMs are not required.'

def extract_display_row_idx(internal_item_id: str) -> int:
    if internal_item_id.startswith('NC-'):
        m = re.match('NC-IT-(\\d+)-IT-(\\d+)', internal_item_id)
        if not m:
            raise ValueError(f'Unrecognized negative-control internal_item_id format: {internal_item_id!r}')
        return int(m.group(1))
    if internal_item_id.startswith('IT-'):
        return int(internal_item_id[3:])
    raise ValueError(f"Unrecognized internal_item_id format; expected 'IT-' or 'NC-IT-': {internal_item_id!r}")

def load_id_map(answer_key_csv: str) -> Dict[str, str]:
    key_df = pd.read_csv(answer_key_csv, dtype=str)
    if 'display_id' not in key_df.columns or 'internal_item_id' not in key_df.columns:
        raise KeyError(f'{answer_key_csv} is missing display_id/internal_item_id columns. Available columns: {list(key_df.columns)}')
    return dict(zip(key_df['display_id'], key_df['internal_item_id']))

def build_original_facts_for_item(display_id: str, id_map: Dict[str, str], df: pd.DataFrame, lang: str):
    internal_item_id = id_map.get(display_id)
    if internal_item_id is None:
        raise KeyError(f'{display_id} has no internal_item_id in the answer key.')
    row_idx = extract_display_row_idx(internal_item_id)
    src_row = df.loc[row_idx]
    meta_vector = {mk: float(src_row[COLUMN_MAP[mk]]) for mk in COLUMN_MAP if mk.startswith('m_')}
    hint_count = float(src_row[COLUMN_MAP['hint_count']])
    sc = None
    if COLUMN_MAP.get('sc') and COLUMN_MAP['sc'] in src_row.index and pd.notna(src_row[COLUMN_MAP['sc']]):
        sc = float(src_row[COLUMN_MAP['sc']])
    return build_symbolic_facts(meta_vector, hint_count=hint_count, sc=sc, lang=lang)

def find_missing_sentences(sentences_by_display_id: Dict[str, List[str]], sentence_key_df: pd.DataFrame) -> Dict[str, List[int]]:
    present = set(zip(sentence_key_df['display_id'], sentence_key_df['sentence_id']))
    missing = {}
    for display_id, texts in sentences_by_display_id.items():
        idxs = [i for i in range(len(texts)) if (display_id, f'S{i + 1}') not in present]
        if idxs:
            missing[display_id] = idxs
    return missing

def recover_tag_residue_rows(sentence_key_df: pd.DataFrame, sentence_answer_key_csv: str, id_map: Dict[str, str], df: pd.DataFrame, verifier, lang: str, infer_unrecoverable: bool, sentences_by_display_id: Optional[Dict[str, List[str]]]=None) -> pd.DataFrame:
    from stage2_generation import CitedSentence
    true_citations = load_true_citations(sentence_answer_key_csv)
    problem_rows = sentence_key_df[sentence_key_df['auto_entailment'].isna()]
    rows = []
    for _, r in problem_rows.iterrows():
        display_id, sentence_id = (r['display_id'], r['sentence_id'])
        text = r['sentence_en'] if lang == 'en' else r['sentence_ko']
        citations = true_citations.get((display_id, sentence_id), [])
        try:
            facts = build_original_facts_for_item(display_id, id_map, df, lang)
        except (KeyError, ValueError) as e:
            print(f'[warning] {display_id}/{sentence_id} fact reconstruction failed; skipping: {e}')
            continue
        if citations:
            method = 'recovered_from_tag_residue'
        elif infer_unrecoverable:
            if sentences_by_display_id is None or display_id not in sentences_by_display_id:
                print(f'[warning] {display_id}: stimuli_master.json is required for inference; skipping.')
                continue
            all_texts = sentences_by_display_id[display_id]
            inferred = infer_original_citations(all_texts, facts, verifier, lang=lang)
            idx = int(sentence_id[1:]) - 1
            citations = [inferred[idx]] if inferred[idx] else []
            method = 'inferred_no_true_citation'
            if not citations:
                continue
        else:
            continue
        result = verifier.verify_narrative([CitedSentence(text=text, cited_indicators=citations)], facts, lang=lang)[0]
        entailment = result.nli_scores['entailment'] if result.nli_scores is not None else np.nan
        rows.append({'item_id': display_id, 'sentence_id': sentence_id, 'recovered_entailment': entailment, 'recovered_citation': ';'.join(citations), 'recovery_method': method})
    return pd.DataFrame(rows)

def recover_negative_control_rows(sentence_key_df: pd.DataFrame, sentences_by_display_id: Dict[str, List[str]], id_map: Dict[str, str], df: pd.DataFrame, verifier, lang: str) -> pd.DataFrame:
    from stage2_generation import CitedSentence
    missing = find_missing_sentences(sentences_by_display_id, sentence_key_df)
    rows = []
    for display_id, idxs in missing.items():
        try:
            facts = build_original_facts_for_item(display_id, id_map, df, lang)
        except (KeyError, ValueError) as e:
            print(f'[warning] {display_id} fact reconstruction failed; skipping ({len(idxs)} sentences): {e}')
            continue
        all_texts = sentences_by_display_id[display_id]
        inferred = infer_original_citations(all_texts, facts, verifier, lang=lang)
        for idx in idxs:
            citation = inferred[idx]
            if not citation:
                continue
            sentence_id = f'S{idx + 1}'
            result = verifier.verify_narrative([CitedSentence(text=all_texts[idx], cited_indicators=[citation])], facts, lang=lang)[0]
            entailment = result.nli_scores['entailment'] if result.nli_scores is not None else np.nan
            rows.append({'item_id': display_id, 'sentence_id': sentence_id, 'recovered_entailment': entailment, 'recovered_citation': citation, 'recovery_method': 'inferred_negative_control'})
    return pd.DataFrame(rows)

def run_recovery(source_csv: str, answer_key_csv: str, sentence_answer_key_csv: str, stimuli_master_path: str, output_csv: str, lang: str='en', infer_unrecoverable: bool=False) -> pd.DataFrame:
    from prepare_calibration_stimuli import load_source_data
    from stage3_verification import NLIVerifier
    print(GPU_NOT_NEEDED_NOTE)
    print('Loading source data...')
    df = load_source_data(source_csv)
    id_map = load_id_map(answer_key_csv)
    sentence_key_df = pd.read_csv(sentence_answer_key_csv)
    sentences_by_display_id = load_stimuli_sentences(stimuli_master_path, lang=lang)
    verifier = NLIVerifier()
    print('\n=== Recovering rows with missing auto_entailment ===')
    recovered_ab = recover_tag_residue_rows(sentence_key_df, sentence_answer_key_csv, id_map, df, verifier, lang, infer_unrecoverable=infer_unrecoverable, sentences_by_display_id=sentences_by_display_id)
    print(f"  {len(recovered_ab)} recovered ({(recovered_ab['recovery_method'].value_counts().to_dict() if len(recovered_ab) else {})})")
    print('\n=== Recovering sentences missing from the sentence answer key ===')
    recovered_c = recover_negative_control_rows(sentence_key_df, sentences_by_display_id, id_map, df, verifier, lang)
    print(f'  {len(recovered_c)} recovered')
    out_df = pd.concat([recovered_ab, recovered_c], ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f'\nCompleted: {len(out_df)} sentences recovered. Saved to: {output_csv}')
    print('Merge this file with extract_baseline_from_sentence_answer_key() output before the three-way comparison. Use recovery_method to select recovery subsets.')
    return out_df

def main():
    parser = argparse.ArgumentParser(description="Recover missing Stage-3 NLI scores from original evidence.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-csv', type=str, default='data/interim/full_predictions2.csv')
    parser.add_argument('--answer-key', type=str, default='data/stimuli/internal_answer_key.csv')
    parser.add_argument('--sentence-answer-key', type=str, default='data/stimuli/internal_answer_key_sentences.csv')
    parser.add_argument('--stimuli-master', type=str, default='data/stimuli/stimuli_master.json')
    parser.add_argument('--output', type=str, default='data/results/rq2/recovered_nli_scores.csv')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'ko'])
    parser.add_argument('--infer-unrecoverable', action='store_true', help='Infer a best-matching citation for otherwise unrecoverable sentences. Disabled by default.')
    args = parser.parse_args()
    required = [args.source_csv, args.answer_key, args.sentence_answer_key, args.stimuli_master]
    if not all(required):
        parser.error('--source-csv, --answer-key, --sentence-answer-key, and --stimuli-master are required.')
    run_recovery(source_csv=args.source_csv, answer_key_csv=args.answer_key, sentence_answer_key_csv=args.sentence_answer_key, stimuli_master_path=args.stimuli_master, output_csv=args.output, lang=args.lang, infer_unrecoverable=args.infer_unrecoverable)
if __name__ == '__main__':
    main()
