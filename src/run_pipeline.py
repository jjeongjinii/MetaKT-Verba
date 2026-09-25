import os


import argparse
import hashlib
import json
import re
import traceback

import pandas as pd

from stage1_grounding import build_symbolic_facts, build_raw_numeric_facts, HIGH_THRESHOLD
from stage2_generation import NarrativeGenerator, generate_template_only_narrative
from stage3_verification import NLIVerifier, summarize_results

META_COLS = [
    'm_overconfidence', 'm_underconfidence', 'm_strategic_help',
    'm_cognitive_avoidance', 'm_productive_struggle', 'm_unproductive_frustration',
    'm_lucky_guess', 'm_slipping', 'm_boredom_offtask'
]

INFORMATIVE_CUTOFF = HIGH_THRESHOLD * 0.3


CONDITIONS = {
    'naive': dict(
        group='baseline', label='Naive Prompting',
        use_stage1=False, require_citation=False, use_llm=True,
        use_stage3=True, uncited_eval=True,
    ),
    'template': dict(
        group='baseline', label='Template-only',
        use_stage1=True, require_citation=True, use_llm=False,
        use_stage3=True, uncited_eval=False,
    ),
    'full': dict(
        group='baseline', label='MetaKT-Verba (Full)',
        use_stage1=True, require_citation=True, use_llm=True,
        use_stage3=True, uncited_eval=False,
    ),
    'ablate_stage1': dict(
        group='ablation', label='w/o Symbolic Grounding (Stage 1)',
        use_stage1=False, require_citation=True, use_llm=True,
        use_stage3=True, uncited_eval=True,
    ),
    'ablate_citation': dict(
        group='ablation', label='w/o Citation-Forcing Prompt',
        use_stage1=True, require_citation=False, use_llm=True,
        use_stage3=True, uncited_eval=True,
    ),
    'ablate_stage3': dict(
        group='ablation', label='w/o Auto Verification (Stage 3)',
        use_stage1=True, require_citation=True, use_llm=True,
        use_stage3=False, uncited_eval=False,
    ),
}
DEFAULT_CONDITIONS = ['naive', 'template', 'full', 'ablate_stage1', 'ablate_citation', 'ablate_stage3']


def set_global_determinism(seed: int = 42) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except AttributeError:
        print("  (Note: this torch version does not expose the SDPA backend control API, so attention determinism is not guaranteed; "
              "consider upgrading torch.)")

    print(f"✅ Global determinism configured (seed={seed}, cudnn.deterministic=True, "
          f"attention backend=math-only)")

_SEED_KEY_ALIASES = {'ablate_stage3': 'full'}


def _slugify_model_name(model_name: str) -> str:
    tag = model_name.rstrip('/').split('/')[-1]
    return re.sub(r'[^A-Za-z0-9._-]+', '_', tag)


def _stable_hash(*parts) -> int:
    s = '|'.join(str(p) for p in parts)
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)


def compute_structural_violation_rate(sentences: list, facts: list) -> float:
    no_citable_evidence = not any(f.status in ('HIGH', 'AMBIGUOUS') for f in facts)
    if not sentences:
        return None
    if no_citable_evidence:
        return 0.0
    n_fail_nocit = sum(1 for s in sentences if not s.cited_indicators)
    return n_fail_nocit / len(sentences)


def run_one_condition(row, condition_key: str, generator: NarrativeGenerator,
                       verifier: NLIVerifier, language: str = 'ko',
                       bilingual: bool = False, max_new_tokens: int = 200,
                       seed: int = None) -> dict:
    cond = CONDITIONS[condition_key]
    meta_vector = {c: row[c] for c in META_COLS}
    hint_count = row['hintCount']
    context = {'skill': row['skill'], 'step': row['step_idx']}
    seed_key = _SEED_KEY_ALIASES.get(condition_key, condition_key)
    gen_seed = (
        (seed + _stable_hash(int(row['ITEST_id']), int(row['step_idx']), seed_key)) % (2**31)
        if seed is not None else None
    )

    gt_facts = build_symbolic_facts(meta_vector, hint_count, sc=None, lang=language)

    base = {
        'condition': condition_key, 'group': cond['group'], 'label': cond['label'],
        'ITEST_id': int(row['ITEST_id']), 'step_idx': int(row['step_idx']),
        'skill': row['skill'], 'language': language,
        'citation_required': cond['require_citation'],
        'stage1_used': cond['use_stage1'], 'stage3_used': cond['use_stage3'],
    }

    facts_shown = None
    if condition_key == 'naive':
        gen_result = generator.generate_naive(
            meta_vector, hint_count, context=context, language=language,
            max_new_tokens=max_new_tokens, seed=gen_seed,
        )
    elif condition_key == 'template':
        facts_shown = gt_facts
        gen_result = generate_template_only_narrative(gt_facts, language=language)
    else:
        facts_shown = gt_facts if cond['use_stage1'] else build_raw_numeric_facts(meta_vector, lang=language)
        gen_result = generator.generate(
            facts_shown, context=context, language=language, max_new_tokens=max_new_tokens,
            require_citation=cond['require_citation'], seed=gen_seed,
        )

    if gen_result.get('truncated_in_thinking'):
        return {**base, 'truncated_in_thinking': True, 'summary': {'n_sentences': 0}}

    sentences = gen_result['sentences']

    if not cond['use_stage3']:
        verif_results = []
        summary = {
            'n_sentences': len(sentences),
            'faithfulness_rate': None, 'hallucination_rate': None, 'hedge_violation_rate': None,
            'structural_violation_rate': (
                compute_structural_violation_rate(sentences, gt_facts)
                if cond['require_citation'] else None
            ),
            'conditional_faithfulness_rate': None, 'conditional_hallucination_rate': None,
            'n_content_eligible': None, 'breakdown': {},
        }
    else:
        verify_fn = verifier.verify_narrative_uncited if cond['uncited_eval'] else verifier.verify_narrative
        verif_results = verify_fn(sentences, gt_facts, lang=language)
        summary = summarize_results(verif_results)
        if cond['uncited_eval']:
            summary['structural_violation_rate'] = (
                gen_result['citation_check']['structural_violation_rate']
                if cond['require_citation'] else None
            )

    bilingual_sentences = None
    if bilingual and language == 'en' and sentences:
        translated = generator.translate_to_korean(sentences)
        if verif_results and len(verif_results) == len(translated):
            for bs, vr in zip(translated, verif_results):
                bs['verification_status'] = vr.status
        else:
            for bs in translated:
                bs['verification_status'] = None if cond['use_stage3'] else 'UNVERIFIED'
        bilingual_sentences = translated

    return {
        **base,
        'truncated_in_thinking': False,
        'facts_shown_to_generator': (
            [f"[{f.indicator}] {f.status}: {f.note}" for f in facts_shown]
            if facts_shown is not None else None
        ),
        'ground_truth_facts': [f"[{f.indicator}] {f.status}: {f.note}" for f in gt_facts],
        'prompt': (
            "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in gen_result['messages'])
            if gen_result.get('messages') else None
        ),
        'raw_narrative': gen_result['raw_text'],
        'sentences': [{'text': s.text, 'cited': s.cited_indicators} for s in sentences],
        'verification': [
            {'text': r.sentence_text, 'status': r.status, 'nli': r.nli_scores, 'detail': r.detail}
            for r in verif_results
        ],
        'bilingual_sentences': bilingual_sentences,
        'summary': summary,
    }


def print_comparison_tables(summary_df: pd.DataFrame) -> None:
    if len(summary_df) == 0:
        print("(No results to summarize)")
        return

    metrics = ['faithfulness_rate', 'hallucination_rate', 'hedge_violation_rate',
               'structural_violation_rate', 'conditional_faithfulness_rate',
               'conditional_hallucination_rate']
    present_metrics = [m for m in metrics if m in summary_df.columns]
    for m in present_metrics:
        summary_df[m] = pd.to_numeric(summary_df[m], errors='coerce')
    agg = summary_df.groupby('label')[present_metrics].mean(numeric_only=True)
    counts = summary_df.groupby('label').size()

    def _print_row(label: str):
        if label not in agg.index:
            print(f"  [{label}]  (No results — not run or all runs failed)")
            return
        row = agg.loc[label]
        print(f"\n  [{label}]  (n={counts.loc[label]})")
        for m in present_metrics:
            v = row.get(m) if hasattr(row, 'get') else (row[m] if m in row.index else None)
            print(f"    {m:32s}: {'N/A' if v is None or pd.isna(v) else f'{v:.2%}'}")

    print("\n" + "=" * 70)
    print("Table 4.5 — Comparison (Naive Prompting vs Template-only vs MetaKT-Verba Full)")
    print("=" * 70)
    for key in ('naive', 'template', 'full'):
        _print_row(CONDITIONS[key]['label'])

    print("\n" + "=" * 70)
    print("Table 4.4 — Ablation study (single-factor removals from MetaKT-Verba Full)")
    print("=" * 70)
    for key in ('full', 'ablate_stage1', 'ablate_citation', 'ablate_stage3'):
        _print_row(CONDITIONS[key]['label'])

    print("\n" + "-" * 70)
    print("Warning: naive / ablate_stage1 / ablate_citation lack citation tags or use unreliable "
          "because they are missing citation tags or use unreliable citations, they are evaluated with "
          "verify_narrative_uncited() using its best-match procedure. Their faithfulness_rate may therefore be "
          "somewhat optimistic; document this limitation in the paper.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Input CSV path relative to the repository root')
    parser.add_argument('--student-id', type=int, default=None)
    parser.add_argument('--step', type=int, default=None)
    parser.add_argument('--sample-size', type=int, default=400,
                         help='Random sample size used when --student-id is not specified')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-prefix', default='data/results/rq1/pipeline_results')
    parser.add_argument('--language', choices=['ko', 'en'], default='en',
                         help='Language for narrative generation and verification. Use --language en to rerun the same sample for cross-lingual robustness checks.')
    parser.add_argument('--model', default='mistralai/Ministral-8B-Instruct-2410',
                         help="Model backbone. Examples: 'Qwen/Qwen3-8B', 'meta-llama/Llama-3.1-8B-Instruct', or "
                              "'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'. Change this value for cross-backbone robustness checks.")
    parser.add_argument('--bilingual', action='store_true',
                         help="Translate English (--language en) narratives into Korean sentence by sentence and print them alongside the original. "
                              "This option requires --language en.")
    parser.add_argument('--enable-thinking', action='store_true',
                         help="Enable <think> reasoning for models such as Qwen3. Disabled by default because the pipeline targets short 2-4 sentence narratives. "
                              "Reasoning can add latency and token usage and may consume the generation budget before a narrative is produced. "
                              "If enabled, increase --max-new-tokens substantially.")
    parser.add_argument('--max-new-tokens', type=int, default=200,
                         help='Maximum new tokens for narrative/translation generation (default: 200). Increase this substantially when using --enable-thinking.')
    parser.add_argument('--trust-remote-code', action='store_true',
                         help="Allow custom model code for models that require trust_remote_code. Enabling this executes Python code from the selected model repository; use only with repositories you trust.")
    parser.add_argument('--conditions', default='all',
                         help="Comma-separated condition keys. Default 'all' runs all six conditions in order: "
                              f"({', '.join(DEFAULT_CONDITIONS)}). Example: --conditions full,naive,template")
    args = parser.parse_args()

    if args.conditions == 'all':
        conditions_to_run = DEFAULT_CONDITIONS
    else:
        conditions_to_run = [c.strip() for c in args.conditions.split(',') if c.strip()]
        unknown = [c for c in conditions_to_run if c not in CONDITIONS]
        if unknown:
            raise ValueError(f"Unknown condition(s): {unknown} (available: {list(CONDITIONS.keys())})")

    if args.bilingual and args.language != 'en':
        print("⚠️  --bilingual is only meaningful when used with --language en. "
              "Because --language is not 'en', the translation step will be skipped.")
        args.bilingual = False

    print(f"Loaded CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Total rows: {len(df):,}")
    print(f"Conditions to run ({len(conditions_to_run)}): {conditions_to_run}")

    if args.student_id is None and args.sample_size is None:
        print("No run-selection arguments provided; using the default case-study reproduction mode (student_id=10)")
        args.student_id = 10

    if args.student_id is not None:
        target = df[df['ITEST_id'] == args.student_id]
        if args.step is not None:
            target = target[target['step_idx'] == args.step]
        target = target.sort_values('step_idx')
        print(f"Selected rows: {len(target)} (student_id={args.student_id}"
              + (f", step={args.step}" if args.step is not None else "") + ")")
    else:
        informative = df[(df[META_COLS].abs() > INFORMATIVE_CUTOFF).any(axis=1)]
        print(f"Rows with informative signals: {len(informative):,} / {len(df):,} "
              f"(criterion: any of the 9 indicators > {INFORMATIVE_CUTOFF:.3f})")
        n = min(args.sample_size, len(informative))
        target = informative.sample(n=n, random_state=args.seed)
        print(f"Random sample of {n} selected (seed={args.seed})")

    if len(target) == 0:
        raise ValueError("No rows selected. Check the student-id/step filters.")

    print(f"Total runs: {len(target)} rows x {len(conditions_to_run)} conditions = "
          f"{len(target) * len(conditions_to_run)} runs")

    model_tag = _slugify_model_name(args.model)
    if model_tag not in args.output_prefix:
        args.output_prefix = f'{args.output_prefix}_{model_tag}'
    if not args.output_prefix.endswith(f'_{args.language}'):
        args.output_prefix = f'{args.output_prefix}_{args.language}'

    print("\nLoading NLI verifier...")
    set_global_determinism(args.seed)
    verifier = NLIVerifier()
    print(f"Loading narrative generator (LLM)... (model={args.model}, language={args.language}, "
          f"bilingual={args.bilingual}, enable_thinking={args.enable_thinking}, "
          f"trust_remote_code={args.trust_remote_code})")
    generator = NarrativeGenerator(
        model_name=args.model,
        chat_template_kwargs={'enable_thinking': args.enable_thinking},
        trust_remote_code=args.trust_remote_code,
    )

    all_results = []
    total = len(target) * len(conditions_to_run)
    counter = 0
    for idx, (_, row) in enumerate(target.iterrows()):
        for condition_key in conditions_to_run:
            counter += 1
            print(f"\n[{counter}/{total}][{condition_key}] ITEST_id={row['ITEST_id']}, "
                  f"step={row['step_idx']}, skill={row['skill']}")
            try:
                result = run_one_condition(
                    row, condition_key, generator, verifier, language=args.language,
                    bilingual=args.bilingual, max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
                all_results.append(result)
                if result.get('truncated_in_thinking'):
                    print(f"  ⚠️ The reasoning block (<think>) reached max_new_tokens={args.max_new_tokens}  before the token limit; "
                          f"so no narrative was generated.")
                    continue
                preview = result['raw_narrative'][:80].replace('\n', ' ')
                print(f"  Generated: {preview}...")
                print(f"  Verification summary: {result['summary']}")
                if result['bilingual_sentences']:
                    for bs in result['bilingual_sentences']:
                        ko_preview = bs['ko'] if bs['translation_ok'] else f"(translation failed, raw: {bs['raw_ko'][:40]}...)"
                        print(f"    [{bs['verification_status']}] EN: {bs['en']}")
                        print(f"    {' ' * (len(str(bs['verification_status'])) + 2)}KO: {ko_preview}")
            except Exception as e:
                print(f"  ⚠️ Failed: {e!r}")
                print("  --- Detailed traceback ---")
                traceback.print_exc()
                print("  ----------------------")
                all_results.append({
                    'condition': condition_key, 'group': CONDITIONS[condition_key]['group'],
                    'label': CONDITIONS[condition_key]['label'],
                    'ITEST_id': int(row['ITEST_id']), 'step_idx': int(row['step_idx']),
                    'skill': row.get('skill', None), 'error': str(e), 'traceback': traceback.format_exc()
                })

    with open(f'{args.output_prefix}.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved detailed results (all conditions): {args.output_prefix}.json")

    summary_rows = []
    for r in all_results:
        if 'summary' in r:
            row_summary = {
                'condition': r.get('condition'), 'group': r.get('group'), 'label': r.get('label'),
                'ITEST_id': r.get('ITEST_id'), 'step_idx': r.get('step_idx'), 'skill': r.get('skill'),
                'citation_required': r.get('citation_required'),
                'stage1_used': r.get('stage1_used'), 'stage3_used': r.get('stage3_used'),
                'truncated_in_thinking': r.get('truncated_in_thinking', False),
            }
            row_summary.update({k: v for k, v in r['summary'].items() if k != 'breakdown'})
            summary_rows.append(row_summary)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = f'{args.output_prefix}_all_conditions_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved condition summary CSV: {summary_csv_path}")
    print("   Use pandas groupby(['condition']) to build the comparison and ablation tables.")

    n_errored = sum(1 for r in all_results if 'summary' not in r)
    if n_errored:
        print(f"\nWarning: {n_errored}/{len(all_results)} row-condition pairs failed and were excluded from summary statistics. "
              f"Check the failure logs or the 'error' entries in {args.output_prefix}.json. "
              f"This may explain a final sample smaller than --sample-size.")

    n_truncated = sum(1 for r in all_results if r.get('truncated_in_thinking'))
    if n_truncated:
        print(f"\nWarning: {n_truncated}/{len(all_results)} items did not produce a narrative because the <think> block was truncated; "
              f"they were excluded from the summary statistics.")

    print_comparison_tables(summary_df)


if __name__ == "__main__":
    main()
