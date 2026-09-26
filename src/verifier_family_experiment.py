import argparse
import ast
import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from compare_auroc import bootstrap_auroc_ci, bootstrap_paired_auroc_diff_test
from few_shot_judge_experiment import TAG_DEFINITIONS, build_case_table, call_judge_llm, stratified_few_shot_sample
from prepare_calibration_stimuli import load_source_data
from recover_missing_nli_scores import build_original_facts_for_item, load_id_map
from stage3_verification import NLIVerifier, _fact_to_premise_text
BASE_NLI = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
LARGE_NLI = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
GEMMA27B = 'google/gemma-3-27b-it'
GROUND_TRUTH_DEFS = {'strict': {'supported'}, 'lenient': {'supported', 'partial'}}
BASELINE_NAME = 'mDeBERTa-v3-base-mnli-xnli'
PUBLIC_CASE_COLUMNS = [
    'item_id',
    'sentence_id',
    'sentence_text',
    'majority_tag',
    'n_raters',
    'cited_indicators',
    'indicator',
    'is_core',
    'is_negative_control',
    'fact_premise_list',
    'fact_premise',
    'record_premise',
]
PUBLIC_SCORE_COLUMNS = [
    'item_id',
    'sentence_id',
    'majority_tag',
    'cited_indicators',
    'indicator',
    'is_core',
    'is_negative_control',
    'score',
    'verifier',
    'premise',
]
_TAG_RE = __import__('re').compile('^[A-Za-z_][A-Za-z0-9_]*$')
_STATUS_WORDS = {'nan', 'none', 'indicator', 'indicators', 'high', 'ambiguous', 'low', 'weak'}

def _split_citation_field(x) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s or s.lower() in {'nan', 'none', '[]'}:
        return []
    import re
    parts = re.split('[,;|\\[\\]]', s)
    return [p.strip().strip('\'"').strip() for p in parts if p.strip().strip('\'"').strip()]

def _parse_citations(x) -> List[str]:
    return [t for t in _split_citation_field(x) if _TAG_RE.match(t) and t.lower() not in _STATUS_WORDS]

def _invalid_citations(x) -> List[str]:
    return [t for t in _split_citation_field(x) if not _TAG_RE.match(t) and t.lower() not in _STATUS_WORDS]

def _parse_steps(steps):
    if steps is None or (isinstance(steps, float) and np.isnan(steps)):
        return []
    if isinstance(steps, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                return list(parser(steps))
            except Exception:
                pass
        return []
    return list(steps)

def _fmt_num(x, nd=2):
    try:
        return f'{float(x):.{nd}f}'
    except (TypeError, ValueError):
        return str(x)

def _is_true(x) -> bool:
    return str(x).strip().lower() in {'1', '1.0', 'true', 'correct', 'yes'}

def _record_premise(row: pd.Series):
    steps = _parse_steps(row.get('steps'))
    if not steps:
        return None
    acc = row.get('skill_cum_accuracy')
    try:
        a = float(acc)
        acc_txt = f'{a * 100:.0f}%' if a <= 1 else f'{a:.0f}%'
    except (TypeError, ValueError):
        acc_txt = 'unknown'
    lines = [f"The student is practicing the skill '{row.get('skill')}'.", f"Before these attempts, the student's cumulative accuracy on this skill was {acc_txt}."]
    for i, st in enumerate(steps, 1):
        corr = 'correctly' if _is_true(st.get('correct')) else 'incorrectly'
        h = st.get('hint_count')
        try:
            h_int = int(float(h))
            hint_txt = 'used no hints' if h_int == 0 else f"used {h_int} hint{('s' if h_int != 1 else '')}"
        except (TypeError, ValueError):
            hint_txt = f'used {h} hints'
        rt = _fmt_num(st.get('response_time_ratio'))
        lines.append(f'On attempt {i}, the student answered {corr}, {hint_txt}, and took {rt} times their own average response time.')
    return ' '.join(lines)

def _load_control_ids(answer_key) -> Dict[str, bool]:
    key = pd.read_csv(answer_key, dtype=str)
    id_col = next((c for c in ['display_id', 'item_id'] if c in key.columns), None)
    if id_col is None or 'is_negative_control' not in key.columns:
        raise KeyError(f'{answer_key}: need an id column (display_id/item_id) and is_negative_control; columns={list(key.columns)}')
    flag = key['is_negative_control'].astype(str).str.lower().eq('true')
    return dict(zip(key[id_col], flag))

def build_cases(responses_dir, stimuli_master, answer_key, sentence_answer_key, source_csv) -> pd.DataFrame:
    cases = build_case_table(responses_dir, stimuli_master, id_map_path=answer_key, min_raters=1)
    sk = pd.read_csv(sentence_answer_key, dtype=str)
    cite_col = next((c for c in ['cited_indicators', 'cited_indicator', 'indicator'] if c in sk.columns), None)
    if cite_col is None:
        raise KeyError(f'{sentence_answer_key}: cited indicator column not found; columns={list(sk.columns)}')
    sk = sk.rename(columns={'display_id': 'item_id'})
    cases = cases.merge(sk[['item_id', 'sentence_id', cite_col]], on=['item_id', 'sentence_id'], how='left')
    cases['cited_indicators'] = cases[cite_col].apply(_parse_citations)
    cases['invalid_citations'] = cases[cite_col].apply(_invalid_citations)
    src = load_source_data(source_csv)
    id_map = load_id_map(answer_key)
    fact_cache: Dict[str, Dict[str, str]] = {}
    fact_lists, missing_facts, primary_inds = ([], [], [])
    for r in cases.itertuples(index=False):
        iid = r.item_id
        if iid not in fact_cache:
            try:
                facts = build_original_facts_for_item(iid, id_map, src, lang='en')
                fact_cache[iid] = {f.indicator: _fact_to_premise_text(f, lang='en') for f in facts}
            except Exception as e:
                print(f'[warning] {iid}: cannot reconstruct Stage-1 facts: {e}')
                fact_cache[iid] = {}
        inds = getattr(r, 'cited_indicators') or []
        texts = [fact_cache[iid][ind] for ind in inds if ind in fact_cache[iid]]
        missing = [ind for ind in inds if ind not in fact_cache[iid]]
        fact_lists.append(texts if texts and (not missing) else None)
        missing_facts.append(missing)
        primary_inds.append(inds[0] if len(inds) == 1 else None)
    cases['fact_premise_list'] = fact_lists
    cases['fact_premise'] = [None if l is None else '\n'.join(l) for l in fact_lists]
    cases['missing_facts'] = missing_facts
    cases['record_premise'] = cases.apply(_record_premise, axis=1)
    cases['indicator'] = primary_inds
    cases['is_core'] = cases['n_raters'] >= 5
    cases['is_negative_control'] = cases['item_id'].map(_load_control_ids(answer_key)).fillna(False)
    return cases

def _validate_cases(cases: pd.DataFrame, allow_missing: bool=False) -> None:
    n = len(cases)
    n_cited = int(cases['cited_indicators'].map(bool).sum())
    n_fact = int(cases['fact_premise'].notna().sum())
    n_record = int(cases['record_premise'].notna().sum())
    print(f'[cases] total={n}, cited={n_cited}, fact_premise={n_fact}, record_premise={n_record}, controls={int(cases.is_negative_control.sum())}')
    tag_counts = pd.Series([t for l in cases['cited_indicators'] for t in l]).value_counts()
    print('[cases] parsed citation tags:\n' + tag_counts.to_string())
    inv = cases[cases['invalid_citations'].map(bool)]
    if len(inv):
        print(f'[cases] {len(inv)} sentences carry a free-text (invalid) citation tag ({int((~inv.is_negative_control).sum())} non-control); {int(inv.cited_indicators.map(bool).sum())} of them also carry a valid tag. Examples:')
        for t in inv['invalid_citations'].head(5):
            print('   ', str(t[0])[:100])
    miss = pd.Series([t for l in cases['missing_facts'] for t in l]).value_counts()
    if len(miss):
        msg = '[cases] cited indicators with NO reconstructed fact:\n' + miss.to_string()
        if not allow_missing:
            raise RuntimeError(msg + '\n(use --allow-missing-premises to continue anyway)')
        print(msg)
    main = cases[~cases.is_negative_control]
    print('[cases] single-citation counts (non-control; compare with Table 5: slipping 45, frustration 35, struggle 25, overconfidence 56, avoidance 77, triad 39):\n' + main['indicator'].value_counts(dropna=False).to_string())
    if n_cited == 0:
        raise RuntimeError('Parsed zero cited indicators. Check internal_answer_key_sentences.csv citation format.')
    if n_fact == 0:
        raise RuntimeError('Constructed zero fact premises. This would invalidate the verifier × premise comparison.')
    if n_fact < 0.5 * n_cited:
        raise RuntimeError(f'Only {n_fact}/{n_cited} cited sentences received a fact premise; check item IDs and indicator names before scoring.')
    if n_record < n:
        bad = cases[cases.record_premise.isna()][['item_id', 'sentence_id']]
        raise RuntimeError(f"{len(bad)} cases have no parsable attempts in 'steps':\n{bad.head(10)}")

class PairScorer:
    name = 'base'

    def score_many(self, premises, claims):
        raise NotImplementedError

    def close(self):
        pass

class NliPairScorer(PairScorer):

    def __init__(self, model_name, name):
        self.name = name
        self.v = NLIVerifier(model_name=model_name)
        self.v._lazy_load()
        id2label = {int(k): str(v).lower() for k, v in self.v._model.config.id2label.items()}
        ent = [i for i, lab in id2label.items() if lab.startswith('entail')]
        if len(ent) != 1:
            raise RuntimeError(f'{model_name}: cannot find entailment label in {id2label}')
        self.ent_idx = ent[0]
        print(f'[{name}] id2label={id2label} -> entailment index {self.ent_idx}')

    def _entail(self, premise, claim):
        import torch
        inputs = self.v._tokenizer(premise, claim, truncation='only_first', return_tensors='pt').to(self.v._device)
        with torch.no_grad():
            logits = self.v._model(**inputs).logits
        return float(torch.softmax(logits[0], dim=-1)[self.ent_idx])

    def score_many(self, premises, claims):
        out = []
        for p, c in zip(premises, claims):
            ps = p if isinstance(p, list) else [p]
            out.append(min((self._entail(x, c) for x in ps)))
        return out

    def close(self):
        self.v = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

class MiniCheckScorer(PairScorer):

    def __init__(self, model_name='Bespoke-MiniCheck-7B', gpu_memory_utilization=0.25, max_model_len=4096, tensor_parallel_size=1, cache_dir='./ckpts'):
        self.name = 'Bespoke-MiniCheck-7B'
        if not 0.0 < gpu_memory_utilization <= 1.0:
            raise ValueError('gpu_memory_utilization must be in (0, 1].')
        try:
            import nltk
            for resource, package in [('tokenizers/punkt', 'punkt'), ('tokenizers/punkt_tab/english', 'punkt_tab')]:
                try:
                    nltk.data.find(resource)
                except LookupError:
                    print(f'[MiniCheck] downloading NLTK resource: {package}')
                    ok = nltk.download(package, quiet=True)
                    if not ok:
                        raise RuntimeError(f'NLTK resource {package!r} is required by MiniCheck and could not be downloaded. Run: python -m nltk.downloader {package}')
            import os
            os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
            from minicheck.minicheck import MiniCheck
            import vllm
        except ImportError as e:
            raise ImportError('Install MiniCheck with: pip install "minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main"') from e
        original_llm = vllm.LLM

        def memory_limited_llm(*args, **kwargs):
            kwargs.setdefault('gpu_memory_utilization', gpu_memory_utilization)
            kwargs.setdefault('enforce_eager', True)
            return original_llm(*args, **kwargs)
        print(f'[MiniCheck] vLLM gpu_memory_utilization={gpu_memory_utilization:.3f}, max_model_len={max_model_len}, tensor_parallel_size={tensor_parallel_size}, enforce_eager=True, flashinfer_sampler=False')
        vllm.LLM = memory_limited_llm
        try:
            self.scorer = MiniCheck(model_name=model_name, enable_prefix_caching=False, cache_dir=cache_dir, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)
        finally:
            vllm.LLM = original_llm

    def score_many(self, premises, claims):
        flat_docs, flat_claims, owner = ([], [], [])
        for i, (p, c) in enumerate(zip(premises, claims)):
            for x in p if isinstance(p, list) else [p]:
                flat_docs.append(x)
                flat_claims.append(c)
                owner.append(i)
        _, raw_prob, _, _ = self.scorer.score(docs=flat_docs, claims=flat_claims)
        out = [np.inf] * len(premises)
        for i, pr in zip(owner, raw_prob):
            out[i] = min(out[i], float(pr))
        return out

    def close(self):
        self.scorer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

def score_pair_model(cases: pd.DataFrame, scorer: PairScorer, premise_mode: str) -> pd.DataFrame:
    col = 'fact_premise_list' if premise_mode == 'fact' else 'record_premise'
    d = cases[cases[col].notna()].copy()
    print(f'[{scorer.name}] scoring {premise_mode}: n={len(d)}')
    d['score'] = scorer.score_many(d[col].tolist(), d['sentence_text'].tolist())
    d['verifier'] = scorer.name
    d['premise'] = premise_mode
    return d

def _llm_fact_prompt(target: pd.Series) -> str:
    return '\n'.join(['You check whether a sentence written for a teacher faithfully restates a symbolic fact produced by a learner model.', 'Judge ONLY whether the fact(s) below entail the sentence. Do not use outside knowledge about the student.', '', f"Fact(s):\n{target['fact_premise']}", f"Sentence: {target['sentence_text']}", 'Return JSON only: {"label":"supported|partial|unsupported|unknown", "confidence_supported":0-100, "reasoning":"one or two sentences"}'])

def _llm_prompt(few: pd.DataFrame, target: pd.Series, premise_mode: str) -> str:
    title = 'Stage-1 symbolic fact' if premise_mode == 'fact' else 'teacher-visible behavior record'
    parts = [f'Judge whether the claim is supported by the {title}. Use the same four expert categories:', TAG_DEFINITIONS, '', 'Examples below are from expert ratings. Learn the decision boundary from their labels and rationales.', '']
    for i, (_, r) in enumerate(few.iterrows(), 1):
        parts += [f'### Example {i}', f"Premise:\n{r[premise_mode + '_premise']}", f"Claim: {r['sentence_text']}", f"Expert label: {r['majority_tag']}"]
        if r.get('rationale'):
            parts.append(f"Expert rationale: {r['rationale']}")
        parts.append('')
    parts += ['### Target', f"Premise:\n{target[premise_mode + '_premise']}", f"Claim: {target['sentence_text']}", 'Return JSON only: {"label":"supported|partial|unsupported|unknown", "confidence_supported":0-100, "reasoning":"one or two sentences"}']
    return '\n'.join(parts)

def score_gemma_nested(cases: pd.DataFrame, premise_mode: str, model: str, api_base: str, n_folds=5, n_shots=8, seed=42, dry_run=False) -> pd.DataFrame:
    d = cases[cases[premise_mode + '_premise'].notna()].copy()
    rng = random.Random(seed)
    items = sorted(d.item_id.unique())
    rng.shuffle(items)
    folds = np.array_split(items, n_folds)
    out = []
    for fi, test_items in enumerate(folds):
        test_items = set(test_items)
        train = d[~d.item_id.isin(test_items) & d.is_core & ~d.is_negative_control]
        test = d[d.item_id.isin(test_items)]
        few = stratified_few_shot_sample(train, n_shots, rng) if premise_mode == 'record' else train.iloc[0:0]
        print(f'[Gemma {premise_mode} fold {fi + 1}/{n_folds}] test={len(test)}, few-shot={len(few)}')
        for _, r in test.iterrows():
            if dry_run:
                result = {'confidence_supported': 50.0, 'label': None, 'reasoning': 'dry-run'}
            else:
                prompt = _llm_fact_prompt(r) if premise_mode == 'fact' else _llm_prompt(few, r, premise_mode)
                result = None
                for _attempt in range(2):
                    result = call_judge_llm(prompt, model, api_key='none', backend='openai_compatible', api_base=api_base)
                    if result and result.get('confidence_supported') is not None:
                        break
            out.append({**r.to_dict(), 'score': None if not result or result.get('confidence_supported') is None else float(result['confidence_supported']) / 100.0, 'verifier': 'Gemma-3-27B-it', 'premise': premise_mode, 'judge_label': (result or {}).get('label'), 'judge_reasoning': (result or {}).get('reasoning')})
    out = pd.DataFrame(out)
    n_fail = int(out['score'].isna().sum()) if len(out) else 0
    print(f'[Gemma {premise_mode}] unparsable judgments: {n_fail}/{len(out)}')
    return out

def analysis_set(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[~scored.is_negative_control & scored.fact_premise.notna() & scored.majority_tag.ne('unknown') & scored.score.notna()]

def summarize_auroc(scored: pd.DataFrame, n_boot=2000, seed=42) -> pd.DataFrame:
    rows = []
    scored = analysis_set(scored)
    for (verifier, premise), g0 in scored.groupby(['verifier', 'premise']):
        for subset_name, g in [('core97', g0[g0.is_core]), ('all282', g0)]:
            for coding, positives in GROUND_TRUTH_DEFS.items():
                x = g[(g.majority_tag != 'unknown') & g.score.notna()].copy()
                y = x.majority_tag.isin(positives).astype(int).to_numpy()
                if len(x) == 0 or len(np.unique(y)) < 2:
                    continue
                auc, lo, hi = bootstrap_auroc_ci(y, x.score.to_numpy(float), n_boot=n_boot, seed=seed)
                rows.append({'verifier': verifier, 'premise': premise, 'subset': subset_name, 'coding': coding, 'n': len(x), 'auroc': auc, 'ci_lo': lo, 'ci_hi': hi, 'ci_includes_0.5': bool(lo <= 0.5 <= hi)})
    return pd.DataFrame(rows)

def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj

def paired_vs_baseline(scored: pd.DataFrame, n_boot=2000, seed=42) -> pd.DataFrame:
    rows = []
    scored = analysis_set(scored)
    for premise in sorted(scored.premise.unique()):
        for subset in ['core97', 'all282']:
            s = scored[scored.premise.eq(premise)]
            if subset == 'core97':
                s = s[s.is_core]
            base = s[s.verifier.eq(BASELINE_NAME)][['item_id', 'sentence_id', 'majority_tag', 'score']].rename(columns={'score': 'score_base'})
            if base.empty:
                continue
            for verifier in sorted(set(s.verifier) - {BASELINE_NAME}):
                other = s[s.verifier.eq(verifier)][['item_id', 'sentence_id', 'score']].rename(columns={'score': 'score_other'})
                m = base.merge(other, on=['item_id', 'sentence_id']).dropna()
                for coding, positives in GROUND_TRUTH_DEFS.items():
                    y = m.majority_tag.isin(positives).astype(int).to_numpy()
                    if len(m) == 0 or len(np.unique(y)) < 2:
                        continue
                    r = bootstrap_paired_auroc_diff_test(y, m.score_other.to_numpy(float), m.score_base.to_numpy(float), n_boot=n_boot, seed=seed)
                    rows.append({'verifier': verifier, 'premise': premise, 'subset': subset, 'coding': coding, 'n': len(m), 'delta_vs_mdeberta': r['diff'], 'diff_ci_lo': r['diff_ci_lo'], 'diff_ci_hi': r['diff_ci_hi'], 'p_raw': r['p_value']})
    out = pd.DataFrame(rows)
    if len(out):
        out['p_holm'] = holm_adjust(out.p_raw.to_numpy())
    return out

def paired_fact_vs_record(scored: pd.DataFrame, n_boot=2000, seed=42) -> pd.DataFrame:
    scored = analysis_set(scored)
    rows = []
    for verifier in sorted(scored.verifier.unique()):
        for subset in ['core97', 'all282']:
            s = scored[scored.verifier.eq(verifier)]
            if subset == 'core97':
                s = s[s.is_core]
            f = s[s.premise.eq('fact')][['item_id', 'sentence_id', 'majority_tag', 'score']]
            r = s[s.premise.eq('record')][['item_id', 'sentence_id', 'score']]
            m = f.merge(r, on=['item_id', 'sentence_id'], suffixes=('_fact', '_record')).dropna()
            for coding, positives in GROUND_TRUTH_DEFS.items():
                y = m.majority_tag.isin(positives).astype(int).to_numpy()
                if len(m) == 0 or len(np.unique(y)) < 2:
                    continue
                t = bootstrap_paired_auroc_diff_test(y, m.score_record.to_numpy(float), m.score_fact.to_numpy(float), n_boot=n_boot, seed=seed)
                rows.append({'verifier': verifier, 'subset': subset, 'coding': coding, 'n': len(m), 'delta_record_minus_fact': t['diff'], 'diff_ci_lo': t['diff_ci_lo'], 'diff_ci_hi': t['diff_ci_hi'], 'p_raw': t['p_value']})
    out = pd.DataFrame(rows)
    if len(out):
        out['p_holm'] = holm_adjust(out.p_raw.to_numpy())
    return out

def negative_controls(scored: pd.DataFrame, threshold=0.5) -> pd.DataFrame:
    d = scored[scored.score.notna()].copy()
    d['group'] = np.where(d.is_negative_control, 'control', 'normal')
    g = d.groupby(['verifier', 'premise', 'group']).score
    out = pd.DataFrame({'n': g.size(), 'mean_score': g.mean(), 'pass_rate': g.apply(lambda x: float((x >= threshold).mean()))})
    return out.reset_index()

def within_indicator(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scored = analysis_set(scored)
    d = scored[scored.indicator.notna() & scored.score.notna()]
    for (v, p, ind), g in d.groupby(['verifier', 'premise', 'indicator']):
        su = g[g.majority_tag.isin(['supported', 'unsupported'])]
        y = (su.majority_tag == 'supported').astype(int)
        auc = roc_auc_score(y, su.score) if y.nunique() == 2 else np.nan
        rows.append({'verifier': v, 'premise': p, 'indicator': ind, 'n': len(g), 'mean_score': g.score.mean(), 'auroc': auc, 'n_auroc': len(su), 'n_supported': int(y.sum()), 'n_unsupported': int((1 - y).sum())})
    return pd.DataFrame(rows)

def check_baseline_reproduction(summary: pd.DataFrame, expected: Dict[tuple, tuple], tol=0.005):
    base = summary[(summary.verifier == BASELINE_NAME) & (summary.premise == 'fact')]
    problems = []
    for (subset, coding), (n_exp, auc_exp) in expected.items():
        row = base[(base.subset == subset) & (base.coding == coding)]
        if row.empty:
            problems.append(f'{subset}/{coding}: missing')
            continue
        n, auc = (int(row.n.iloc[0]), float(row.auroc.iloc[0]))
        if n != n_exp or abs(auc - auc_exp) > tol:
            problems.append(f'{subset}/{coding}: got n={n}, AUROC={auc:.3f}; paper n={n_exp}, AUROC={auc_exp:.3f}')
    return problems

def _checkpoint_path(output_dir: str, verifier_key: str) -> str:
    return os.path.join(output_dir, f'scores_{verifier_key}.pkl')

def _run_local_worker(args, verifier_key: str, cases_path: str, output_path: str):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--_worker-verifier', verifier_key, '--_worker-cases', cases_path, '--_worker-output', output_path, '--minicheck-gpu-memory-utilization', str(args.minicheck_gpu_memory_utilization), '--minicheck-max-model-len', str(args.minicheck_max_model_len), '--minicheck-tensor-parallel-size', str(args.minicheck_tensor_parallel_size), '--minicheck-cache-dir', args.minicheck_cache_dir]
    print('\n[isolated worker]', ' '.join(cmd))
    subprocess.run(cmd, check=True)

def _worker_main(args):
    cases = pd.read_pickle(args._worker_cases)
    if args._worker_verifier == 'mdeberta':
        scorer = NliPairScorer(BASE_NLI, BASELINE_NAME)
    elif args._worker_verifier == 'deberta_large':
        scorer = NliPairScorer(LARGE_NLI, 'DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
    elif args._worker_verifier == 'minicheck':
        scorer = MiniCheckScorer(gpu_memory_utilization=args.minicheck_gpu_memory_utilization, max_model_len=args.minicheck_max_model_len, tensor_parallel_size=args.minicheck_tensor_parallel_size, cache_dir=args.minicheck_cache_dir)
    else:
        raise ValueError(f'Unknown worker verifier: {args._worker_verifier}')
    parts = []
    try:
        for pm in ['fact', 'record']:
            parts.append(score_pair_model(cases, scorer, pm))
        scored = pd.concat(parts, ignore_index=True)
        scored.to_pickle(args._worker_output)
        print(f'[worker] wrote {len(scored)} rows -> {args._worker_output}')
    finally:
        scorer.close()

def _build_parser():
    ap = argparse.ArgumentParser(description="Run the verifier-family robustness experiment.", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--responses-dir', default='data/ratings')
    ap.add_argument('--stimuli-master', default='data/stimuli/stimuli_master.json')
    ap.add_argument('--answer-key', default='data/stimuli/internal_answer_key.csv')
    ap.add_argument('--sentence-answer-key', default='data/stimuli/internal_answer_key_sentences.csv')
    ap.add_argument('--source-csv', default='data/interim/full_predictions2.csv')
    ap.add_argument('--verifiers', nargs='+', default=['mdeberta', 'deberta_large', 'minicheck', 'gemma27b'], choices=['mdeberta', 'deberta_large', 'minicheck', 'gemma27b'])
    ap.add_argument('--gemma-model', default=GEMMA27B)
    ap.add_argument('--gemma-api-base', default='http://localhost:8000/v1')
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--n-shots', type=int, default=8)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run-gemma', action='store_true')
    ap.add_argument('--output-dir', default='outputs/verifier_family')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--allow-missing-premises', action='store_true')
    ap.add_argument('--skip-reproduction-check', action='store_true')
    ap.add_argument('--minicheck-gpu-memory-utilization', type=float, default=0.25)
    ap.add_argument('--minicheck-max-model-len', type=int, default=4096)
    ap.add_argument('--minicheck-tensor-parallel-size', type=int, default=1)
    ap.add_argument('--minicheck-cache-dir', default='./ckpts')
    ap.add_argument('--_worker-verifier', choices=['mdeberta', 'deberta_large', 'minicheck'])
    ap.add_argument('--_worker-cases')
    ap.add_argument('--_worker-output')
    return ap

def main():
    ap = _build_parser()
    args = ap.parse_args()
    if args._worker_verifier:
        if not args._worker_cases or not args._worker_output:
            ap.error('worker mode requires --_worker-cases and --_worker-output')
        _worker_main(args)
        return
    required = ['responses_dir', 'stimuli_master', 'answer_key', 'sentence_answer_key', 'source_csv']
    missing = [x for x in required if not getattr(args, x)]
    if missing:
        ap.error('missing required parent-mode arguments: ' + ', '.join(('--' + x.replace('_', '-') for x in missing)))
    if not 0.0 < args.minicheck_gpu_memory_utilization <= 1.0:
        ap.error('--minicheck-gpu-memory-utilization must be in (0, 1].')
    if args.minicheck_max_model_len <= 300:
        ap.error('--minicheck-max-model-len must be > 300.')
    if 'gemma27b' in args.verifiers and 'qwen' in args.gemma_model.lower():
        raise ValueError('Qwen is the generator and is excluded as a judge.')
    os.makedirs(args.output_dir, exist_ok=True)
    cases_path = os.path.join(args.output_dir, 'cases.pkl')
    rebuilt_cases = False
    if args.resume and os.path.exists(cases_path):
        print(f'[resume] loading {cases_path}')
        cases = pd.read_pickle(cases_path)
        stale = 'fact_premise' not in cases.columns or 'cited_indicators' not in cases.columns or 'fact_premise_list' not in cases.columns or ('invalid_citations' not in cases.columns) or ('is_negative_control' not in cases.columns) or (int(cases.get('fact_premise', pd.Series(dtype=object)).notna().sum()) == 0) or (int(cases.get('cited_indicators', pd.Series(dtype=object)).map(bool).sum()) == 0)
        if stale:
            print('[resume] stale/broken cases.pkl detected; rebuilding cases and invalidating verifier score checkpoints')
            cases = build_cases(args.responses_dir, args.stimuli_master, args.answer_key, args.sentence_answer_key, args.source_csv)
            rebuilt_cases = True
    else:
        cases = build_cases(args.responses_dir, args.stimuli_master, args.answer_key, args.sentence_answer_key, args.source_csv)
        rebuilt_cases = True
    _validate_cases(cases, allow_missing=args.allow_missing_premises)
    if rebuilt_cases:
        cases.to_pickle(cases_path)
        case_cols = [c for c in PUBLIC_CASE_COLUMNS if c in cases.columns]
        cases[case_cols].to_csv(os.path.join(args.output_dir, 'cases.csv'), index=False)
    parts = []
    for key in ['mdeberta', 'deberta_large', 'minicheck']:
        if key not in args.verifiers:
            continue
        ckpt = _checkpoint_path(args.output_dir, key)
        if args.resume and (not rebuilt_cases) and os.path.exists(ckpt):
            print(f'[resume] reusing {ckpt}')
        else:
            _run_local_worker(args, key, cases_path, ckpt)
        parts.append(pd.read_pickle(ckpt))
    if 'gemma27b' in args.verifiers:
        ckpt = _checkpoint_path(args.output_dir, 'gemma27b')
        if args.resume and (not rebuilt_cases) and os.path.exists(ckpt):
            print(f'[resume] reusing {ckpt}')
            gemma_scored = pd.read_pickle(ckpt)
        else:
            gemma_parts = []
            for pm in ['fact', 'record']:
                gemma_parts.append(score_gemma_nested(cases, pm, args.gemma_model, args.gemma_api_base, args.n_folds, args.n_shots, args.seed, args.dry_run_gemma))
            gemma_scored = pd.concat(gemma_parts, ignore_index=True)
            if args.dry_run_gemma:
                gemma_scored.to_pickle(ckpt.replace('.pkl', '_DRYRUN.pkl'))
            else:
                gemma_scored.to_pickle(ckpt)
        parts.append(gemma_scored)
    if not parts:
        raise RuntimeError('No verifier scores were produced.')
    scored = pd.concat(parts, ignore_index=True)
    scored.to_pickle(os.path.join(args.output_dir, 'scores_all_verifiers.pkl'))
    score_cols = [c for c in PUBLIC_SCORE_COLUMNS if c in scored.columns]
    scored[score_cols].to_csv(os.path.join(args.output_dir, 'scores_all_verifiers.csv'), index=False)
    summary = summarize_auroc(scored, args.n_boot, args.seed)
    if not args.skip_reproduction_check:
        problems = check_baseline_reproduction(summary, {('all282', 'strict'): (282, 0.479), ('all282', 'lenient'): (282, 0.482), ('core97', 'strict'): (97, 0.591), ('core97', 'lenient'): (97, 0.591)})
        if problems:
            raise RuntimeError('Baseline does not reproduce the paper; do not interpret other rows:\n  ' + '\n  '.join(problems))
        print('[check] mDeBERTa/fact reproduces Tables 3-4.')
    summary.to_csv(os.path.join(args.output_dir, 'table4_verifier_x_premise.csv'), index=False)
    within = within_indicator(scored)
    within.to_csv(os.path.join(args.output_dir, 'table5_within_indicator_by_verifier_premise.csv'), index=False)
    paired = paired_vs_baseline(scored, args.n_boot, args.seed)
    paired.to_csv(os.path.join(args.output_dir, 'paired_vs_mdeberta_holm.csv'), index=False)
    premise_diff = paired_fact_vs_record(scored, args.n_boot, args.seed)
    premise_diff.to_csv(os.path.join(args.output_dir, 'paired_record_vs_fact_holm.csv'), index=False)
    controls = negative_controls(scored)
    controls.to_csv(os.path.join(args.output_dir, 'negative_controls_by_premise.csv'), index=False)
if __name__ == '__main__':
    main()
