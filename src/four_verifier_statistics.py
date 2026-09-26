from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
VERIFIER_ORDER = ['mDeBERTa-v3-base-mnli-xnli', 'DeBERTa-v3-large-mnli-fever-anli-ling-wanli', 'Bespoke-MiniCheck-7B', 'Gemma-3-27B-it']
MDEBERTA = 'mDeBERTa-v3-base-mnli-xnli'
KEY = ['item_id', 'sentence_id']
VALID_PREMISES = {'fact', 'record'}
VALID_TAGS = {'supported', 'partial', 'unsupported', 'unknown'}

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {'.pkl', '.pickle'}:
        return pd.read_pickle(p)
    if p.suffix.lower() == '.csv':
        return pd.read_csv(p)
    raise ValueError(f'Unsupported input format: {p.suffix}')

def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    m = {'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any():
        bad = s[out.isna()].drop_duplicates().tolist()[:10]
        raise ValueError(f'Cannot parse boolean values: {bad}')
    return out.astype(bool)

def validate(df: pd.DataFrame) -> pd.DataFrame:
    required = {'item_id', 'sentence_id', 'majority_tag', 'is_core', 'is_negative_control', 'score', 'verifier', 'premise'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    d = df.copy()
    d['is_core'] = as_bool(d['is_core'])
    d['is_negative_control'] = as_bool(d['is_negative_control'])
    d['score'] = pd.to_numeric(d['score'], errors='coerce')
    d['premise'] = d['premise'].astype(str).str.strip().str.lower()
    d['majority_tag'] = d['majority_tag'].astype(str).str.strip().str.lower()
    d.loc[d['majority_tag'].isin(['nan', 'none', '']), 'majority_tag'] = np.nan
    typo_mask = d['majority_tag'].eq('unsuppported')
    if typo_mask.any():
        warnings.warn(f'Normalizing {int(typo_mask.sum())} majority_tag value(s): unsuppported -> unsupported')
        d.loc[typo_mask, 'majority_tag'] = 'unsupported'
    unknown_verifiers = sorted(set(d['verifier'].dropna()) - set(VERIFIER_ORDER))
    missing_verifiers = sorted(set(VERIFIER_ORDER) - set(d['verifier'].dropna()))
    if unknown_verifiers:
        warnings.warn(f'Unexpected verifier names found: {unknown_verifiers}')
    if missing_verifiers:
        raise ValueError(f'Missing required verifiers: {missing_verifiers}')
    bad_premises = sorted(set(d['premise'].dropna()) - VALID_PREMISES)
    if bad_premises:
        raise ValueError(f'Unexpected premise values: {bad_premises}')
    bad_tags = sorted(set(d['majority_tag'].dropna()) - VALID_TAGS)
    if bad_tags:
        raise ValueError(f'Unexpected majority_tag values: {bad_tags}')
    dup = d.duplicated(['verifier', 'premise'] + KEY, keep=False)
    if dup.any():
        ex = d.loc[dup, ['verifier', 'premise'] + KEY].head(10)
        raise ValueError('Duplicate verifier/premise/case keys found:\n' + ex.to_string(index=False))
    return d

def coding_y(tags: pd.Series, coding: str) -> np.ndarray:
    if coding == 'strict':
        return (tags == 'supported').astype(int).to_numpy()
    if coding == 'lenient':
        return tags.isin(['supported', 'partial']).astype(int).to_numpy()
    raise ValueError(coding)

def analysis_mask(d: pd.DataFrame, subset: str) -> pd.Series:
    m = ~d['is_negative_control'] & d['majority_tag'].isin(['supported', 'partial', 'unsupported']) & d['score'].notna()
    if subset == 'core97':
        m &= d['is_core']
    elif subset != 'all282':
        raise ValueError(subset)
    return m

def auc(y, s):
    y, s = (np.asarray(y), np.asarray(s, dtype=float))
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))

def bootstrap_auc(y, s, n_boot=2000, seed=42):
    y, s = (np.asarray(y), np.asarray(s, dtype=float))
    point = auc(y, s)
    if not np.isfinite(point):
        return (point, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return (point, np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (point, float(lo), float(hi))

def paired_auc_diff(y, a, b, n_boot=2000, seed=42):
    y = np.asarray(y)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    point = auc(y, a) - auc(y, b)
    rng = np.random.RandomState(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (point, np.nan, np.nan, np.nan)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    n_le = int(np.sum(vals <= 0))
    n_ge = int(np.sum(vals >= 0))
    p = min(1.0, 2.0 * (min(n_le, n_ge) + 1) / (len(vals) + 1))
    return (float(point), float(lo), float(hi), float(p))

def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, j in enumerate(order):
        adj = (m - rank) * p[j]
        running = max(running, adj)
        out[j] = min(1.0, running)
    return out

def diagnostics(df: pd.DataFrame, expected_core=None, expected_all=None, fail=False):
    rows = []
    for v in VERIFIER_ORDER:
        for p in ['fact', 'record']:
            g = df[(df.verifier == v) & (df.premise == p)]
            for subset in ['core97', 'all282']:
                a = g[analysis_mask(g, subset)]
                rows.append({'verifier': v, 'premise': p, 'subset': subset, 'n_raw': len(g), 'n_analysis': len(a), 'n_score_missing': int(g.score.isna().sum()), 'n_unknown': int((g.majority_tag == 'unknown').sum()), 'n_negative_control': int(g.is_negative_control.sum()), 'n_supported': int((a.majority_tag == 'supported').sum()), 'n_partial': int((a.majority_tag == 'partial').sum()), 'n_unsupported': int((a.majority_tag == 'unsupported').sum())})
    out = pd.DataFrame(rows)
    problems = []
    for subset, exp in [('core97', expected_core), ('all282', expected_all)]:
        if exp is None:
            continue
        bad = out[(out.subset == subset) & (out.n_analysis != exp)]
        if len(bad):
            problems.append(f'{subset}: expected n={exp}, observed ' + ', '.join((f'{r.verifier}/{r.premise}={r.n_analysis}' for r in bad.itertuples())))
    if problems:
        msg = 'Analysis-set size mismatch:\n  ' + '\n  '.join(problems)
        if fail:
            raise RuntimeError(msg)
        warnings.warn(msg)
    return out

def table4(df, n_boot, seed):
    rows = []
    for vi, v in enumerate(VERIFIER_ORDER):
        for pi, premise in enumerate(['fact', 'record']):
            g0 = df[(df.verifier == v) & (df.premise == premise)]
            for si, subset in enumerate(['core97', 'all282']):
                g = g0[analysis_mask(g0, subset)].sort_values(KEY)
                for ci, coding in enumerate(['strict', 'lenient']):
                    y = coding_y(g.majority_tag, coding)
                    point, lo, hi = bootstrap_auc(y, g.score.to_numpy(), n_boot, seed + 1000 * vi + 100 * pi + 10 * si + ci)
                    rows.append({'verifier': v, 'premise': premise, 'subset': subset, 'coding': coding, 'n': len(g), 'auroc': point, 'ci_lo': lo, 'ci_hi': hi, 'ci_includes_0.5': bool(lo <= 0.5 <= hi) if np.isfinite(lo) else np.nan})
    return pd.DataFrame(rows)

def single_indicator_series(d):
    if 'indicator' in d.columns:
        ind = d['indicator'].astype(str).str.strip()
        ind = ind.mask(ind.isin(['', 'nan', 'None', '[]']))
    else:
        ind = pd.Series(np.nan, index=d.index, dtype=object)
    if 'cited_indicators' in d.columns:
        raw = d['cited_indicators'].astype(str).str.strip()
        clean = raw.str.replace('^\\[|\\]$', '', regex=True).str.replace("'", '', regex=False).str.replace('"', '', regex=False).str.strip()
        single = ~clean.str.contains('[,;|]', regex=True) & ~clean.isin(['', 'nan', 'None'])
        ind = ind.where(ind.notna(), clean.where(single))
    return ind

def table5(df):
    rows = []
    for v in VERIFIER_ORDER:
        for premise in ['fact', 'record']:
            g = df[(df.verifier == v) & (df.premise == premise) & ~df.is_negative_control & df.score.notna()].copy()
            g = g[g.majority_tag.isin(['supported', 'partial', 'unsupported'])]
            g['_indicator'] = single_indicator_series(g)
            g = g[g._indicator.notna()]
            for ind, h in g.groupby('_indicator'):
                su = h[h.majority_tag.isin(['supported', 'unsupported'])]
                y = (su.majority_tag == 'supported').astype(int).to_numpy()
                rows.append({'verifier': v, 'premise': premise, 'indicator': ind, 'n': len(h), 'mean_score': h.score.mean(), 'auroc': auc(y, su.score.to_numpy()), 'n_auroc': len(su), 'n_supported': int(y.sum()), 'n_unsupported': int(len(y) - y.sum())})
            su = g[g.majority_tag.isin(['supported', 'unsupported'])]
            y = (su.majority_tag == 'supported').astype(int).to_numpy()
            rows.append({'verifier': v, 'premise': premise, 'indicator': 'pooled', 'n': len(g), 'mean_score': np.nan, 'auroc': auc(y, su.score.to_numpy()), 'n_auroc': len(su), 'n_supported': int(y.sum()), 'n_unsupported': int(len(y) - y.sum())})
    return pd.DataFrame(rows)

def paired_vs_mdeberta(df, n_boot, seed):
    rows = []
    others = [v for v in VERIFIER_ORDER if v != MDEBERTA]
    for vi, v in enumerate(others):
        for pi, premise in enumerate(['fact', 'record']):
            for si, subset in enumerate(['core97', 'all282']):
                a = df[(df.verifier == v) & (df.premise == premise)]
                b = df[(df.verifier == MDEBERTA) & (df.premise == premise)]
                a = a[analysis_mask(a, subset)][KEY + ['majority_tag', 'score']].rename(columns={'score': 'score_a', 'majority_tag': 'tag_a'})
                b = b[analysis_mask(b, subset)][KEY + ['majority_tag', 'score']].rename(columns={'score': 'score_b', 'majority_tag': 'tag_b'})
                m = a.merge(b, on=KEY, how='inner', validate='one_to_one')
                m = m[m.tag_a == m.tag_b].sort_values(KEY)
                for ci, coding in enumerate(['strict', 'lenient']):
                    y = coding_y(m.tag_a, coding)
                    delta, lo, hi, p = paired_auc_diff(y, m.score_a, m.score_b, n_boot, seed + 1000 * vi + 100 * pi + 10 * si + ci)
                    rows.append({'verifier': v, 'premise': premise, 'subset': subset, 'coding': coding, 'n': len(m), 'delta_vs_mdeberta': delta, 'diff_ci_lo': lo, 'diff_ci_hi': hi, 'p_raw': p})
    out = pd.DataFrame(rows)
    out['p_holm'] = holm_adjust(out.p_raw.to_numpy())
    return out

def paired_record_vs_fact(df, n_boot, seed):
    rows = []
    for vi, v in enumerate(VERIFIER_ORDER):
        f = df[(df.verifier == v) & (df.premise == 'fact')]
        r = df[(df.verifier == v) & (df.premise == 'record')]
        for si, subset in enumerate(['core97', 'all282']):
            ff = f[analysis_mask(f, subset)][KEY + ['majority_tag', 'score']].rename(columns={'score': 'fact_score', 'majority_tag': 'fact_tag'})
            rr = r[analysis_mask(r, subset)][KEY + ['majority_tag', 'score']].rename(columns={'score': 'record_score', 'majority_tag': 'record_tag'})
            m = ff.merge(rr, on=KEY, how='inner', validate='one_to_one')
            m = m[m.fact_tag == m.record_tag].sort_values(KEY)
            for ci, coding in enumerate(['strict', 'lenient']):
                y = coding_y(m.fact_tag, coding)
                delta, lo, hi, p = paired_auc_diff(y, m.record_score, m.fact_score, n_boot, seed + 1000 * vi + 10 * si + ci)
                rows.append({'verifier': v, 'subset': subset, 'coding': coding, 'n': len(m), 'delta_record_minus_fact': delta, 'diff_ci_lo': lo, 'diff_ci_hi': hi, 'p_raw': p})
    out = pd.DataFrame(rows)
    out['p_holm'] = holm_adjust(out.p_raw.to_numpy())
    return out

def negative_controls(df):
    rows = []
    for v in VERIFIER_ORDER:
        for premise in ['fact', 'record']:
            g = df[(df.verifier == v) & (df.premise == premise) & df.score.notna()]
            for flag, name in [(False, 'normal'), (True, 'control')]:
                h = g[g.is_negative_control == flag]
                if len(h) == 0:
                    continue
                rows.append({'verifier': v, 'premise': premise, 'group': name, 'n': len(h), 'mean_score': h.score.mean(), 'pass_rate': (h.score >= 0.5).mean()})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="Analyze verifier-family scores and export summary tables.")
    ap.add_argument('--input', default='outputs/verifier_family/scores_all_verifiers.csv', help='scores_all_verifiers.pkl or .csv')
    ap.add_argument('--output-dir', default='outputs/verifier_family')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--expected-core', type=int, default=None)
    ap.add_argument('--expected-all', type=int, default=None)
    ap.add_argument('--fail-on-size-mismatch', action='store_true')
    args = ap.parse_args()
    if args.n_boot < 100:
        warnings.warn('n_boot < 100 is suitable only for smoke tests, not final inference.')
    df = validate(read_table(args.input))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    diag = diagnostics(df, args.expected_core, args.expected_all, args.fail_on_size_mismatch)
    diag.to_csv(outdir / 'diagnostics.csv', index=False)
    print('\n=== Diagnostics ===')
    print(diag.to_string(index=False))
    outputs = {'table4_verifier_x_premise.csv': table4(df, args.n_boot, args.seed), 'table5_within_indicator_by_verifier_premise.csv': table5(df), 'paired_vs_mdeberta_holm.csv': paired_vs_mdeberta(df, args.n_boot, args.seed), 'paired_record_vs_fact_holm.csv': paired_record_vs_fact(df, args.n_boot, args.seed), 'negative_controls_by_premise.csv': negative_controls(df)}
    for name, tab in outputs.items():
        tab.to_csv(outdir / name, index=False)
        print(f'[wrote] {outdir / name} ({len(tab)} rows)')
    print('\n=== Table 4 ===')
    print(outputs['table4_verifier_x_premise.csv'].round(4).to_string(index=False))
    print('\nDone.')
if __name__ == '__main__':
    main()
