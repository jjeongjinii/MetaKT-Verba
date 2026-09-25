


import argparse

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

PAIRS = [('1A', '1B'), ('2A', '2B')]
STANDALONE = ['3', '4']
VALID_ITEMS = {'1A', '1B', '2A', '2B', '3', '4'}


def load_responses(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'item_id': str}, encoding='utf-8-sig')
    df['item_id'] = df['item_id'].str.strip().str.upper()
    bad = set(df['item_id']) - VALID_ITEMS
    if bad:
        raise ValueError(f"알 수 없는 item_id가 있습니다: {bad} (허용값: {sorted(VALID_ITEMS)})")
    if not df['agreement_score'].between(1, 5).all():
        raise ValueError("agreement_score는 1~5 사이여야 합니다.")
    return df


def describe_items(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('item_id')['agreement_score'].agg(['count', 'mean', 'std']).round(2)


def pair_test(df: pd.DataFrame, item_a: str, item_b: str) -> dict:
    wide = df[df['item_id'].isin([item_a, item_b])].pivot(
        index='rater_id', columns='item_id', values='agreement_score'
    ).dropna()
    if wide.empty:
        return {'error': f'{item_a}/{item_b} 둘 다 응답한 평정자가 없습니다.'}

    a = wide[item_a].to_numpy(dtype=float)
    b = wide[item_b].to_numpy(dtype=float)
    diff = a - b
    n_a_gt_b = int((diff > 0).sum())
    n_b_gt_a = int((diff < 0).sum())
    n_tie = int((diff == 0).sum())

    result = {
        'n': len(wide), 'mean_a': a.mean(), 'mean_b': b.mean(), 'mean_diff': diff.mean(),
        'n_a_gt_b': n_a_gt_b, 'n_b_gt_a': n_b_gt_a, 'n_tie': n_tie,
    }
    if _HAS_SCIPY and len(wide) >= 2 and not np.all(diff == diff[0]):
        try:
            stat, p = scipy_stats.wilcoxon(a, b)
            result['wilcoxon_stat'] = float(stat)
            result['wilcoxon_p'] = float(p)
        except ValueError as e:
            result['wilcoxon_note'] = f'검정 불가: {e}'
    else:
        result['wilcoxon_note'] = ('scipy 미설치 또는 표본이 너무 작거나 차이가 전부 동일해 '
                                    '검정 통계량을 계산하지 않았습니다. 위 카운트로 방향성만 판단하세요.')
    return result


def format_report(desc: pd.DataFrame, pair_results: dict, standalone_desc: pd.DataFrame) -> str:
    lines = ["=" * 70, "cognitive_avoidance 미니 서베이 분석 결과", "=" * 70, ""]
    lines.append("[1] 문항별 기술통계")
    lines.append(desc.to_string())
    lines.append("")

    lines.append("[2] 페어 비교 (1A vs 1B, 2A vs 2B)")
    for (a, b), res in pair_results.items():
        if 'error' in res:
            lines.append(f"  - {a} vs {b}: {res['error']}")
            continue
        lines.append(f"  - {a}(평균 {res['mean_a']:.2f}) vs {b}(평균 {res['mean_b']:.2f}), "
                     f"차이={res['mean_diff']:.2f} (n={res['n']})")
        lines.append(f"    · {a}>{b}: {res['n_a_gt_b']}명 / {b}>{a}: {res['n_b_gt_a']}명 / 동점: {res['n_tie']}명")
        if 'wilcoxon_p' in res:
            lines.append(f"    · Wilcoxon signed-rank: stat={res['wilcoxon_stat']:.2f}, p={res['wilcoxon_p']:.4f}")
        elif 'wilcoxon_note' in res:
            lines.append(f"    · {res['wilcoxon_note']}")
    lines.append("")

    lines.append("[3] 대조 문항(3=극단적 힌트, 4=원형 사례) 기술통계")
    lines.append(standalone_desc.to_string())
    lines.append("")

    lines.append("[4] 자동 해석 (mini_survey.md의 분석 방법 절 기준)")
    p1 = pair_results.get(('1A', '1B'), {})
    p2 = pair_results.get(('2A', '2B'), {})
    diffs = [p.get('mean_diff') for p in (p1, p2) if 'mean_diff' in p]
    if diffs and all(d > 0.5 for d in diffs):
        lines.append("  - 두 페어 모두 '힌트 미사용형(A) > 힌트 사용형(B)' 방향이 뚜렷합니다 "
                     "(차이 0.5점 이상). 힌트 궤적이 실제로 '회피' 판단을 가른다는 증거입니다 "
                     "— Stage 2가 cognitive_avoidance를 서술할 때 힌트 궤적 정보를 반영해 "
                     "확신도를 조절하도록 프롬프트/후처리를 조정할 근거가 됩니다.")
    elif diffs and any(d > 0.5 for d in diffs):
        lines.append("  - 두 페어 중 일부만 뚜렷한 방향을 보입니다 — 문항별 이유(reason) 텍스트를 "
                     "직접 읽고 어떤 조건에서 갈리는지 확인이 필요합니다.")
    else:
        lines.append("  - 페어 간 차이가 뚜렷하지 않습니다 — 힌트 궤적 자체보다 다른 요인이 "
                     "판단을 좌우하고 있을 수 있습니다. reason 텍스트를 정성적으로 검토하세요.")

    item3_mean = standalone_desc.loc['3', 'mean'] if '3' in standalone_desc.index else None
    item4_mean = standalone_desc.loc['4', 'mean'] if '4' in standalone_desc.index else None
    if item3_mean is not None:
        tag = "낮음(비동의 쪽)" if item3_mean <= 2.5 else ("중간" if item3_mean <= 3.5 else "높음(동의 쪽)")
        lines.append(f"  - 문항 3(힌트 7~8개, 실제 Q113 사례) 평균 {item3_mean:.2f} — {tag}. "
                     "낮을수록 '힌트를 많이 쓰고도 회피로 서술되는' 현재 파이프라인 동작이 "
                     "반직관적이라는 게 지지됩니다.")
    if item4_mean is not None:
        tag = "낮음" if item4_mean <= 2.5 else ("중간" if item4_mean <= 3.5 else "높음(동의 쪽, 원래 의도대로)")
        lines.append(f"  - 문항 4(원형 사례) 평균 {item4_mean:.2f} — {tag}. 이게 낮다면 문제가 "
                     "힌트 방향성이 아니라 '회피'라는 서술 자체의 확신도 문제일 수 있습니다.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def run(responses_path: str):
    df = load_responses(responses_path)
    desc = describe_items(df)
    pair_results = {(a, b): pair_test(df, a, b) for a, b in PAIRS}
    standalone_desc = df[df['item_id'].isin(STANDALONE)].groupby('item_id')['agreement_score'] \
        .agg(['count', 'mean', 'std']).round(2)
    report = format_report(desc, pair_results, standalone_desc)
    print(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=str, default="data/minimal_pairs/mini_survey_responses.csv", help="응답 CSV 경로 (rater_id,item_id,agreement_score,reason)")

    args = parser.parse_args()


    if not args.responses:
        parser.error("--responses가 필요합니다")
    run(args.responses)


if __name__ == "__main__":
    main()