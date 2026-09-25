
import re
from dataclasses import dataclass, field
from typing import Optional


HINT_ZERO_EPS = 1e-6
HIGH_THRESHOLD = 0.5
SC_AMBIGUOUS_BAND = 0.15



_TRIAD_NOTE_INACTIVE = {
    'ko': '낮은 확신/추측/생산적 분투 관련 신호 모두 미약함',
    'en': 'All signals related to low confidence, guessing, and productive struggle are weak.',
}
_TRIAD_NOTE_WEAK_HINTUSED = {
    'ko': '세 지표 모두 약함',
    'en': 'All three indicators are weak.',
}
_TRIAD_DIRECTION_HINT = {
    'ko': {
        'guess': "(참고: 확신도가 낮아 추측 쪽에 약간 더 가까운 신호)",
        'struggle': "(참고: 확신도가 높아 생산적 분투 쪽에 약간 더 가까운 신호)",
        'mid': "(확신도도 중간값이라 방향성조차 약함)",
    },
    'en': {
        'guess': "(Note: confidence is low, so the signal leans slightly toward guessing.)",
        'struggle': "(Note: confidence is high, so the signal leans slightly toward productive struggle.)",
        'mid': "(Confidence is also mid-range, so even the direction of the signal is weak.)",
    },
}
_TRIAD_AMBIGUOUS_NOTE = {
    'ko': ("힌트 미사용 상태에서 정답을 맞혔으나, 이것이 낮은 확신 속의 "
           "정답(underconfidence), 생산적 분투 끝의 정답(productive_struggle), "
           "우연한 정답(lucky_guess) 중 무엇인지는 현재 데이터로 구조적으로 "
           "구분 불가능함. {direction_hint}"),
    'en': ("The student answered correctly without using a hint, but the current data "
           "cannot structurally distinguish whether this reflects a correct answer despite "
           "low confidence (underconfidence), a correct answer after productive struggle "
           "(productive_struggle), or a lucky guess (lucky_guess). {direction_hint}"),
}
_TRIAD_NOTE_HINTUSED_DOMINANT = {
    'ko': '{dominant} 우세 (힌트 사용 구간이라 독립 판정 가능)',
    'en': '{dominant} is dominant (independent judgment is possible since a hint was used).',
}
_WEAK_SIGNAL_NOTE = {
    'ko': '약한 신호, 서술 시 단정 표현 지양 권장',
    'en': 'Weak signal — avoid definitive phrasing when describing this.',
}
_OVERCONF_SLIP_OVERLAP_NOTE = {
    'ko': (" [지시: overconfidence와 slipping이 함께 나타났지만, "
           "이 둘의 관계나 상관성에 대해 언급하는 문장은 절대 추가하지 마세요 "
           "— 마치 둘 중 하나만 있는 것처럼, 각각을 있는 그대로 독립적으로만 서술하세요.]"),
    'en': (" [Instruction: overconfidence and slipping are both present here, "
           "but do not add any sentence commenting on the relationship or correlation between them "
           "— just describe each one plainly and independently, exactly as you would if only that one indicator applied.]"),
}


_BRACKETED_INSTRUCTION_PATTERN = re.compile(r'\s*\[[^\]]*\]\s*')


def strip_generator_only_instructions(note: str) -> str:
    return _BRACKETED_INSTRUCTION_PATTERN.sub(' ', note).strip()


def _strip_m_prefix(key: str) -> str:
    return key[2:] if key.startswith('m_') else key


@dataclass
class SymbolicFact:
    indicator: str
    status: str              # 'HIGH' | 'LOW' | 'AMBIGUOUS'
    value: Optional[float] = None
    note: str = ""


def resolve_confidence_triad(m_underconfidence: float,
                              m_productive_struggle: float,
                              m_lucky_guess: float,
                              hint_count: float,
                              sc: Optional[float] = None,
                              lang: str = 'ko') -> SymbolicFact:
    triad_active = any(v > HIGH_THRESHOLD * 0.3 for v in
                        (m_underconfidence, m_productive_struggle, m_lucky_guess))
    if not triad_active:
        return SymbolicFact(indicator='confidence_triad', status='LOW',
                             note=_TRIAD_NOTE_INACTIVE[lang])

    if hint_count <= HINT_ZERO_EPS:
        direction_hint = ""
        if sc is not None:
            hints = _TRIAD_DIRECTION_HINT[lang]
            if sc < 0.5 - SC_AMBIGUOUS_BAND:
                direction_hint = hints['guess']
            elif sc > 0.5 + SC_AMBIGUOUS_BAND:
                direction_hint = hints['struggle']
            else:
                direction_hint = hints['mid']

        return SymbolicFact(
            indicator='confidence_triad',
            status='AMBIGUOUS',
            note=_TRIAD_AMBIGUOUS_NOTE[lang].format(direction_hint=direction_hint)
        )

    scores = {
        'underconfidence': m_underconfidence,
        'productive_struggle': m_productive_struggle,
        'lucky_guess': m_lucky_guess,
    }
    dominant = max(scores, key=scores.get)
    if scores[dominant] <= HIGH_THRESHOLD * 0.3:
        return SymbolicFact(indicator='confidence_triad', status='LOW',
                             note=_TRIAD_NOTE_WEAK_HINTUSED[lang])

    return SymbolicFact(
        indicator=dominant,
        status='HIGH',
        value=scores[dominant],
        note=_TRIAD_NOTE_HINTUSED_DOMINANT[lang].format(dominant=dominant)
    )


def build_symbolic_facts(meta_vector: dict, hint_count: float, sc: Optional[float] = None,
                          lang: str = 'ko') -> list:
    facts = []

    facts.append(resolve_confidence_triad(
        meta_vector.get('m_underconfidence', 0.0),
        meta_vector.get('m_productive_struggle', 0.0),
        meta_vector.get('m_lucky_guess', 0.0),
        hint_count,
        sc,
        lang=lang,
    ))

    simple_indicators = [
        'm_overconfidence', 'm_strategic_help', 'm_cognitive_avoidance',
        'm_unproductive_frustration', 'm_slipping', 'm_boredom_offtask',
    ]
    for key in simple_indicators:
        val = meta_vector.get(key, 0.0)
        name = _strip_m_prefix(key)
        if val > HIGH_THRESHOLD:
            facts.append(SymbolicFact(indicator=name, status='HIGH', value=val))
        elif val > HIGH_THRESHOLD * 0.3:
            facts.append(SymbolicFact(indicator=name, status='LOW', value=val,
                                       note=_WEAK_SIGNAL_NOTE[lang]))

    oc_fact = next((f for f in facts if f.indicator == 'overconfidence' and f.status == 'HIGH'), None)
    sl_fact = next((f for f in facts if f.indicator == 'slipping' and f.status == 'HIGH'), None)
    if oc_fact is not None and sl_fact is not None:
        overlap_note = _OVERCONF_SLIP_OVERLAP_NOTE[lang]
        oc_fact.note = (oc_fact.note or "") + overlap_note
        sl_fact.note = (sl_fact.note or "") + overlap_note

    return facts


_RAW_VALUE_NOTE = {
    'ko': '기호화(임계값 판정) 없이 원 지표 수치를 그대로 제시함 — Stage 1 제거 조건.',
    'en': 'Shown as the raw indicator value with no symbolic thresholding applied — Stage 1 ablated.',
}

_RAW_INDICATOR_ORDER = [
    'm_overconfidence', 'm_underconfidence', 'm_strategic_help',
    'm_cognitive_avoidance', 'm_productive_struggle', 'm_unproductive_frustration',
    'm_lucky_guess', 'm_slipping', 'm_boredom_offtask',
]


def build_raw_numeric_facts(meta_vector: dict, lang: str = 'ko') -> list:
    return [
        SymbolicFact(
            indicator=_strip_m_prefix(key),
            status='RAW',
            value=float(meta_vector.get(key, 0.0)),
            note=_RAW_VALUE_NOTE[lang],
        )
        for key in _RAW_INDICATOR_ORDER
    ]


def facts_to_prompt_block(facts: list) -> str:
    lines = []
    for f in facts:
        if f.status == 'AMBIGUOUS':
            lines.append(f"- [{f.indicator}] 상태=판단불가. {f.note} "
                          f"(이 항목은 확정적으로 서술하지 말 것 — 가능성으로만 언급)")
        else:
            val_str = f" (값={f.value:.3f})" if f.value is not None else ""
            lines.append(f"- [{f.indicator}] 상태={f.status}{val_str}. {f.note}")
    return "\n".join(lines)




INDICATOR_DESCRIPTIONS_HIGH = {
    'overconfidence': "학생이 실제 자신의 이해 수준보다 더 높은 자신감을 보이며 과도하게 확신했다.",
    'underconfidence': "학생이 실제 자신의 이해 수준보다 낮은 자신감을 보이며 스스로를 과소평가했다.",
    'strategic_help': "학생이 필요한 순간에 힌트를 전략적으로 잘 활용했다.",
    'cognitive_avoidance': "학생이 어려운 문제나 도움 요청을 회피하려는 경향을 보였다.",
    'productive_struggle': "학생이 힌트를 사용하며 어려움을 겪으면서도 생산적으로 분투한 끝에 문제를 해결했다.",
    'unproductive_frustration': "학생이 반복적인 실패나 어려움 속에서 비생산적인 좌절감을 느꼈다.",
    'lucky_guess': "학생이 문제를 제대로 이해하지 못한 채 우연히 정답을 맞혔다.",
    'slipping': "학생이 이전에는 잘 이해하고 있던 내용에서 뜻밖의 실수를 저질렀다.",
    'boredom_offtask': "학생이 지루함을 느끼며 과제에서 벗어난 행동을 보였다.",
}

INDICATOR_DESCRIPTIONS_HIGH_EN = {
    'overconfidence': "The student showed excessive confidence, higher than their actual level of understanding warranted.",
    'underconfidence': "The student showed lower confidence than their actual level of understanding warranted, underestimating themselves.",
    'strategic_help': "The student made good strategic use of hints exactly when they were needed.",
    'cognitive_avoidance': "The student showed a tendency to avoid difficult problems or avoid asking for help.",
    'productive_struggle': "The student struggled with difficulty while using hints, but ultimately solved the problem through productive effort.",
    'unproductive_frustration': "The student experienced unproductive frustration amid repeated failures or difficulty.",
    'lucky_guess': "The student answered correctly by chance, without properly understanding the problem.",
    'slipping': "The student made an unexpected mistake on content they had previously understood well.",
    'boredom_offtask': "The student showed signs of boredom and off-task behavior.",
}


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


