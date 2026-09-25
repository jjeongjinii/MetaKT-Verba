
import difflib
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from stage1_grounding import (
    SymbolicFact, build_symbolic_facts, strip_generator_only_instructions,
    INDICATOR_DESCRIPTIONS_HIGH, INDICATOR_DESCRIPTIONS_HIGH_EN, _strip_m_prefix,
)



ENGLISH_SYSTEM_INSTRUCTION = """You are an AI teaching assistant explaining a student's learning-behavior log to a teacher.

Using only the facts in the [Evidence List] below, write a narrative of 2 to 4 sentences.

Absolute rules:
1. Do not add anything that is not in the [Evidence List] (theoretical terms, causal interpretations, other indicators) on your own.
2. Every sentence must end with a bracketed tag showing which piece of evidence it is based on. Example: "...shows a tendency toward this. [cognitive_avoidance]" Do NOT collect all citation tags together at the very end of the narrative — each sentence needs its own tag placed immediately after that sentence, even when multiple sentences cite the same piece of evidence. (Bad example: "Sentence one. Sentence two. Sentence three. [overconfidence] [slipping]" / Good example: "Sentence one. [overconfidence] Sentence two. [overconfidence] Sentence three. [slipping]")
3. For any item whose evidence status is "Cannot Determine (AMBIGUOUS)", never use definitive phrasing such as "was" or "this was the case" — use only hedging language such as "may have been" or "appears to be, though this is not certain."
   Do not commit to a single possibility (e.g., do not assert flatly that "it was a guess").
4. Do not mention educational-theory terms that are not in the evidence list (e.g., the Yerkes-Dodson law, Flow Theory, etc.) on your own.
5. Focus only on clearly and concretely describing the observed behavior. Never write what the teacher should do, or what is "needed" or "recommended" — the purpose of this narrative is observational description, not advice or prescription.
6. Do not write internal state labels such as 'HIGH', 'AMBIGUOUS', or 'LOW', or raw indicator variable names (e.g., cognitive_avoidance), directly in the body of a sentence. Keep such labels only inside the bracketed citation tags, and describe the observed behavior itself in natural language. (Bad example: "Both indicators show a HIGH state, and..." / Good example: "Two patterns are clearly observed together, and...")
7. The output must consist only of narrative sentences, and every sentence must still end with its bracketed citation tag from Rule 2 — that tag is required formatting and is NOT affected by the restrictions below (do not drop it for the sake of "plain" prose). Beyond that tag, never include any of the following:
   self-censoring or revision comments (e.g., "This narrative should be revised as follows", "The part above is not permitted"), action suggestions or advice/guidance/intervention phrasing directed at the teacher (e.g., entire recommendation/prescriptive sentences such as "it is necessary to...", "should be supported by...", "by providing...", "more materials are needed"), apologies or explanations, code, any language other than English (Korean, Japanese, Chinese, etc.) or Han characters, bullet markers (*, -) or numbered lists, or meta-questions ("how should this change in a different situation?"). Stop immediately once you finish the narrative — do not rewrite yourself or offer alternatives. Write the sentence content itself in ordinary English prose (and numbers/punctuation where necessary); this restriction is about the sentence content and language, not about the bracketed citation tag, which must always be present.
8. Some entries in the [Evidence List] include a bracketed note starting with "[Instruction: ...]". That bracketed text is a direction for how YOU should write (e.g., telling you not to comment on the relationship between two co-occurring indicators, and to simply describe each one independently instead) — it is not a fact about the student and must never be restated, paraphrased, or turned into a sentence of its own (e.g., never write a sentence like "these two indicators are correlated" or "this suggests a shared pattern" or "there is a consistent relationship between these behaviors"). Also, do not cite the raw decimal value (e.g., "0.840") inside a sentence as if it were evidence — translate it into the same kind of plain behavioral language used elsewhere in the narrative instead.
"""

KOREAN_SYSTEM_INSTRUCTION = """당신은 학생의 학습 행동 로그를 교사에게 설명하는 AI 보조교사입니다.

아래 [근거 목록]에 있는 사실만을 근거로, 2~4문장의 서술을 작성하세요.

절대 규칙:
1. [근거 목록]에 없는 내용(이론 용어, 원인 해석, 다른 지표)을 임의로 추가하지 마세요.
2. 각 문장은 반드시 끝에 어떤 근거를 사용했는지 대괄호로 표시하세요. 예: "...경향을 보입니다. [cognitive_avoidance]" 인용 태그를 서술 맨 끝에 한꺼번에 몰아서 쓰지 마세요 — 같은 근거를 여러 문장이 인용하더라도, 각 문장은 자기 바로 뒤에 자신의 태그를 가져야 합니다. (나쁜 예: "문장 하나. 문장 둘. 문장 셋. [overconfidence] [slipping]" / 좋은 예: "문장 하나. [overconfidence] 문장 둘. [overconfidence] 문장 셋. [slipping]")
3. 근거 상태가 '판단불가(AMBIGUOUS)'인 항목에 대해서는 "~였다", "~한 것이다" 같은 확정적 표현을 절대 쓰지 말고, "~일 수 있습니다", "~로 보이나 확실하지 않습니다" 같은 헤지(hedge) 표현만 사용하세요.
   가능성들을 하나로 단정하지 마세요 (예: "추측이었다"라고 단정 금지).
4. 근거 목록에 없는 교육이론 용어(예: Yerkes-Dodson 법칙, Flow Theory 등)를 임의로 언급하지 마세요.
5. 서술은 관찰된 행동을 명확하고 구체적으로 묘사하는 데만 집중하세요. 교사가 무엇을 해야 하는지, 무엇이 "필요"하거나 "권장"되는지는 절대 쓰지 마세요 — 이 서술의 목적은 관찰 기술(description)이지 조언이나 처방이 아닙니다.
6. [근거 목록]에 적힌 'HIGH', 'AMBIGUOUS', 'LOW' 같은 내부 상태 표기나 지표 변수명(예: cognitive_avoidance)을 문장 본문에 그대로 쓰지 마세요. 이런 라벨은 대괄호 인용 태그에만 남기고, 문장 자체는 관찰된 행동을 자연스러운 우리말로 서술하세요. (나쁜 예: "두 지표 모두 HIGH 상태를 나타내며..." / 좋은 예: "두 가지 패턴이 함께 뚜렷하게 관찰되며...")
7. 출력은 오직 서술 문장들뿐이어야 하며, 각 문장은 반드시 규칙 2에 따른 대괄호 인용 태그로 끝나야 합니다 — 이 태그는 필수 형식이며, 아래 제한사항("순수한 한글만" 등)의 적용을 받지 않습니다(형식을 지킨다고 태그까지 빼면 안 됩니다). 태그 외에 다음과 같은 내용은 절대 포함하지 마세요:
   자기 검열/수정 코멘트(예: "이 서술은 다음과 같이 수정되어야 합니다", "위 서술에서 ~ 부분은 허용되지 않습니다"), 교사에게 주는 행동 제안이나 조언·지도·개입 문구(예: "~하는 것이 필요합니다", "~을 지원해야 한다", "~을 제공하여", "더 많은 자료가 필요합니다" 같은 권고/처방형 문장 전체), 사과나 설명, 코드, 영어·일본어·중국어 등 한국어가 아닌 언어나 한자, 글머리 기호(*, -)나 번호 매기기, 메타 질문("다른 상황에서는 어떻게 바꿔야 할까요?" 등). 서술을 다 쓴 뒤에는 즉시 멈추세요 — 스스로 다시 쓰거나 대안을 제시하지 마세요. 문장 내용 자체는 순수한 한글(및 필요한 숫자/문장부호)만으로 작성하세요 — 이 제한은 문장 내용과 언어에 대한 것이지, 항상 붙어 있어야 하는 대괄호 인용 태그와는 무관합니다. 
8. [근거 목록]의 일부 항목에는 "[지시: ...]"로 시작하는 대괄호 메모가 붙어 있을 수 있습니다. 이 메모는 여러분이 '어떻게 써야 하는지'에 대한 지시(예: 함께 등장한 두 지표의 관계를 언급하지 말고 각각을 독립적으로 서술하라는 지시)이지, 학생에 대한 사실이 아닙니다. 이 지시문 자체를 그대로 옮기거나 바꿔 말해서 독립된 문장으로 만들지 마세요(나쁜 예: "이 두 지표는 서로 상관되어 있다", "이는 공통된 패턴을 시사한다", "이 두 행동 사이에는 일관된 관계가 있다" 같은 문장을 새로 쓰지 말 것). 또한 문장 안에 원시 소수값(예: "0.840")을 그대로 인용해 근거처럼 쓰지 말고, 서술의 다른 부분과 마찬가지로 평범한 행동 묘사 언어로 바꿔서 쓰세요. 
"""

SYSTEM_INSTRUCTION = KOREAN_SYSTEM_INSTRUCTION

_SYSTEM_INSTRUCTIONS = {'ko': KOREAN_SYSTEM_INSTRUCTION, 'en': ENGLISH_SYSTEM_INSTRUCTION}


KOREAN_SYSTEM_INSTRUCTION_NO_CITATION = """당신은 학생의 학습 행동 로그를 교사에게 설명하는 AI 보조교사입니다.

아래 [근거 목록]에 있는 사실만을 근거로, 2~4문장의 서술을 작성하세요.

절대 규칙:
1. [근거 목록]에 없는 내용(이론 용어, 원인 해석, 다른 지표)을 임의로 추가하지 마세요.
2. 근거 상태가 '판단불가(AMBIGUOUS)'인 항목에 대해서는 "~였다", "~한 것이다" 같은 확정적 표현을 절대 쓰지 말고, "~일 수 있습니다", "~로 보이나 확실하지 않습니다" 같은 헤지(hedge) 표현만 사용하세요. 가능성들을 하나로 단정하지 마세요 (예: "추측이었다"라고 단정 금지).
3. 근거 목록에 없는 교육이론 용어(예: Yerkes-Dodson 법칙, Flow Theory 등)를 임의로 언급하지 마세요.
4. 서술은 관찰된 행동을 명확하고 구체적으로 묘사하는 데만 집중하세요. 교사가 무엇을 해야 하는지, 무엇이 "필요"하거나 "권장"되는지는 절대 쓰지 마세요 — 이 서술의 목적은 관찰 기술(description)이지 조언이나 처방이 아닙니다.
5. [근거 목록]에 적힌 'HIGH', 'AMBIGUOUS', 'LOW' 같은 내부 상태 표기나 지표 변수명(예: cognitive_avoidance)을 문장 본문에 그대로 쓰지 마세요. 이런 라벨 없이, 관찰된 행동을 자연스러운 우리말로만 서술하세요.
6. 출력은 오직 서술 문장들뿐이어야 합니다. 대괄호 인용 태그나 출처 표시, 각주 등 어떤 형태의 참조 표기도 붙이지 말고 순수한 문장으로만 쓰세요. 그 외에 다음과 같은 내용도 절대 포함하지 마세요: 자기 검열/수정 코멘트, 교사에게 주는 행동 제안이나 조언·지도·개입 문구, 사과나 설명, 코드, 영어·일본어·중국어 등 한국어가 아닌 언어나 한자, 글머리 기호(*, -)나 번호 매기기, 메타 질문. 서술을 다 쓴 뒤에는 즉시 멈추세요.
7. [근거 목록]의 일부 항목에는 "[지시: ...]"로 시작하는 대괄호 메모가 붙어 있을 수 있습니다. 이 메모는 여러분이 '어떻게 써야 하는지'에 대한 지시이지, 학생에 대한 사실이 아닙니다. 이 지시문 자체를 그대로 옮기거나 바꿔 말해서 독립된 문장으로 만들지 마세요. 또한 문장 안에 원시 소수값(예: "0.840")을 그대로 인용해 근거처럼 쓰지 마세요.
"""

ENGLISH_SYSTEM_INSTRUCTION_NO_CITATION = """You are an AI teaching assistant explaining a student's learning-behavior log to a teacher.

Using only the facts in the [Evidence List] below, write a narrative of 2 to 4 sentences.

Absolute rules:
1. Do not add anything that is not in the [Evidence List] (theoretical terms, causal interpretations, other indicators) on your own.
2. For any item whose evidence status is "Cannot Determine (AMBIGUOUS)", never use definitive phrasing such as "was" or "this was the case" — use only hedging language such as "may have been" or "appears to be, though this is not certain." Do not commit to a single possibility (e.g., do not assert flatly that "it was a guess").
3. Do not mention educational-theory terms that are not in the evidence list (e.g., the Yerkes-Dodson law, Flow Theory, etc.) on your own.
4. Focus only on clearly and concretely describing the observed behavior. Never write what the teacher should do, or what is "needed" or "recommended" — the purpose of this narrative is observational description, not advice or prescription.
5. Do not write internal state labels such as 'HIGH', 'AMBIGUOUS', or 'LOW', or raw indicator variable names (e.g., cognitive_avoidance), directly in the body of a sentence. Describe the observed behavior itself in natural language, without such labels.
6. The output must consist only of narrative sentences. Do NOT add bracketed citation tags, source markers, footnotes, or any other kind of reference marking — write plain prose only. Also never include: self-censoring or revision comments, action suggestions or advice/guidance/intervention phrasing directed at the teacher, apologies or explanations, code, any language other than English or Han characters, bullet markers (*, -) or numbered lists, or meta-questions. Stop immediately once you finish the narrative.
7. Some entries in the [Evidence List] include a bracketed note starting with "[Instruction: ...]". That bracketed text is a direction for how YOU should write — it is not a fact about the student and must never be restated, paraphrased, or turned into a sentence of its own. Also, do not cite the raw decimal value (e.g., "0.840") inside a sentence as if it were evidence.
"""

_SYSTEM_INSTRUCTIONS_NO_CITATION = {
    'ko': KOREAN_SYSTEM_INSTRUCTION_NO_CITATION,
    'en': ENGLISH_SYSTEM_INSTRUCTION_NO_CITATION,
}


TRANSLATION_INSTRUCTION_EN_TO_KO = """You are a professional translator. Translate the given English sentence into natural, fluent Korean.

Rules:
- Preserve the meaning exactly. Do not add, remove, or soften any content.
- Do not add explanations, notes, quotation marks, or anything besides the translation itself.
- Never add a number, bullet, or list marker (e.g., "1.") before the translation.
- Never add a bracketed tag (e.g., "[overconfidence]") — the source sentence you are given will never contain one, and you must not invent one.
- Output only the single translated Korean sentence, nothing else.
"""


def _is_translation_output_suspect(ko_text: str) -> bool:
    text = ko_text.strip()
    if not text:
        return True
    if not re.search(r'[가-힣]', text):
        return True
    if is_latin_script_contaminated(text):
        return True
    return False


TRANSLATION_INSTRUCTION_EN_TO_KO = """You are a precise Korean translator working on short educational narrative sentences.

You will receive a numbered list of English sentences. Each sentence ends with one or more bracketed citation tags, e.g. [cognitive_avoidance].

Rules:
1. Translate every sentence into natural, fluent Korean. Do not add, omit, or reinterpret any content — this must be a faithful translation, not a new narrative.
2. Keep every bracketed tag EXACTLY as it appears in the English input (do not translate the tag content, do not change, merge, split, add, or drop any tag), and keep it attached at the end of its corresponding sentence.
3. Output exactly one translated line per input sentence, in the same order, using the same number prefix (e.g. "1. ..."). Do not add commentary, headers, blank lines, or extra sentences.
4. Do not include any English words in the Korean sentence body other than the bracketed tags themselves.
"""


_FACTS_BLOCK_LABELS = {
    'ko': {
        'skill': '- 스킬: {v}',
        'step': '- 시점(step): {v}',
        'context_header': '[상황 정보]',
        'evidence_header': '[근거 목록]',
        'ambiguous_line': '- [{indicator}] 상태=판단불가(AMBIGUOUS). {note}',
        'high_line': '- [{indicator}] 상태=HIGH{val_str}. {note}',
        'raw_line': '- [{indicator}] 원값={v:.3f} (0~1 범위, 임계값 판정 없음). {note}',
        'val_str': ' (값={v:.3f})',
        'no_signal': ("- (특별히 강한 신호 없음. 이 경우 '뚜렷한 특이 패턴이 관찰되지 않았다' 정도로만 서술하고, "
                       "이 문장 뒤에는 대괄호 인용 태그를 붙이지 마세요 — 인용할 구체적인 근거 항목이 없기 때문입니다. "
                       "이것은 규칙 2의 유일한 예외입니다.)"),
        'user_instruction': (
            "위 [근거 목록]만을 근거로 2~4문장의 서술을 작성하세요. "
            "서술 문장들 외에는 그 어떤 텍스트도 출력하지 마세요."
        ),
    },
    'en': {
        'skill': '- Skill: {v}',
        'step': '- Step: {v}',
        'context_header': '[Context]',
        'evidence_header': '[Evidence List]',
        'ambiguous_line': '- [{indicator}] status=Cannot Determine (AMBIGUOUS). {note}',
        'high_line': '- [{indicator}] status=HIGH{val_str}. {note}',
        'raw_line': '- [{indicator}] raw value={v:.3f} (range 0-1, no thresholding applied). {note}',
        'val_str': ' (value={v:.3f})',
        'no_signal': ("- (No particularly strong signal observed. In this case, write only something "
                       "like 'no clear unusual pattern was observed' — and do NOT add a bracketed "
                       "citation tag after this sentence, since there is no specific evidence item to "
                       "cite. This is the one exception to Rule 2.)"),
        'user_instruction': (
            "Based only on the [Evidence List] above, write a narrative of 2 to 4 sentences. "
            "Output nothing except the narrative sentences themselves."
        ),
    },
}


def _build_facts_block(facts: list, context: Optional[dict] = None, language: str = 'ko') -> str:
    labels = _FACTS_BLOCK_LABELS[language]
    context = context or {}
    ctx_lines = []
    if 'skill' in context:
        ctx_lines.append(labels['skill'].format(v=context['skill']))
    if 'step' in context:
        ctx_lines.append(labels['step'].format(v=context['step']))
    ctx_str = (labels['context_header'] + "\n" + "\n".join(ctx_lines) + "\n\n") if ctx_lines else ""

    fact_lines = []
    for f in facts:
        if f.status == 'AMBIGUOUS':
            fact_lines.append(labels['ambiguous_line'].format(indicator=f.indicator, note=f.note))
        elif f.status == 'HIGH':
            val_str = labels['val_str'].format(v=f.value) if f.value is not None else ""
            fact_lines.append(labels['high_line'].format(indicator=f.indicator, val_str=val_str, note=f.note))
        elif f.status == 'RAW':
            fact_lines.append(labels['raw_line'].format(indicator=f.indicator, v=f.value, note=f.note))

    if not fact_lines:
        fact_lines.append(labels['no_signal'])

    facts_block = labels['evidence_header'] + "\n" + "\n".join(fact_lines)
    return f"{ctx_str}{facts_block}"


def build_messages(facts: list, context: Optional[dict] = None, language: str = 'ko',
                    require_citation: bool = True) -> list:
    if language not in _SYSTEM_INSTRUCTIONS:
        raise ValueError(f"Unsupported language='{language}'; expected 'ko' or 'en'.")
    system_instructions = _SYSTEM_INSTRUCTIONS if require_citation else _SYSTEM_INSTRUCTIONS_NO_CITATION
    facts_block = _build_facts_block(facts, context, language=language)
    user_content = f"{facts_block}\n\n{_FACTS_BLOCK_LABELS[language]['user_instruction']}"
    return [
        {"role": "system", "content": system_instructions[language]},
        {"role": "user", "content": user_content},
    ]


def build_prompt(facts: list, context: Optional[dict] = None, language: str = 'ko',
                  require_citation: bool = True) -> str:
    messages = build_messages(facts, context, language=language, require_citation=require_citation)
    parts = [f"[{m['role'].upper()}]\n{m['content']}" for m in messages]
    return "\n\n".join(parts)


_NAIVE_SYSTEM_INSTRUCTION = {
    'ko': ("당신은 학생의 학습 데이터를 교사에게 설명하는 AI 보조교사입니다. "
           "아래 학생의 행동 관련 수치를 참고하여, 학생의 학습 행동에 대한 2~4문장의 "
           "자연스러운 서술을 작성하세요."),
    'en': ("You are an AI teaching assistant explaining a student's learning data to a "
           "teacher. Using the numeric values about the student's behavior below, write "
           "a natural narrative of 2 to 4 sentences describing the student's learning "
           "behavior."),
}

_NAIVE_VALUE_LABELS = {
    'ko': {
        'header': '[학생 행동 지표 (원값, 0~1 범위)]',
        'skill': '- 스킬: {v}', 'step': '- 시점(step): {v}', 'hint': '- 힌트 사용 횟수: {v}',
    },
    'en': {
        'header': '[Student Behavior Indicators (raw values, range 0-1)]',
        'skill': '- Skill: {v}', 'step': '- Step: {v}', 'hint': '- Hint count: {v}',
    },
}

_NAIVE_INDICATOR_ORDER = [
    'm_overconfidence', 'm_underconfidence', 'm_strategic_help',
    'm_cognitive_avoidance', 'm_productive_struggle', 'm_unproductive_frustration',
    'm_lucky_guess', 'm_slipping', 'm_boredom_offtask',
]


def build_naive_messages(meta_vector: dict, hint_count: float,
                          context: Optional[dict] = None, language: str = 'ko') -> list:
    if language not in _NAIVE_SYSTEM_INSTRUCTION:
        raise ValueError(f"Unsupported language='{language}'; expected 'ko' or 'en'.")
    labels = _NAIVE_VALUE_LABELS[language]
    context = context or {}
    lines = [labels['header']]
    for key in _NAIVE_INDICATOR_ORDER:
        lines.append(f"- {key}: {float(meta_vector.get(key, 0.0)):.3f}")
    lines.append(labels['hint'].format(v=hint_count))
    if 'skill' in context:
        lines.append(labels['skill'].format(v=context['skill']))
    if 'step' in context:
        lines.append(labels['step'].format(v=context['step']))
    return [
        {"role": "system", "content": _NAIVE_SYSTEM_INSTRUCTION[language]},
        {"role": "user", "content": "\n".join(lines)},
    ]


@dataclass
class CitedSentence:
    text: str
    cited_indicators: list


_DECIMAL_POINT_PATTERN = re.compile(r'(?<=\d)\.(?=\d)')
_DECIMAL_PLACEHOLDER = '\uE000'


def parse_generated_narrative(raw_text: str) -> list:
    raw_text = raw_text.strip()
    protected_text = _DECIMAL_POINT_PATTERN.sub(_DECIMAL_PLACEHOLDER, raw_text)
    pattern = re.compile(r'([^.!?]*[.!?])\s*((?:\[[^\]]+\]\s*)*)')
    results = []
    for match in pattern.finditer(protected_text):
        sentence_part = match.group(1).strip().replace(_DECIMAL_PLACEHOLDER, '.')
        tags_part = (match.group(2) or "").replace(_DECIMAL_PLACEHOLDER, '.')
        if not sentence_part:
            continue
        tags = re.findall(r'\[([^\]]+)\]', tags_part)
        inline_tags = re.findall(r'\[([^\]]+)\]', sentence_part)
        clean_sentence = re.sub(r'\s*\[[^\]]+\]\s*', ' ', sentence_part).strip()
        all_tags = list(dict.fromkeys(tags + inline_tags))
        results.append(CitedSentence(text=clean_sentence, cited_indicators=all_tags))
    return results


def parse_numbered_translation(raw_text: str) -> list:
    lines = [ln.strip() for ln in raw_text.strip().split('\n') if ln.strip()]
    results = []
    for ln in lines:
        ln = re.sub(r'^\s*\d+[\.\)]\s*', '', ln)
        tags = re.findall(r'\[([^\]]+)\]', ln)
        clean = re.sub(r'\s*\[[^\]]+\]\s*', ' ', ln).strip()
        if clean:
            results.append(CitedSentence(text=clean, cited_indicators=tags))
    return results


_LEAKED_FACT_LINE_PATTERNS_KO = [
    re.compile(r'HIGH|LOW|AMBIGUOUS'),
    re.compile(r'\(\s*값\s*='),
    re.compile(r'판단불가\s*\(\s*AMBIGUOUS\s*\)'),
    re.compile(r'\[\s*지시\s*[:：]'),
    re.compile(r'규칙\s*\d'),
    re.compile(r'근거\s*목록'),
    re.compile(r'내부\s*상태\s*표기'),
    re.compile(r'지표\s*변수명'),
    re.compile(r'대괄호로?\s*(표시|인용)'),
    re.compile(r'헤지\s*\(\s*hedge\s*\)'),
    re.compile(r'교육이론\s*용어'),
    re.compile(r'Yerkes-Dodson|Flow Theory'),
]

_LEAKED_FACT_LINE_PATTERNS_EN = [
    re.compile(r'\bHIGH\b|\bLOW\b|\bAMBIGUOUS\b'),
    re.compile(r'\(\s*value\s*='),
    re.compile(r'Cannot Determine\s*\(\s*AMBIGUOUS\s*\)'),
    re.compile(r'\[\s*Instruction\s*[:：]'),
    re.compile(r'\bRule\s*\d'),
    re.compile(r'Evidence\s*List'),
    re.compile(r'internal\s*state\s*label'),
    re.compile(r'indicator\s*variable\s*name'),
    re.compile(r'bracket(ed)?\s*(citation\s*)?tag'),
    re.compile(r'hedg(e|ing)\s*(language|phrasing)?'),
    re.compile(r'educational[- ]theory\s*term'),
    re.compile(r'Yerkes-Dodson|Flow Theory'),
]

_LEAKED_FACT_LINE_PATTERNS = {'ko': _LEAKED_FACT_LINE_PATTERNS_KO, 'en': _LEAKED_FACT_LINE_PATTERNS_EN}

_VARIABLE_NAME_PATTERN = re.compile(
    r'\(\s*(overconfidence|underconfidence|strategic_help|cognitive_avoidance|'
    r'productive_struggle|unproductive_frustration|lucky_guess|slipping|boredom_offtask)\s*\)'
)


_ECHOED_BULLET_PATTERNS = {
    'ko': re.compile(r'^\s*-\s*\[[^\]]+\]\s*상태\s*='),
    'en': re.compile(r'^\s*-\s*\[[^\]]+\]\s*status\s*='),
}


def _looks_like_echoed_bullet_line(line: str, language: str = 'ko') -> bool:
    return bool(_ECHOED_BULLET_PATTERNS[language].match(line))


def is_leaked_fact_line(text: str, language: str = 'ko') -> bool:
    return any(p.search(text) for p in _LEAKED_FACT_LINE_PATTERNS[language])


def strip_leaked_fact_block(raw_text: str, language: str = 'ko') -> str:
    lines = raw_text.split('\n')
    kept = [ln for ln in lines if not _looks_like_echoed_bullet_line(ln, language=language)]
    return '\n'.join(kept)


def is_code_artifact(text: str) -> bool:
    return bool(re.search(r'```|^\s*import\s+\w|\bdef\s+\w+\s*\(', text))


_FOREIGN_SCRIPT_PATTERN = re.compile(
    r'[぀-ゟ゠-ヿ一-鿿]'
)


def is_foreign_script_contaminated(text: str) -> bool:
    return bool(_FOREIGN_SCRIPT_PATTERN.search(text))


_LATIN_SCRIPT_PATTERN = re.compile(r'[A-Za-zÀ-ɏ]{2,}')


def is_latin_script_contaminated(text: str) -> bool:
    return bool(_LATIN_SCRIPT_PATTERN.search(text))


_HANGUL_SCRIPT_PATTERN = re.compile(r'[가-힣]')


def is_hangul_contaminated(text: str) -> bool:
    return bool(_HANGUL_SCRIPT_PATTERN.search(text))


_HINT_TYPO_PATTERN = re.compile(r'힐프|힐티|힐난|힐끔|힐링')


def is_hint_typo_severely_corrupted(text: str) -> bool:
    return len(_HINT_TYPO_PATTERN.findall(text)) >= 2


def fix_hint_typo(text: str) -> str:
    return _HINT_TYPO_PATTERN.sub('힌트', text)


_RECOMMENDATION_PATTERN_KO = re.compile(
    r'(필요합니다|필요하다|필요해\s|권장|지도가|지도를|도와줄\s*수\s*있을|관리하여|관리해야|'
    r'개입하여|개입해야|가이드|피드백을\s*주|격려|지원해야|지원이\s*필요|제공하여|제공하는\s*것이|'
    r'확인하는\s*것이\s*필요|확보하는\s*것이\s*필요)'
)

_RECOMMENDATION_PATTERN_EN = re.compile(
    r'\b(should|needs? to|must|it (is|would be) (necessary|advisable|recommended|beneficial)|'
    r'is recommended|requires? (more|additional|further)|consider (providing|offering|giving)|'
    r'would benefit from|needs? (support|guidance|intervention|encouragement|feedback)|'
    r'provide (more|additional)|ought to)\b',
    re.IGNORECASE,
)

_RECOMMENDATION_PATTERNS = {'ko': _RECOMMENDATION_PATTERN_KO, 'en': _RECOMMENDATION_PATTERN_EN}


def is_recommendation_sentence(text: str, language: str = 'ko') -> bool:
    return bool(_RECOMMENDATION_PATTERNS[language].search(text))


def sanitize_sentences(sentences: list, language: str = 'ko') -> list:
    script_contaminated = (
        (lambda t: is_latin_script_contaminated(t)) if language == 'ko'
        else (lambda t: is_hangul_contaminated(t))
    )
    cleaned = []
    for s in sentences:
        if (is_leaked_fact_line(s.text, language=language) or is_code_artifact(s.text)
                or is_recommendation_sentence(s.text, language=language)
                or is_foreign_script_contaminated(s.text) or script_contaminated(s.text)
                or (language == 'ko' and is_hint_typo_severely_corrupted(s.text))):
            continue
        text = _VARIABLE_NAME_PATTERN.sub('', s.text)
        if language == 'ko':
            text = fix_hint_typo(text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        if not text:
            continue
        cleaned.append(CitedSentence(text=text, cited_indicators=s.cited_indicators))
    return cleaned


def _char_ngrams(text: str, n: int = 2) -> set:
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _text_similarity(a: str, b: str) -> float:
    na, nb = _char_ngrams(a), _char_ngrams(b)
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def dedupe_and_flag_repetition(sentences: list, similarity_threshold: float = 0.6):
    kept = []
    kept_norm = []
    had_repetition = False
    for s in sentences:
        text = s.text.strip()
        if not text:
            continue
        norm = re.sub(r'[\s,.!?"\'~()\[\]]+', '', text)
        is_dup = False
        for prev_s, prev_norm in zip(kept, kept_norm):
            if text == prev_s.text.strip():
                is_dup = True
                break
            if set(s.cited_indicators) & set(prev_s.cited_indicators) and \
                    _text_similarity(norm, prev_norm) >= similarity_threshold:
                is_dup = True
                break
        if is_dup:
            had_repetition = True
            continue
        kept.append(s)
        kept_norm.append(norm)
    return kept, had_repetition


_CITATION_TYPO_CUTOFF = 0.75


def correct_citation_typos(sentences: list, valid_indicators: set) -> list:
    if not valid_indicators:
        return sentences
    valid_lower_to_canonical = {v.lower(): v for v in valid_indicators}
    corrected = []
    for s in sentences:
        new_tags = []
        for raw_tag in s.cited_indicators:
            parts = [p.strip() for p in re.split(r'[,，、]', raw_tag) if p.strip()]
            for part in parts:
                if part in valid_indicators:
                    new_tags.append(part)
                    continue
                match = difflib.get_close_matches(
                    part.lower(), list(valid_lower_to_canonical.keys()),
                    n=1, cutoff=_CITATION_TYPO_CUTOFF
                )
                new_tags.append(valid_lower_to_canonical[match[0]] if match else part)
        corrected.append(CitedSentence(
            text=s.text, cited_indicators=list(dict.fromkeys(new_tags))
        ))
    return corrected


def validate_citations(sentences: list, valid_indicators: set) -> dict:
    n_total = len(sentences)
    n_uncited = sum(1 for s in sentences if not s.cited_indicators)
    n_invalid_ref = sum(
        1 for s in sentences
        if s.cited_indicators and not set(s.cited_indicators).issubset(valid_indicators)
    )
    return {
        'n_sentences': n_total,
        'n_uncited': n_uncited,
        'n_invalid_reference': n_invalid_ref,
        'structural_violation_rate': (n_uncited + n_invalid_ref) / n_total if n_total else None,
    }


_THINK_BLOCK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
_UNCLOSED_THINK_PATTERN = re.compile(r'<think>')


def _strip_thinking_block(raw_text: str) -> tuple:
    if _UNCLOSED_THINK_PATTERN.search(raw_text) and '</think>' not in raw_text:
        return "", True
    cleaned = _THINK_BLOCK_PATTERN.sub('', raw_text).strip()
    return cleaned, False


class NarrativeGenerator:
    def __init__(self, model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
                 chat_template_kwargs: Optional[dict] = None,
                 trust_remote_code: bool = False):
        self.model_name = model_name
        self.chat_template_kwargs = (
            {'enable_thinking': False} if chat_template_kwargs is None else chat_template_kwargs
        )
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        else:
            print("Note: HF_TOKEN is not set. Ungated models can proceed normally; "
                  "gated models such as Llama may fail authentication below.")

        print(f"Loading {self.model_name}... (trust_remote_code={self.trust_remote_code})")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=hf_token, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float16, device_map="auto", token=hf_token,
            trust_remote_code=self.trust_remote_code,
        )
        self._model.eval()

    def generate(self, facts: list, context: Optional[dict] = None,
                 max_new_tokens: int = 200, language: str = 'ko',
                 require_citation: bool = True, seed: Optional[int] = None) -> dict:
        self._lazy_load()
        import torch
        if seed is not None:
            torch.manual_seed(seed)

        messages = build_messages(facts, context, language=language, require_citation=require_citation)
        encoded = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            **self.chat_template_kwargs,
        ).to(self._model.device)
        input_ids = encoded["input_ids"]

        eos_ids = [self._tokenizer.eos_token_id]
        eot_id = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self._tokenizer.unk_token_id:
            eos_ids.append(eot_id)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=8,
                eos_token_id=eos_ids,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        raw_output = self._tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        raw_output, truncated_in_thinking = _strip_thinking_block(raw_output)

        if truncated_in_thinking:
            return {
                'raw_text': '',
                'messages': messages,
                'language': language,
                'sentences': [],
                'citation_check': validate_citations([], {f.indicator for f in facts}),
                'had_repetition': False,
                'truncated_in_thinking': True,
            }

        sentences = parse_generated_narrative(strip_leaked_fact_block(raw_output, language=language))
        sentences = sanitize_sentences(sentences, language=language)
        sentences, had_repetition = dedupe_and_flag_repetition(sentences)
        valid_indicators = {f.indicator for f in facts}
        sentences = correct_citation_typos(sentences, valid_indicators)
        citation_check = validate_citations(sentences, valid_indicators)

        return {
            'raw_text': raw_output,
            'messages': messages,
            'language': language,
            'sentences': sentences,
            'citation_check': citation_check,
            'had_repetition': had_repetition,
            'truncated_in_thinking': False,
        }

    def generate_naive(self, meta_vector: dict, hint_count: float,
                        context: Optional[dict] = None, max_new_tokens: int = 200,
                        language: str = 'ko', seed: Optional[int] = None) -> dict:
        self._lazy_load()
        import torch
        if seed is not None:
            torch.manual_seed(seed)

        messages = build_naive_messages(meta_vector, hint_count, context, language=language)
        encoded = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            **self.chat_template_kwargs,
        ).to(self._model.device)
        input_ids = encoded["input_ids"]

        eos_ids = [self._tokenizer.eos_token_id]
        eot_id = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self._tokenizer.unk_token_id:
            eos_ids.append(eot_id)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=8,
                eos_token_id=eos_ids,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        raw_output = self._tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        raw_output, truncated_in_thinking = _strip_thinking_block(raw_output)

        valid_indicators = {_strip_m_prefix(k) for k in meta_vector.keys()}

        if truncated_in_thinking:
            return {
                'raw_text': '', 'messages': messages, 'language': language,
                'sentences': [], 'citation_check': validate_citations([], valid_indicators),
                'had_repetition': False, 'truncated_in_thinking': True,
            }

        sentences = parse_generated_narrative(raw_output)
        sentences, had_repetition = dedupe_and_flag_repetition(sentences)
        citation_check = validate_citations(sentences, valid_indicators)

        return {
            'raw_text': raw_output, 'messages': messages, 'language': language,
            'sentences': sentences, 'citation_check': citation_check,
            'had_repetition': had_repetition, 'truncated_in_thinking': False,
        }

    def translate_to_korean(self, sentences: list, max_new_tokens: int = 150) -> list:
        self._lazy_load()
        import torch

        results = []
        for s in sentences:
            messages = [
                {"role": "system", "content": TRANSLATION_INSTRUCTION_EN_TO_KO},
                {"role": "user", "content": s.text},
            ]
            encoded = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                **self.chat_template_kwargs,
            ).to(self._model.device)
            input_ids = encoded["input_ids"]

            eos_ids = [self._tokenizer.eos_token_id]
            eot_id = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and eot_id != self._tokenizer.unk_token_id:
                eos_ids.append(eot_id)

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=encoded.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=eos_ids,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            raw_ko = self._tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            raw_ko, ko_truncated_in_thinking = _strip_thinking_block(raw_ko)
            raw_ko = raw_ko.strip().strip('"').strip("'").strip()

            display_ko = re.sub(r'^\s*\d+[.).\s]+', '', raw_ko)
            display_ko = re.sub(r'\s*\[[^\]]*\]\s*', ' ', display_ko).strip()

            ok = (not ko_truncated_in_thinking) and (not _is_translation_output_suspect(display_ko))
            results.append({
                'en': s.text,
                'ko': display_ko if ok else None,
                'cited_indicators': s.cited_indicators,
                'translation_ok': ok,
                'raw_ko': raw_ko,
            })
        return results


_TEMPLATE_NO_SIGNAL_SENTENCE = {
    'ko': "특별히 강한 신호가 관찰되지 않았다.",
    'en': "No particularly strong signal was observed.",
}


def generate_template_only_narrative(facts: list, language: str = 'ko') -> dict:
    descriptions = INDICATOR_DESCRIPTIONS_HIGH if language == 'ko' else INDICATOR_DESCRIPTIONS_HIGH_EN
    citable = [f for f in facts if f.status in ('HIGH', 'AMBIGUOUS')]
    sentences = []

    if not citable:
        sentences.append(CitedSentence(text=_TEMPLATE_NO_SIGNAL_SENTENCE[language], cited_indicators=[]))
    else:
        for f in citable:
            if f.status == 'AMBIGUOUS':
                text = strip_generator_only_instructions(f.note) if f.note else (
                    "판단이 불가능한 신호가 관찰되었다." if language == 'ko'
                    else "An indeterminate signal was observed.")
            else:
                text = descriptions.get(
                    f.indicator,
                    (f"{f.indicator} 신호가 강하게 관찰되었다." if language == 'ko'
                     else f"A strong {f.indicator} signal was observed."),
                )
            sentences.append(CitedSentence(text=text, cited_indicators=[f.indicator]))

    raw_text = " ".join(
        f"{s.text} [{','.join(s.cited_indicators)}]" if s.cited_indicators else s.text
        for s in sentences
    )
    valid_indicators = {f.indicator for f in facts}
    citation_check = validate_citations(sentences, valid_indicators)

    return {
        'raw_text': raw_text,
        'messages': None,
        'language': language,
        'sentences': sentences,
        'citation_check': citation_check,
        'had_repetition': False,
        'truncated_in_thinking': False,
    }


def _dry_run():
    demo_vector = {
        'm_overconfidence': 0.0, 'm_underconfidence': 0.7,
        'm_strategic_help': 0.0, 'm_cognitive_avoidance': 0.6,
        'm_productive_struggle': 0.1, 'm_unproductive_frustration': 0.0,
        'm_lucky_guess': 0.7, 'm_slipping': 0.0, 'm_boredom_offtask': 0.0,
    }

    print("=" * 20, "KOREAN (language='ko')", "=" * 20)
    facts_ko = build_symbolic_facts(demo_vector, hint_count=0.0, sc=0.3, lang='ko')
    print(build_prompt(facts_ko, context={'skill': 'fraction addition', 'step': 33}, language='ko'))

    print("\n" + "=" * 20, "ENGLISH (language='en')", "=" * 20)
    facts_en = build_symbolic_facts(demo_vector, hint_count=0.0, sc=0.3, lang='en')
    print(build_prompt(facts_en, context={'skill': 'Fraction Addition', 'step': 33}, language='en'))

    print("\n" + "=" * 20, "ablate_citation (require_citation=False)", "=" * 20)
    print(build_prompt(facts_ko, context={'skill': 'fraction addition', 'step': 33},
                        language='ko', require_citation=False))

    print("\n" + "=" * 20, "ablate_stage1 (RAW facts, without symbolic grounding)", "=" * 20)
    from stage1_grounding import build_raw_numeric_facts
    raw_facts = build_raw_numeric_facts(demo_vector, lang='ko')
    print(build_prompt(raw_facts, context={'skill': 'fraction addition', 'step': 33}, language='ko'))

    print("\n" + "=" * 20, "naive (Naive Prompting baseline)", "=" * 20)
    naive_msgs = build_naive_messages(demo_vector, hint_count=0.0,
                                       context={'skill': 'fraction addition', 'step': 33}, language='ko')
    print("\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in naive_msgs))

    print("\n" + "=" * 20, "template (Template-only baseline (no LLM))", "=" * 20)
    template_result = generate_template_only_narrative(facts_ko, language='ko')
    print(template_result['raw_text'])


if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv or len(sys.argv) == 1:
        _dry_run()
    else:
        print("For full generation, import and use the NarrativeGenerator class directly.")
