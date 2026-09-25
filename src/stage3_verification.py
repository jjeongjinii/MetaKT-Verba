"""Stage 3: automatic faithfulness verification for MetaKT-Verba.

Each Stage-2 sentence is checked against the Stage-1 fact(s) it cites. NLI
entailment provides the content score; sentences citing ``AMBIGUOUS`` facts are
also checked for overly definitive wording.

The default checkpoint is ``MoritzLaurer/mDeBERTa-v3-base-mnli-xnli``. Class
indices are resolved from ``model.config.id2label`` rather than hard-coded.
"""

import re
from dataclasses import dataclass
from typing import Optional

from stage1_grounding import (
    SymbolicFact, INDICATOR_DESCRIPTIONS_HIGH, INDICATOR_DESCRIPTIONS_HIGH_EN,
    strip_generator_only_instructions,
)

DEFINITIVE_PATTERNS_KO = [
    r'것이다\.?$', r'였다\.?$', r'했다\.?$',
    r'확실히', r'분명히', r'명백히', r'틀림없이',
    r'임이? 분명', r'것이 확실',
]
HEDGE_PATTERNS_KO = [
    r'수 있(다|습니다)', r'로 보(인다|입니다)', r'가능성이 있',
    r'것으로 (추정|보임|판단)', r'인 듯', r'수도 있',
    r'단정하기 어렵', r'확실하지 않',
]

DEFINITIVE_PATTERNS_EN = [
    r'\bdefinitely\b', r'\bcertainly\b', r'\bclearly\b', r'\bobviously\b',
    r'\bundoubtedly\b', r'\bwithout (a )?doubt\b', r'\bsurely\b',
    r'\bwas (a|an)\b', r'\bwas the case\b',
]
HEDGE_PATTERNS_EN = [
    r'\bmay (have )?\b', r'\bmight (have )?\b', r'\bcould (have )?\b',
    r'\bpossibly\b', r'\bperhaps\b', r'\bappears? to\b', r'\bseems? to\b',
    r'\bis (unclear|uncertain)\b', r'\bdifficult to (say|determine)\b',
    r'\bnot certain\b', r'\blikely\b',
]

DEFINITIVE_PATTERNS = {'ko': DEFINITIVE_PATTERNS_KO, 'en': DEFINITIVE_PATTERNS_EN}
HEDGE_PATTERNS = {'ko': HEDGE_PATTERNS_KO, 'en': HEDGE_PATTERNS_EN}

def check_hedge_violation(sentence_text: str, lang: str = 'ko') -> bool:
    if lang not in DEFINITIVE_PATTERNS:
        raise ValueError(f"Unsupported language: {lang!r}; expected 'ko' or 'en'.")
    has_definitive = any(re.search(p, sentence_text, re.IGNORECASE) for p in DEFINITIVE_PATTERNS[lang])
    has_hedge = any(re.search(p, sentence_text, re.IGNORECASE) for p in HEDGE_PATTERNS[lang])
    return has_definitive and not has_hedge

@dataclass
class VerificationResult:
    sentence_text: str
    cited_indicators: list
    status: str
    nli_scores: Optional[dict] = None
    detail: str = ""

def _fact_to_premise_text(fact: SymbolicFact, lang: str = 'ko') -> str:
    descriptions = INDICATOR_DESCRIPTIONS_HIGH if lang == 'ko' else INDICATOR_DESCRIPTIONS_HIGH_EN
    if fact.status == 'HIGH' and fact.indicator in descriptions:
        base = descriptions[fact.indicator]
    elif fact.status == 'RAW' and fact.value is not None:
        base = (f"이 지표({fact.indicator})의 원값은 {fact.value:.3f}이다 (0~1 범위, 임계값 판정 없음)."
                if lang == 'ko' else
                f"The raw value of this indicator ({fact.indicator}) is {fact.value:.3f} "
                f"(range 0-1, no thresholding applied).")
    else:
        base = (f"이 지표({fact.indicator})의 상태는 {fact.status}이다." if lang == 'ko'
                else f"The status of this indicator ({fact.indicator}) is {fact.status}.")
    if fact.note:
        base += f" {strip_generator_only_instructions(fact.note)}"
    return base

class NLIVerifier:

    def __init__(self, model_name='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli',
                 entailment_threshold: float = 0.5):
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self._model = None
        self._tokenizer = None
        self._label_names = None
        self._device = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        print(f"Loading NLI model: {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

        id2label = getattr(self._model.config, "id2label", None) or {}
        normalized = {}
        for idx, label in id2label.items():
            key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
            if "entail" in key:
                normalized[int(idx)] = "entailment"
            elif "neutral" in key:
                normalized[int(idx)] = "neutral"
            elif "contrad" in key:
                normalized[int(idx)] = "contradiction"

        if "entailment" not in normalized.values():
            raise ValueError(
                f"{self.model_name}: could not identify an entailment class from "
                f"model.config.id2label={id2label!r}. Check the checkpoint label mapping."
            )
        n_labels = int(getattr(self._model.config, "num_labels", len(id2label)))
        self._label_names = [normalized.get(i, str(id2label.get(i, f"label_{i}")).lower())
                             for i in range(n_labels)]

    def score(self, premise: str, hypothesis: str) -> dict:
        self._lazy_load()
        import torch

        inputs = self._tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits[0], dim=-1).cpu().tolist()
        return dict(zip(self._label_names, probs))

    def verify_narrative(self, sentences: list, facts: list, lang: str = 'ko') -> list:
        facts_by_indicator = {f.indicator: f for f in facts}
        valid_indicators = set(facts_by_indicator.keys())
        no_citable_evidence = not any(f.status in ('HIGH', 'AMBIGUOUS') for f in facts)
        results = []

        for s in sentences:
            if not s.cited_indicators:
                if no_citable_evidence:
                    results.append(VerificationResult(
                        sentence_text=s.text, cited_indicators=[], status='PASS_NO_SIGNAL',
                        detail='No citable HIGH/AMBIGUOUS facts are available; no-signal case.'
                    ))
                else:
                    results.append(VerificationResult(
                        sentence_text=s.text, cited_indicators=[], status='FAIL_NO_CITATION',
                        detail='Missing citation tag; violates the Stage-2 citation requirement.'
                    ))
                continue

            invalid_refs = set(s.cited_indicators) - valid_indicators
            if invalid_refs:
                results.append(VerificationResult(
                    sentence_text=s.text, cited_indicators=s.cited_indicators, status='FAIL_INVALID_REF',
                    detail=f'Citation refers to indicator(s) absent from Stage 1: {invalid_refs}.'
                ))
                continue

            worst_scores = None
            hedge_check_needed = False
            for ind in s.cited_indicators:
                fact = facts_by_indicator[ind]
                if fact.status == 'AMBIGUOUS':
                    hedge_check_needed = True
                premise = _fact_to_premise_text(fact, lang=lang)
                scores = self.score(premise, s.text)
                if worst_scores is None or scores['entailment'] < worst_scores['entailment']:
                    worst_scores = scores

            if hedge_check_needed and check_hedge_violation(s.text, lang=lang):
                results.append(VerificationResult(
                    sentence_text=s.text, cited_indicators=s.cited_indicators, status='FAIL_HEDGE',
                    nli_scores=worst_scores,
                    detail='Definitive wording used for an AMBIGUOUS fact; violates the hedging requirement.'
                ))
                continue

            if worst_scores['entailment'] >= self.entailment_threshold:
                results.append(VerificationResult(
                    sentence_text=s.text, cited_indicators=s.cited_indicators, status='PASS',
                    nli_scores=worst_scores,
                ))
            else:
                results.append(VerificationResult(
                    sentence_text=s.text, cited_indicators=s.cited_indicators, status='FAIL_NLI',
                    nli_scores=worst_scores,
                    detail=f"entailment={worst_scores['entailment']:.3f} < threshold={self.entailment_threshold}"
                ))

        return results

    def verify_narrative_uncited(self, sentences: list, facts: list, lang: str = 'ko') -> list:
        citable_facts = [f for f in facts if f.status in ('HIGH', 'AMBIGUOUS', 'RAW')]
        no_citable_evidence = not citable_facts
        results = []

        for s in sentences:
            if no_citable_evidence:
                results.append(VerificationResult(
                    sentence_text=s.text, cited_indicators=[], status='PASS_NO_SIGNAL',
                    detail='No citable HIGH/AMBIGUOUS/RAW facts are available; no-signal case.'
                ))
                continue

            best_fact, best_scores = None, None
            for f in citable_facts:
                premise = _fact_to_premise_text(f, lang=lang)
                scores = self.score(premise, s.text)
                if best_scores is None or scores['entailment'] > best_scores['entailment']:
                    best_scores, best_fact = scores, f

            if best_fact.status == 'AMBIGUOUS' and check_hedge_violation(s.text, lang=lang):
                results.append(VerificationResult(
                    sentence_text=s.text, cited_indicators=[best_fact.indicator], status='FAIL_HEDGE',
                    nli_scores=best_scores,
                    detail='Best-matching fact is AMBIGUOUS but the sentence uses definitive wording.'
                ))
            elif best_scores['entailment'] >= self.entailment_threshold:
                results.append(VerificationResult(
                    sentence_text=s.text, cited_indicators=[best_fact.indicator], status='PASS_INFERRED',
                    nli_scores=best_scores,
                    detail=f'Citation unavailable or unreliable; inferred from best-matching fact [{best_fact.indicator}].'
                ))
            else:
                results.append(VerificationResult(
                    sentence_text=s.text, cited_indicators=[best_fact.indicator], status='FAIL_NLI_UNCITED',
                    nli_scores=best_scores,
                    detail=(f'entailment={best_scores["entailment"]:.3f} < threshold={self.entailment_threshold}; '
                            f'best-matching fact [{best_fact.indicator}] does not meet the entailment threshold.')
                ))
        return results

def summarize_results(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {'n_sentences': 0}
    from collections import Counter
    counts = Counter(r.status for r in results)

    n_pass = counts.get('PASS', 0) + counts.get('PASS_INFERRED', 0)
    n_pass_no_signal = counts.get('PASS_NO_SIGNAL', 0)
    n_fail_nli = counts.get('FAIL_NLI', 0) + counts.get('FAIL_NLI_UNCITED', 0)
    n_fail_hedge = counts.get('FAIL_HEDGE', 0)
    n_fail_nocit = counts.get('FAIL_NO_CITATION', 0)
    n_fail_invref = counts.get('FAIL_INVALID_REF', 0)

    n_content_eligible = n - n_fail_nocit
    conditional_faithfulness_rate = (
        (n_pass + n_pass_no_signal) / n_content_eligible if n_content_eligible > 0 else None
    )
    conditional_hallucination_rate = (
        (n_fail_nli + n_fail_invref) / n_content_eligible if n_content_eligible > 0 else None
    )

    return {
        'n_sentences': n,
        'faithfulness_rate': (n_pass + n_pass_no_signal) / n,
        'hallucination_rate': (n_fail_nli + n_fail_invref) / n,
        'hedge_violation_rate': n_fail_hedge / n,
        'structural_violation_rate': n_fail_nocit / n,
        'conditional_faithfulness_rate': conditional_faithfulness_rate,
        'conditional_hallucination_rate': conditional_hallucination_rate,
        'n_content_eligible': n_content_eligible,
        'breakdown': {
            'PASS': counts.get('PASS', 0), 'PASS_NO_SIGNAL': n_pass_no_signal,
            'FAIL_NLI': counts.get('FAIL_NLI', 0), 'FAIL_HEDGE': n_fail_hedge,
            'FAIL_NO_CITATION': n_fail_nocit, 'FAIL_INVALID_REF': n_fail_invref,
            **({'PASS_INFERRED': counts['PASS_INFERRED']} if 'PASS_INFERRED' in counts else {}),
            **({'FAIL_NLI_UNCITED': counts['FAIL_NLI_UNCITED']} if 'FAIL_NLI_UNCITED' in counts else {}),
        }
    }

def threshold_sensitivity_report(verification_dicts: list, thresholds=(0.5, 0.4, 0.3, 0.2, 0.1)) -> dict:
    scored = [v for v in verification_dicts if v.get('nli') is not None and v.get('status') != 'FAIL_HEDGE']
    n_scored = len(scored)
    report = {}
    for t in thresholds:
        n_would_pass = sum(1 for v in scored if v['nli']['entailment'] >= t)
        report[t] = {
            'n_scored': n_scored,
            'n_would_pass': n_would_pass,
            'pass_rate': (n_would_pass / n_scored) if n_scored else None,
        }
    return report

