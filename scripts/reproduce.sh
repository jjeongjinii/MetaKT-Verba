#!/usr/bin/env bash
# Recomputes every expert-based result in the paper from the released data (CPU only).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

echo "== Table 3: reliability, negative controls, convergent validity"
python src/aggregate_ratings.py
echo "== Table 3: sentence-level discrimination and threshold sweep"
python src/threshold_retuning_analysis.py
echo "== Table 4: NLI vs. few-shot LLM judges"
python src/run_three_way_comparison.py --label-mode strict
python src/run_three_way_comparison.py --label-mode lenient
echo "== Table 5: within-indicator discrimination"
python src/within_indicator_auroc.py
echo "== Section 5.4: error direction and rater comments"
python src/export_disagreement_cases.py
python src/code_free_text_disagreements.py
echo "== Section 5.4: minimal-pair study"
python src/analyze_cognitive_avoidance_survey.py
echo "Done. Outputs are in outputs/."
