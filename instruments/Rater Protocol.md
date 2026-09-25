# Expert Rating Protocol

This document summarizes the instructions given to expert raters evaluating MetaKT-Verba narratives.

## Task

For each item, raters were shown:

1. a summary of a student's recent problem-solving behavior, and
2. an AI-generated narrative based on that record.

Raters judged whether the narrative was supported **only by the displayed behavioral record**, rather than whether it sounded plausible from general knowledge. Narratives were generated in English; judgments were made using the English text.

The displayed context included the skill, recent correctness, hint use, response time relative to the student's personal mean, and prior cumulative accuracy for the skill.

## Sentence-level support

Each generated sentence was assigned one label:

| Label | Criterion |
|---|---|
| Supported | Clearly supported by the displayed behavioral record |
| Partially Supported | Directionally consistent but exaggerated or only partly supported |
| Unsupported | Unrelated to or contradicted by the behavioral record |
| Cannot Determine | The displayed information is insufficient to judge |

## Holistic ratings

Raters also scored each narrative from 1 to 5 on three dimensions.

### Faithfulness

**Question:** Is the narrative consistent with the displayed behavioral record?

| Score | Anchor |
|---:|---|
| 1 | Clearly contradicts the behavioral record |
| 2 | Most content is not supported by the behavioral record |
| 3 | Some content is supported, but at least one central claim lacks support |
| 4 | Nearly all content is supported, with only minor exaggeration or nuance differences |
| 5 | All content is fully supported by the behavioral record |

### Actionability

**Question:** Can a teacher infer a concrete next action from the narrative?

| Score | Anchor |
|---:|---|
| 1 | No actionable direction is apparent |
| 2 | Conveys only a vague concern, without concrete cues |
| 3 | Suggests a direction, but a concrete action is difficult to identify |
| 4 | A concrete intervention direction can be inferred |
| 5 | Clearly suggests a concrete and immediately actionable intervention direction |

### Clarity

**Question:** Is the narrative clear and easy to understand?

| Score | Anchor |
|---:|---|
| 1 | Difficult to understand or highly ambiguous |
| 2 | Requires repeated reading to understand |
| 3 | Generally understandable, with some ambiguity |
| 4 | Clear and easy to understand |
| 5 | Very clear, concise, and immediately understandable |

## Optional comment

Raters could identify any part of the narrative that they felt did not match the behavioral record, specifying the relevant sentence when possible.

## Rating procedure

Each rater received an anonymous rater ID, an assigned JSON item file, an optional calibration-practice file, and `rater_tool.html`. Progress could be resumed in the same browser. After completing the assigned items, ratings were exported as a CSV file.
