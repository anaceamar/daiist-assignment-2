# Evaluation summary

## Retrieval (answerable questions)

| design | hit@1 | hit@3 | hit@5 | MRR@5 | multi: both in top 5 | words@3 | words@5 |
|---|---|---|---|---|---|---|---|
| fixed | 69% | 91% | 97% | 0.808 | 71% | 750 | 1250 |
| structure | 91% | 94% | 94% | 0.929 | 71% | 588 | 976 |
| structure-hybrid | 66% | 83% | 86% | 0.739 | 57% | 640 | 1001 |
| structure-hybrid-mode | 60% | 83% | 89% | 0.716 | 57% | 623 | 1008 |

Best by hit@3: **structure**. The server uses **structure**.

## End to end

| metric | value |
|---|---|
| graded answers | 42 / 42 |
| correct (all) | 40 / 42 |
| correct (answerable) | 33 / 35 |
| unsupported answers | 2 |
| correct refusals | 7 / 7 |
| answered an unanswerable question | 0 |
| wrong refusals | 0 / 35 |
| answers with valid citations (code) | 42 / 42 |
| citation guardrail fired | q23 |

## Human vs judge

No human grades yet (eval/human_labels.csv).

## Failure causes

- retrieval miss: 1
- wrong answer from right passage: 1

## Cost per question

- 4539 input + 225 output tokens, 2.05 model calls, 1.14 searches
- 6.3 s per question
- about $0.00192 at the assumed list price ($0.3/M in, $2.5/M out); free tier: $0
