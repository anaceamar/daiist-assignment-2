# Evaluation files

## questions.csv

30+ questions written by you from the documents, before tuning anything.

| Column | Content |
|---|---|
| `id` | `q01`, `q02`… |
| `question` | As a user would ask it. |
| `kind` | `single`, `multi` (two passages) or `unanswerable` (5+ of each of the last two). |
| `expected_source` | Document that answers it, as in your chunk metadata; `;` between several. Empty if unanswerable. |
| `expected_location` | Page, section or article, as in your metadata. Empty if unanswerable. |
| `reference_answer` | Short correct answer; for unanswerable, what the assistant should say. |

Make unanswerable questions close to the corpus's topic, not obviously off-topic.

## human_labels.csv

Your own grades of 20+ agent answers, done before reading the judge's verdicts.

| Column | Content |
|---|---|
| `id` | Question id. |
| `correct` | `1`/`0`: right against the reference (or a proper refusal). |
| `supported` | `1`/`0`: every claim is in the retrieved text. |
| `notes` | Why, for every `0`. |

Commit what your evaluation stage writes here too (answers, verdicts, summary).
