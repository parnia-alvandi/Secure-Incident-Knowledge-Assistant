# Baseline RAG: three iterations and why v1r was chosen

Evaluation set: 80 English questions, fixed by seed 42 (all 40 unanswerable ones, plus 20 answerable
questions from public documents and 20 from synthetic documents). Every number below comes from
`scripts/eval_baseline.py` on this machine (CPU, `qwen2.5:3b-instruct`, vector top-5). Results files:
`eval/results/baseline_eval_v1*.jsonl` and `baseline_eval_v2.jsonl`.

## Variants

| name | prompt | safeguards |
|---|---|---|
| **v1** | inline `[n]` citations, refuse with `INSUFFICIENT_EVIDENCE` | none |
| **v2** | `ANSWER:` / `SOURCES:` format plus a rule "refuse if the specific detail is not stated" | empty-answer guard, 1 repair retry |
| **v1r** | same as v1 | empty-answer guard, 1 repair retry (message matched to the v1 prompt) |

## Results (automatic counts)

| metric | v1 | v2 | **v1r** |
|---|---|---|---|
| Unanswerable questions correctly refused (40) | 35 (88%) | 10 (25%) | **40 (100%)** |
| ... `out_of_scope` / `near_miss` refused (20 each) | 20 / 15 | 4 / 6 | **20 / 20** |
| Unanswerable answered with real text | 0 | **29** | 0 |
| Answerable (40) answered with real text | 11 (28%) | 37 (92%) | **21 (52%)** |
| Answerable replies with no text (citation markers only) | 22 (55%) | 0 | 0 |
| Answerable wrongly refused | 5 | 0 | 16 |
| Answerable `unverified` (no valid citation) | 6 | 3 | 3 |
| Gold passage in the prompt (retrieval ceiling) | 32 (80%) | 32 (80%) | 32 (80%) |
| Answers containing numbers absent from the passages | 1 of 29 | 9 of 37 | 3 of 21 |
| Mean time per question | 29.1 s | 30.8 s | 29.2 s |

## What happened

1. **v1 looked fine but was not.** 27 of 80 replies were only a citation marker such as `[1]`. The pipeline counted
   them as `ANSWERED` because it only checked that a citation was valid, not that an answer existed. Fix: a reply
   without answer text is never `ANSWERED`; it gets one repair retry and then `no_answer`.
2. **v2 fixed the empty replies but broke refusals.** Forcing the `ANSWER:` format made the model answer almost
   everything, including 29 of 40 unanswerable questions, and the explicit "refuse if the detail is missing" rule did not
   change that. Rejected: refusing when the collection has no answer is a hard requirement.
3. **v1r keeps the refusal behaviour of v1 and removes the empty replies.** Of 27 repaired questions, 10 became real
   answers, 16 became refusals, 1 stayed `unverified`.

## Selection rule (fixed before the v1r run)

Refusal of unanswerable questions must be at least 80%; among variants that pass, choose the one with the most
answerable questions answered with real text. v2 fails the first condition; v1r beats v1 on the second.

## Reviewer's reading of answer correctness (draft, to be confirmed by the project owner)

| | v1 | v1r |
|---|---|---|
| Answers with text that match the reference | 12 of 13 | 23 of 24 (20 answered, 3 `unverified`) |
| Wrong | 1 (PHI-09) | 1 (PHI-09) |

The three `unverified` answers in v1r have the right text but cite a passage number that does not exist, so they
are correctly not labelled `ANSWERED`. v2 answers were not reviewed.

## Reading the v1r numbers honestly

- **16 of 40 answerable questions are refused.** 7 because the right passage never reached the model (retrieval),
  9 although it did (the model is conservative). The system is safe but misses answers.
- The perfect refusal rate is partly caused by this conservatism. Overall, the refuse-or-answer decision is right for
  64 of 80 questions (80%).
- Retrieval is the ceiling: the right passage is in the top 5 for 80% of the answerable questions. Hybrid search and
  re-ranking (phase 5) target this. Checking that an answer is supported by its cited passage (phase 7) targets the rest.

## Limits of this evidence

- One run per variant, 80 questions, all written from the same documents.
- The 40 unanswerable questions are all the English ones that exist, so there is no held-out set. The choice among
  three variants was made on the same data it is reported on. The prompt is not tuned further on this set.
- Correctness was judged against the reference answers by reading, not by an automatic faithfulness metric.
