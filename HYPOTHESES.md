# Hypotheses and analysis plan

Written 2026-10-19 at 11:50 EDT, before the full run. Nothing below is edited after the full run starts;
corrections go in a dated addendum at the end.

What had been seen when this was written: a pilot on users 0 and 1 (all eight conditions, `qwen3:8b`
answering), used to check the memory stores, prompts and judge on real transcripts. Those two users are
excluded from every reported number; the full run uses users 2 to 11. The pilot showed, on 12 questions per
cell: the plain append store stale on roughly 40% of baseline questions, the overwrite store and the
rolling summary rarely stale, and the full-context baseline stale on most implied updates. The predictions
below were written with that in view, so the pilot-informed ones are marked (P).

## Primary question

When a fact about the user changes, how often does an agent with long-term memory answer with the old
value, and how does that depend on how often the old value was stated, how long ago the update was, how
the update was phrased, and the memory design?

## Predictions

Stale rate = share of questions answered with an old value. pp = percentage points.

1. Baseline (P). The append-only vector store (a) is stale on 25% or more of baseline questions for every
   model. Full context (f) is stale on 15% or fewer.
2. Repetition is the main driver for append stores. For a, stale rate rises from 1 to 10 statements by at
   least 20 pp for every model. The overwrite store (c) shows little or no repetition effect (under 10 pp),
   because restatements replace each other. Full context (f) shows a smaller effect than a.
3. Recency matters little for similarity-only retrieval. For a, gap20 minus gap0 is within the A/A gap,
   because cosine similarity does not see time and the filler sessions rarely outrank fact entries. For
   recency-weighted retrieval (d), stale rate rises with the gap (the recency bonus of the update fades).
4. Implied updates (P) are worse than explicit ones for every system that relies on the model noticing a
   change: a, b, d, e and f each at least 15 pp more stale. The overwrite store (c) is affected less, because
   its writer compares the new entry with the old one directly.
5. A second update (A to B to C) raises the stale rate for the append stores (two old values compete with
   one current value).
6. Fixes, against a on baseline questions:
   - timestamps (b) cut the stale rate by at least 10 pp for the larger models; smaller models use dates
     less reliably;
   - overwrite on update (c) cuts it the most among the vector stores, at the price of more "neither"
     answers when the writer deletes the wrong entry;
   - recency weighting (d) helps less than timestamps or overwrite;
   - the rolling summary (e) has a low stale rate but more "hedged" answers that merge old and new values;
   - write-side cost: c and e need extra writer calls per session; read-side cost: f has by far the longest
     prompts and b adds tokens for dates.
7. Retrieval depth: larger k for a puts both values in context more often and raises the hedged rate.
8. Model size: given the same memories, `qwen3:30b-a3b` is stale less often than the 8B models,
   mostly when both values are in context.
9. Answers rarely mention that the fact changed (under 30% of correct answers) except with timestamps or
   full context.

What would contradict "repetition beats recency in agent memory": a repetition effect for a within the
A/A gap, or stale rates for a that fall with more statements.

## Analysis plan (fixed in advance)

- Unit of resampling: user (one history family, six probed facts). 95% percentile bootstrap, 2,000 draws.
  Effects are paired by user and fact.
- Primary metric: stale rate. Secondary: correct, hedged, neither, mentions-change, and retrieval
  diagnostics (whether the retrieved entries held the current value, an old value, or both).
- Primary contrasts: rep10 minus rep1 and gap20 minus gap0 for each system and model; each of b, c, d, e
  and f minus a on baseline questions. Everything else is secondary and reported without a claim.
- Noise floors: rewording A/A (`aa` minus `base`, same users and values, different phrasing seed and
  filler draw) and rerun A/A (`qwen3:8b` baseline questions asked again with the cache bypassed). An effect is
  reported as real only if its interval excludes zero and its size exceeds the rewording gap for the same
  model and system.
- Multiple comparisons: about 7 systems x 4 models x 7 contrasts. No correction; the primary contrasts
  above are the ones read as confirmatory, the rest as exploratory.
- Exclusions: none planned. A failed call is retried; if it still fails the run stops rather than dropping
  the question.
- Judge audit: blind hand labels on a random sample of 50 answers plus 25 answers that mention both values;
  agreement is published.

## Run settings

- Writer for every store: `qwen3:8b`. Answering models: `qwen3:8b`, `llama3.1:8b`, `gemma3:12b`,
  `qwen3:30b-a3b`, as many as time allows, in that order.
- Temperature 0, seed 7, `num_ctx` 8192, `think: false` for qwen3, `num_predict` caps per call type.
- Embeddings `BAAI/bge-small-en-v1.5`; k = 5 (3 and 10 for a on baseline); recency decay 0.95 per session
  with equal weights; overwrite shows the writer the top 3 similar entries per new fact, at most 8.
- Users 2 to 11 (10 users x 6 facts = 60 questions per cell).

## Addenda

**2026-10-19, 14:40 EDT (during the run, before any full-run results were read).** Grid cut for time: the
build took 2.6 hours and each 8B answering model about 2 hours while another workload shared the GPU.
`gemma3:12b` and `qwen3:30b-a3b` therefore answer only conditions base, rep1, rep10, gap0, gap20 and aa
(no implied or chain2), systems a to f (no rule-extracted store), and k = 5 only. `qwen3:8b` and
`llama3.1:8b` run the full grid. Prediction 8 is tested on the conditions both larger models have.

**2026-10-19, 18:05 EDT.** The other workload on the GPU finished, so the cut above was reversed: after
their reduced runs, `gemma3:12b` and `qwen3:30b-a3b` are run on the full grid (all eight conditions, the
rule-extracted store, and k = 3 and 10), filling in from the cache. Every model then has the same grid.

**2026-10-19, 18:05 EDT.** `qwen3:30b-a3b` as installed is the thinking-only Qwen3 release. With
`think: false` it still reasons, in the visible answer, and the 80-token cap cut those answers off before
they reached a conclusion (85% of its first full-context answers were scored stale for that reason). That
run was discarded and removed from the cache. The model is rerun with `think: true` and `num_predict`
2048; only its final answer is judged. Its results are therefore a reasoning model's results and are
labelled that way; prediction 8 (size helps) is confounded with reasoning and is reported as such. It runs
systems a to f on all eight conditions at k = 5 (no rule-extracted store, no extra k levels), in the order
base, aa, rep1, rep10, gap20, gap0, implied, chain2, as far as time allows.
At about 20 seconds per call while another of these runs shared the GPU, the full grid would take most of a
day, so `qwen3:30b-a3b` (thinking) is cut to: base on systems a to f, then rep10, rep1 and aa on systems
a, b and f.
