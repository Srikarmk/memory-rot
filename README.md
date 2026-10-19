# memory-rot

**Question.** An agent with long-term memory learns a fact about its user, the fact changes, and later the
agent is asked. Does it answer with the old fact, the new one, or both? What makes that worse, and which
memory designs fix it?

This repository is a controlled study on synthetic, fictional histories with local open-weight models.
Every number in [reports/results.md](reports/results.md) is produced by the code here from cached model
calls, so it regenerates exactly. Pre-registered predictions, with dated addenda: [HYPOTHESES.md](HYPOTHESES.md).

## Findings

Numbers are from [reports/results.md](reports/results.md): 10 users x 6 facts = 60 questions per cell,
95% bootstrap intervals over users. "Stale" = the answer gives an old value as the current one.

- **The common design goes stale often.** With a plain append-only vector store (facts extracted by an LLM,
  top-5 by similarity), after one explicit update the answer was stale on 43% [32, 55] of questions for
  `qwen3:8b`, 32% [20, 43] for `llama3.1:8b`, 23% [13, 33] for `gemma3:12b` and 43% [33, 53] for
  `qwen3:30b-a3b` (run with thinking on). Retrieval was not the main problem: on 80% of questions both the old
  and the new value were in the retrieved entries, and the model picked the old one or hedged.
- **Repetition beats recency.** Stating the old value 10 times instead of once raised the stale rate of
  the append store by 57 to 60 pp for every model (`qwen3:8b`: 25% to 85%), far above the rewording noise
  floor (at most 10 pp). Moving the question 20 sessions further from the update instead of 0 changed it by
  5 to 7 pp, none of which passes the noise-floor rule: similarity retrieval does not see time, so how long
  ago the update was barely matters, and how often the old fact was repeated dominates.
- **Implied updates break even full context.** When the new value is simply mentioned ("the commute here in
  Austin") rather than announced as a change, the full-transcript baseline answered with the old value on
  72% (`qwen3:8b`) and 90% (`llama3.1:8b`) of questions, up from 5% and 18%; `gemma3:12b` went from 0% to
  25%. The rolling summary went from 0% to 20 to 27% stale.
- **What fixes it.** Against the append store on baseline questions: overwrite-on-update cut stale answers by
  23 to 38 pp and the rolling summary by 23 to 43 pp, for every model. Timestamps on entries helped the
  8B/12B models only a little (-3 to -8 pp, intervals touching zero) but helped the thinking model a lot
  (-25 pp), which used the dates. Recency-weighted retrieval helped by 13 to 18 pp, mainly by turning
  retrievals that held only the old value (15% of questions) into ones that included the update (3% left).
- **What the fixes cost.** Overwrite needs a second writer call on most sessions (1.82 calls and 296 writer
  tokens per session, against 1.00 and 160), and on 13% of questions the current value was missing from
  what it retrieved, mostly because the writer had deleted or dropped the new entry (the store itself
  lacked it on 8%). The summary rewrites itself every session (278 writer tokens and about 3 s of compute
  per session). Full context needs about 6 times the answer prompt (873 tokens against 142).
- **Noise floor.** Rerunning the baseline questions with the cache bypassed changed 0 of 420 answers.
  Rewording the histories moved stale rates by up to 10 pp, which is the bar every effect is held to.
- **Judge.** Blind labels (one pass, no second labeller yet) agreed with the deterministic judge on 48 of 50 random answers (96%) and on
  20 of 25 answers that mention both values (80%); the disagreements are on the hedged boundary.

How the pre-registered predictions fared (details in results.md): repetition driving staleness (2), recency
mattering little for similarity retrieval (3), chains making the append store worse (5) and overwrite and
summary being the strongest fixes (6) held. Wrong or partly wrong: the append store was not stale on 25% or
more for every model (`gemma3:12b` 23%); full context was stale on more than 15% for `llama3.1:8b` (18%);
recency weighting did not get worse with a longer gap and helped more than timestamps for the 8B models;
the summary did not hedge more than the append store; and the larger model was no less stale on the plain
append store (43%, the same as `qwen3:8b`), though with timestamps it was.

What this adds to prior work: a factor-by-factor measurement, inside common agent memory designs, of the
repetition-versus-recency bias that Kurfali and Östling found for contradictory facts in long documents,
with the cost of each fix.

## Design

**Histories.** Each fictional user has six facts: the city they live in, their manager, a project deadline,
their note-taking app, their diet and their weekly team-meeting day. Each fact has a chain of values
(A, then B, then C), and every value is a distinctive string (a city, a full name, a month and day) so an
answer can be checked by matching strings. A history is a sequence of dated sessions, each one short user
message:

1. the first value of every fact is stated `rep` times, two facts per session, phrased from templates
   ("For context, I'm based in Cleveland.", "My notes live in Zettlr, so plain text with headings works
   best."), with six filler sessions mixed in;
2. each fact is updated once (A to B), or twice (A to B to C);
3. filler sessions follow: plain requests with nothing personal ("How long should I steep green tea?"),
   unrelated personal facts, and near misses that use the same kind of value for something else ("My
   sister lives in Phoenix.");
4. each fact is asked about exactly `gap` sessions after its last update ("Which city do I live in?").

Randomness comes from named streams keyed by user and purpose, so changing one factor leaves the rest of a
history unchanged (histories that differ only in the gap are prefixes of each other). `memrot generate`
checks every history: the value a question expects is the last one stated before it, the factor levels are
what the condition says, and no fact's value appears in any session that is not about that fact.

**Conditions** (one factor at a time from the baseline):

| condition | old value stated | sessions since update | update | chain |
|---|---|---|---|---|
| base | 3 | 5 | explicit | A to B |
| rep1, rep10 | 1, 10 | 5 | explicit | A to B |
| gap0, gap20 | 3 | 0, 20 | explicit | A to B |
| implied | 3 | 5 | implied | A to B |
| chain2 | 3 | 5 | explicit | A to B to C |
| aa | as base, reworded with a different phrasing seed and filler draw (the A/A noise floor) | | | |

An explicit update says something changed ("I no longer report to Ingrid Brandauer; my manager is now
Leandra Arceneaux."). An implied update is phrased exactly like an ordinary statement of the new value
("The commute here in Austin..."), with nothing marking a change.

**Users.** Users 0 and 1 were the pilot (used to check the stores, prompts and judge on real transcripts)
and are excluded from every reported number; the full run uses users 2 to 11.

**Memory systems.** One writer model (`qwen3:8b`) builds every store after every session, so all answering
models read identical memories:

| | system | write side | read side |
|---|---|---|---|
| a_rule | append, rule-extracted | every first-person sentence the user wrote | top-k by cosine similarity |
| a | append (vector store) | facts extracted by the writer with a fixed prompt, one per line | top-k by cosine similarity |
| b | append + timestamps | as a | as a, each entry shown with the date it was saved |
| c | overwrite on update | as a, but the writer is first shown the most similar existing entries (top 3 per new fact, at most 8) and names the ones the new facts replace; those are deleted | top-k by cosine similarity |
| d | recency-weighted retrieval | as a | min-max normalised similarity plus min-max normalised recency (0.95 per session), equal weights as in Generative Agents |
| e | rolling summary | the writer rewrites one summary (under 80 words) after every session | the summary |
| f | full context | nothing | every session verbatim, with dates (all histories fit in 8,192 tokens) |

Embeddings are `BAAI/bge-small-en-v1.5`; k = 5, and 3 and 10 for the append store on the baseline. The
answer prompt is the same for every system apart from the memory block, gives today's date, and asks for
one or two sentences. All prompts are in [src/memrot/memory.py](src/memrot/memory.py).

**Outcomes.** A deterministic judge ([src/memrot/judge.py](src/memrot/judge.py)) finds which of the fact's
values the answer mentions:

- *correct*: the current value is given as the answer (an old value may appear, framed as past: "you moved
  from Denver to Boston");
- *stale*: an old value is given as the answer;
- *hedged*: both appear and neither is clearly the current one;
- *neither*: no value of the fact appears;

plus whether the answer mentions a change. When both values appear, cue words in the same clause ("used
to", "from", "no longer", "now", "rescheduled to") decide which is presented as current. A blind label check
of a random sample, and of an extra sample of answers mentioning both values, measures the judge's
agreement with a person.

**Statistics.** Intervals are 95% percentile bootstraps over users (2,000 draws); a user's six questions
share one history, so users, not answers, are resampled. Effects are paired by user and fact. Two noise
floors: rewording (the `aa` condition minus `base`) and rerun (the baseline questions asked again with the
cache bypassed). An effect counts only if its interval excludes zero and it is larger than the rewording
gap. The statistics follow agent-interface-lab's task-level bootstrap and A/A gap, adapted to resample users.

**Models.** `qwen3:8b`, `llama3.1:8b` and `gemma3:12b` answer the full grid (8 conditions x 7 systems,
plus k = 3 and 10 for the append store). `qwen3:30b-a3b` as installed is the thinking-only Qwen3 release:
`think: false` does not stop its reasoning (it moves into the visible answer and the length cap cuts it off),
so it runs with `think: true` and a 2,048-token cap, and only its final answer is judged. It is slower, so it
answers a reduced grid: the baseline on systems a to f, and rep1, rep10 and aa on systems a, b and f.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) with `qwen3:8b`,
`llama3.1:8b`, `gemma3:12b` and `qwen3:30b-a3b` pulled; the embedding model downloads on first use.

```sh
uv sync
uv run pytest
uv run memrot generate                      # write and validate data/histories.jsonl
uv run memrot build                         # writer builds every store (cached)
uv run memrot answer --models qwen3:8b,llama3.1:8b,gemma3:12b   # ask, full grid
uv run memrot answer --models qwen3:30b-a3b --systems a,b,c,d,e,f --no-k --conditions base
uv run memrot answer --models qwen3:30b-a3b --systems a,b,f --no-k --conditions rep10,rep1,aa
uv run memrot answer --models qwen3:8b --conditions base --tag rerun --no-k   # rerun noise floor
uv run memrot report                        # reports/results.md and reports/figures/
uv run memrot show --user 2 --condition base   # read one history, its memories and answers
uv run memrot handcheck sample              # then label handcheck/labels.csv and run: handcheck score
```

Every model call is cached in `cache/llm.sqlite`, keyed by model, messages, options and a run tag. With the
cache present, `--cache-only` rebuilds every store and answer without a server; this was checked, and all 96
builds and every stored answer came back identical. Settings: temperature 0, seed 7, `num_ctx` 8192,
`think: false` for `qwen3:8b` and `think: true` for `qwen3:30b-a3b` (Ollama's `think` request field), a
`num_predict` cap per call type. Model digests are recorded in `results/models.json`.

Runtime on an M5 Pro (48 GB), much of it sharing the GPU with another study: building the stores took
2.6 hours (3,575 writer calls), and answering about 5 more hours of wall time across the four models. The
cache holds about 16,800 calls and 12.7 hours of server compute time.

## Limits

- Small local models, 4-bit quantised, at temperature 0. A hosted frontier model may behave differently.
- Synthetic, templated histories. Real users restate and revise facts less cleanly, and real memories hold
  far more entries; the effects here are about mechanisms, not production rates.
- Ten users with six facts each per condition. Intervals are wide, and many cells cannot separate small
  effects from rewording noise.
- `qwen3:30b-a3b` could only be run as a reasoning model and on a reduced grid, so "larger model" and
  "thinks before answering" are confounded for it.
- Latencies were measured while another workload shared the GPU; they are comparable across systems, not
  across machines.
- One writer model builds every store, so model differences are differences in reading, not writing.
- The memory systems are deliberately simple versions of common designs, not reimplementations of any
  product.
- The judge is string matching plus cue-word rules; its agreement with a single blind labelling pass is reported.

## Prior work

- Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory* (ICLR 2025,
  [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)). A benchmark with a knowledge-update category and
  time-aware query expansion as one fix. This study isolates single factors (repetition, recency, update
  style) on fully synthetic histories instead of measuring overall accuracy.
- Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents* (ACL 2024,
  [LoCoMo](https://aclanthology.org/2024.acl-long.747)). Long multi-session conversations; reports that
  models struggle with temporal reasoning.
- Chhikara et al., *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*
  ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413), 2025). Extracts facts, retrieves similar existing
  memories and has an LLM choose ADD, UPDATE, DELETE or NOOP; system c here is a minimal version of that
  update step.
- Rasmussen et al., *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*
  ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956), 2025). Marks contradicted facts invalid with
  timestamps instead of deleting them.
- Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (UIST 2023,
  [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)). Retrieval as the sum of min-max normalised
  recency, importance and relevance with all weights 1; system d uses the recency and relevance terms.
- Packer et al., *MemGPT: Towards LLMs as Operating Systems*
  ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560), 2023). Tiered memory managed by the model.
- Xu et al., *Knowledge Conflicts for LLMs: A Survey* (EMNLP 2024,
  [arXiv:2403.08319](https://arxiv.org/abs/2403.08319)). Taxonomy of context-memory, inter-context and
  intra-memory conflicts; stale-versus-new memory entries are an inter-context conflict.
- Xie et al., *Adaptive Chameleon or Stubborn Sloth* (ICLR 2024,
  [arXiv:2305.13300](https://arxiv.org/abs/2305.13300)). Models accept coherent contradicting context but
  show confirmation bias when evidence is mixed.
- Kurfali and Östling, *Conflicting Needles in a Haystack: How LLMs behave when faced with contradictory
  information* (EMNLP 2025, [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1742/)). With
  mutually exclusive facts in long documents, models rarely flag the conflict and show biases toward
  repetition and recency. That is the closest prior result; this study asks the same question inside an
  agent memory pipeline, where retrieval and summarisation decide what the model gets to see.

## License

MIT, Kameswara Sai Srikar Manda, 2026.
