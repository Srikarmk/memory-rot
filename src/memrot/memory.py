"""Memory stores: how memories are written after each session, and what the
answering model sees at question time.

Write side (done once per history by the writer model):
  rule      every first-person sentence the user wrote, appended
  llm       facts extracted by the writer model with a fixed prompt, appended
  overwrite llm facts, but before adding them the writer is shown the most
            similar existing entries and names the ones they replace, which are deleted
  summary   one rolling summary, rewritten by the writer after every session

Read side (per answering model), the systems compared in the study:
  a_rule  rule store, top-k by similarity
  a       llm store, top-k by similarity
  b       as a, with the date each entry was saved shown next to it
  c       overwrite store, top-k by similarity
  d       llm store, top-k by similarity plus a recency bonus
  e       the rolling summary
  f       every session verbatim (full context, an upper bound)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from . import config as C
from .embed import Embedder
from .history import History
from .llm import Client

SYSTEMS = ["a_rule", "a", "b", "c", "d", "e", "f"]
SYSTEM_LABEL = {
    "a_rule": "append, rule-extracted", "a": "append (vector store)", "b": "append + timestamps",
    "c": "overwrite on update", "d": "recency-weighted retrieval", "e": "rolling summary",
    "f": "full context",
}
SYSTEM_STORE = {"a_rule": "rule", "a": "llm", "b": "llm", "c": "overwrite", "d": "llm"}

EXTRACT_SYSTEM = "You write the long-term memory of a personal assistant."
EXTRACT_PROMPT = (
    "Below is the user's message from one conversation. List the facts about the user that are worth "
    "remembering for future conversations, such as where they live, the people they work with, their "
    "projects and dates, the tools they use, what they eat and their routines. Write each fact as a short "
    "standalone sentence that starts with \"The user\". Put one fact per line. If there is nothing worth "
    "remembering, reply NONE.\n\nMessage:\n{text}"
)
OVERWRITE_PROMPT = (
    "Existing memories:\n{existing}\n\nNew memories from the latest conversation:\n{new}\n\n"
    "An existing memory should be replaced if a new memory is about the same thing (the same fact about "
    "the user). Replaced memories are deleted and the new ones are kept. List the numbers of the existing "
    "memories that the new memories replace, separated by commas. If none, reply NONE."
)
SUMMARY_SYSTEM = "You maintain a running summary of what a personal assistant knows about the user."
SUMMARY_PROMPT = (
    "Current summary:\n{summary}\n\nLatest conversation ({date}):\nUser: {text}\n\n"
    "Rewrite the summary so that it also includes anything worth remembering about the user from the latest "
    "conversation. Keep it under {words} words. Reply with the summary only."
)
ANSWER_SYSTEM = (
    "You are a personal assistant with a long-term memory of earlier conversations with the user. "
    "Today is {date}.\n\n{memory}\n\nAnswer the user's question using what you remember. "
    "Answer in one or two sentences."
)

FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'll|my|me|we|our|us)\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Entry:
    eid: int
    text: str
    session: int
    date: str
    removed_at: int | None = None

    def active(self, t: int) -> bool:
        return self.session <= t and (self.removed_at is None or self.removed_at > t)


def rule_extract(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip() and FIRST_PERSON.search(s)]


def parse_facts(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line or line.upper().startswith("NONE") or len(line) < 6:
            continue
        out.append(line)
    return out


def parse_numbers(text: str, n: int) -> list[int]:
    if text.strip().upper().startswith("NONE"):
        return []
    return sorted({int(x) for x in re.findall(r"\d+", text) if 1 <= int(x) <= n})


@dataclass
class Build:
    hid: str
    rule: list[Entry]
    llm: list[Entry]
    overwrite: list[Entry]
    summaries: list[str]
    costs: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Build":
        return cls(d["hid"], [Entry(**e) for e in d["rule"]], [Entry(**e) for e in d["llm"]],
                   [Entry(**e) for e in d["overwrite"]], d["summaries"], d["costs"])

    def store(self, name: str) -> list[Entry]:
        return {"rule": self.rule, "llm": self.llm, "overwrite": self.overwrite}[name]


def _cost(session: int, kind: str, r: dict) -> dict:
    return {"session": session, "kind": kind, "prompt_tokens": r["prompt_tokens"],
            "output_tokens": r["output_tokens"], "seconds": r["seconds"], "cached": r["cached"]}


def build(h: History, client: Client, emb: Embedder, writer: str = C.WRITER, summary: bool = True) -> Build:
    rule, llm, ow, summaries, costs = [], [], [], [], []
    summ = ""
    for s in h.sessions:
        for t in rule_extract(s.text):
            rule.append(Entry(len(rule), t, s.idx, s.date))

        r = client.chat(writer, [{"role": "system", "content": EXTRACT_SYSTEM},
                                 {"role": "user", "content": EXTRACT_PROMPT.format(text=s.text)}],
                        C.options("extract"))
        costs.append(_cost(s.idx, "extract", r))
        new = parse_facts(r["text"])
        for t in new:
            llm.append(Entry(len(llm), t, s.idx, s.date))

        active = [e for e in ow if e.removed_at is None]
        if new and active:
            ev = emb.encode([e.text for e in active])
            nv = emb.encode(new)
            sims = nv @ ev.T
            best: dict[int, float] = {}
            for row in sims:
                for j in np.argsort(-row)[:C.OVERWRITE_NEIGHBOURS]:
                    best[int(j)] = max(best.get(int(j), -1.0), float(row[j]))
            cand = sorted(best, key=lambda j: -best[j])[:C.OVERWRITE_MAX_CANDIDATES]
            existing = "\n".join(f"{i + 1}. {active[j].text}" for i, j in enumerate(cand))
            r = client.chat(writer, [{"role": "user", "content": OVERWRITE_PROMPT.format(
                existing=existing, new="\n".join(f"- {t}" for t in new))}], C.options("overwrite"))
            costs.append(_cost(s.idx, "overwrite", r))
            for n in parse_numbers(r["text"], len(cand)):
                active[cand[n - 1]].removed_at = s.idx
        for t in new:
            ow.append(Entry(len(ow), t, s.idx, s.date))

        if summary:
            r = client.chat(writer, [{"role": "system", "content": SUMMARY_SYSTEM},
                                     {"role": "user", "content": SUMMARY_PROMPT.format(
                                         summary=summ or "(empty)", date=s.date, text=s.text,
                                         words=C.SUMMARY_WORDS)}], C.options("summary"))
            costs.append(_cost(s.idx, "summary", r))
            summ = r["text"].strip()
        summaries.append(summ)
    return Build(h.hid, rule, llm, ow, summaries, costs)


def retrieve(entries: list[Entry], t: int, qvec: np.ndarray, emb: Embedder, k: int,
             recency: bool = False) -> list[Entry]:
    act = [e for e in entries if e.active(t)]
    if not act:
        return []
    sims = emb.encode([e.text for e in act]) @ qvec
    if recency:
        rec = np.array([C.RECENCY_DECAY ** (t - e.session) for e in act])

        def norm(x):
            span = x.max() - x.min()
            return (x - x.min()) / span if span > 0 else np.zeros_like(x)

        score = norm(sims) + C.RECENCY_WEIGHT * norm(rec)
    else:
        score = sims
    # ties broken by recency, then insertion order, so ranking is deterministic
    order = sorted(range(len(act)), key=lambda i: (-round(float(score[i]), 6), -act[i].session, act[i].eid))
    return [act[i] for i in order[:k]]


def memory_block(system: str, h: History, b: Build, q, emb: Embedder, k: int) -> tuple[str, list[str]]:
    """Text placed in the answer prompt, and the list of memory items it shows."""
    t = q.asked_after
    if system == "e":
        s = b.summaries[t] or "(empty)"
        return f"Your summary of what you know about the user:\n{s}", [s]
    if system == "f":
        lines = [f"[{s.date}] User: {s.text}" for s in h.sessions[:t + 1]]
        return "Transcript of all earlier conversations:\n" + "\n".join(lines), lines
    hits = retrieve(b.store(SYSTEM_STORE[system]), t, emb.query(q.text), emb, k, recency=system == "d")
    if system == "b":
        lines = [f"- [{e.date}] {e.text}" for e in hits]
        head = "Memories retrieved for this question, each with the date it was saved:"
    else:
        lines = [f"- {e.text}" for e in hits]
        head = "Memories retrieved for this question:"
    return head + "\n" + ("\n".join(lines) if lines else "(none)"), [e.text for e in hits]


def answer_messages(system: str, h: History, b: Build, q, emb: Embedder, k: int) -> tuple[list[dict], list[str]]:
    block, items = memory_block(system, h, b, q, emb, k)
    msgs = [{"role": "system", "content": ANSWER_SYSTEM.format(date=q.date, memory=block)},
            {"role": "user", "content": q.text}]
    return msgs, items
