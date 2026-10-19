"""Running the study: build memories with the writer model, then ask questions."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import config as C
from .embed import Embedder
from .history import History
from .judge import context_has, judge
from .llm import Client
from .memory import SYSTEMS, Build, answer_messages, build

# conditions sharing a prefix with a longer one are built after it, so they hit the cache
BUILD_ORDER = ["gap20", "base", "gap0", "chain2", "implied", "rep1", "rep10", "aa"]


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def build_path(hid: str) -> str:
    return os.path.join(C.RESULTS, "builds", f"{hid}.json")


class Progress:
    def __init__(self, total: int, label: str, every: int = 25):
        self.total, self.label, self.every = total, label, every
        self.done, self.t0, self.lock = 0, time.time(), threading.Lock()

    def tick(self, client: Client):
        with self.lock:
            self.done += 1
            if self.done % self.every == 0 or self.done == self.total:
                el = time.time() - self.t0
                print(f"[{self.label}] {self.done}/{self.total} new_calls={client.stats['misses']} "
                      f"hits={client.stats['hits']} elapsed={el:.0f}s", flush=True)


def run_builds(hs: list[History], client: Client, emb: Embedder, workers: int = 2,
               writer: str = C.WRITER) -> dict[str, Build]:
    os.makedirs(os.path.join(C.RESULTS, "builds"), exist_ok=True)
    by_user: dict[int, list[History]] = defaultdict(list)
    for h in sorted(hs, key=lambda h: BUILD_ORDER.index(h.condition)):
        by_user[h.user].append(h)
    out: dict[str, Build] = {}
    prog = Progress(len(hs), f"build {writer}", every=1)

    def work(user_hs: list[History]):
        for h in user_hs:
            b = build(h, client, emb, writer)
            with open(build_path(h.hid), "w") as f:
                json.dump(b.to_dict(), f)
            out[h.hid] = b
            prog.tick(client)
        emb.save()

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, by_user.values()))
    emb.save()
    return out


def load_builds(hs: list[History]) -> dict[str, Build]:
    out = {}
    for h in hs:
        with open(build_path(h.hid)) as f:
            out[h.hid] = Build.from_dict(json.load(f))
    return out


def answer_jobs(hs: list[History], systems: list[str], k_extra: dict[str, list[int]]) -> list[tuple]:
    jobs = []
    for h in hs:
        for qi, q in enumerate(h.questions):
            for s in systems:
                ks = [C.K_DEFAULT] if s in ("a_rule", "a", "b", "c", "d") else [0]
                if s in k_extra.get(h.condition, []):
                    ks = sorted(set(ks) | set(C.K_LEVELS))
                for k in ks:
                    jobs.append((h, qi, s, k))
    return jobs


def run_answers(hs: list[History], builds: dict[str, Build], client: Client, emb: Embedder, model: str,
                systems: list[str] = SYSTEMS, k_extra: dict | None = None, tag: str = "",
                workers: int = 2) -> list[dict]:
    k_extra = {"base": ["a"]} if k_extra is None else k_extra
    jobs = answer_jobs(hs, systems, k_extra)
    prog = Progress(len(jobs), f"answer {model}{' ' + tag if tag else ''}", every=50)

    def work(job):
        h, qi, s, k = job
        q = h.questions[qi]
        msgs, items = answer_messages(s, h, builds[h.hid], q, emb, k)
        r = client.chat(model, msgs, C.options("answer", model), tag=tag)
        v = judge(q.attr, r["text"], q.current, q.old)
        prog.tick(client)
        return {
            "model": model, "tag": tag, "hid": h.hid, "user": h.user, "condition": h.condition,
            "attr": q.attr, "system": s, "k": k, "question": q.text, "current": q.current, "old": q.old,
            "answer": r["text"], "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"],
            "seconds": r["seconds"], "cached": r["cached"], "n_items": len(items),
            "truncated": r.get("done_reason") == "length",
            "ctx_current": context_has(q.attr, items, q.current),
            "ctx_old": any(context_has(q.attr, items, o) for o in q.old),
            "outcome": v.outcome, "change_noted": v.change_noted,
        }

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(work, jobs))
    emb.save()
    os.makedirs(os.path.join(C.RESULTS, "answers"), exist_ok=True)
    path = os.path.join(C.RESULTS, "answers", f"{slug(model)}{'_' + tag if tag else ''}.jsonl")
    old = []
    if os.path.exists(path):
        keys = {(r["hid"], r["attr"], r["system"], r["k"]) for r in rows}
        with open(path) as f:
            old = [r for r in map(json.loads, f) if (r["hid"], r["attr"], r["system"], r["k"]) not in keys]
    with open(path, "w") as f:
        for r in old + rows:
            f.write(json.dumps(r) + "\n")
    return rows
