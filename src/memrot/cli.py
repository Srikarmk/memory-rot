"""memrot command line."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config as C


def _hs(args):
    from .history import CONDITIONS, generate
    conds = args.conditions.split(",") if args.conditions else list(CONDITIONS)
    users = [int(x) for x in args.user_list.split(",")] if args.user_list else C.USERS
    return [generate(u, c) for c in conds for u in users]


def _client(args):
    from .llm import Cache, Client
    os.makedirs(os.path.dirname(C.CACHE_DB), exist_ok=True)
    return Client(Cache(C.CACHE_DB), cache_only=getattr(args, "cache_only", False))


def cmd_generate(args):
    from .history import validate
    hs = _hs(args)
    bad = [(h.hid, e) for h in hs for e in validate(h)]
    os.makedirs(C.DATA, exist_ok=True)
    with open(os.path.join(C.DATA, "histories.jsonl"), "w") as f:
        for h in hs:
            f.write(json.dumps(h.to_dict()) + "\n")
    print(f"{len(hs)} histories, {sum(len(h.sessions) for h in hs)} sessions, {len(bad)} problems")
    for b in bad[:20]:
        print(b)
    return 1 if bad else 0


def cmd_build(args):
    from .embed import Embedder
    from .runner import run_builds
    hs = _hs(args)
    client = _client(args)
    if not args.cache_only:
        _record_model(C.WRITER)
    run_builds(hs, client, Embedder(), workers=args.workers)
    print("cache", client.stats)


def cmd_answer(args):
    from .embed import Embedder
    from .runner import load_builds, run_answers
    hs = _hs(args)
    client = _client(args)
    systems = args.systems.split(",") if args.systems else None
    kx = {} if args.no_k else None
    kw = {"systems": systems} if systems else {}
    for m in args.models.split(","):
        if not args.cache_only:
            _record_model(m)
        run_answers(hs, load_builds(hs), client, Embedder(), m, k_extra=kx, tag=args.tag,
                    workers=args.workers, **kw)
    print("cache", client.stats)


def _record_model(model: str):
    from .llm import model_info
    path = os.path.join(C.RESULTS, "models.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    from .llm import think_flag
    meta[model] = {**model_info(model), "think": think_flag(model), "options": C.options("answer", model)}
    if model == C.WRITER:
        meta[model]["writer"] = True
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)


def cmd_show(args):
    """Print one history with its memory stores and answers, for reading transcripts."""
    from .history import generate
    from .runner import build_path, slug
    from .memory import Build
    h = generate(args.user, args.condition)
    print(f"== {h.hid} {h.params}")
    for s in h.sessions:
        print(f"[{s.idx:2d} {s.date} {s.role}] {s.text}")
    if os.path.exists(build_path(h.hid)):
        with open(build_path(h.hid)) as f:
            b = Build.from_dict(json.load(f))
        print("\n== llm store")
        for e in b.llm:
            print(f"  s{e.session}: {e.text}")
        print("\n== overwrite store (final)")
        for e in b.overwrite:
            print(f"  s{e.session}{' removed@' + str(e.removed_at) if e.removed_at is not None else ''}: {e.text}")
        print("\n== final summary\n" + b.summaries[-1])
    path = os.path.join(C.RESULTS, "answers", f"{slug(args.model)}.jsonl")
    if os.path.exists(path):
        print("\n== answers")
        with open(path) as f:
            for r in map(json.loads, f):
                if r["hid"] == h.hid and (not args.attr or r["attr"] == args.attr):
                    print(f"  {r['attr']:11s} {r['system']:6s} k={r['k']:<2d} {r['outcome']:8s} "
                          f"ctx(cur={int(r['ctx_current'])},old={int(r['ctx_old'])}) | {r['answer']}")


def cmd_report(args):
    from .report import write_report
    write_report()


def cmd_handcheck(args):
    from . import handcheck
    if args.action == "sample":
        handcheck.sample(args.n)
    else:
        handcheck.score()


def main(argv=None):
    p = argparse.ArgumentParser(prog="memrot")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--user-list", default="")
        sp.add_argument("--conditions", default="")
        sp.add_argument("--workers", type=int, default=2)
        sp.add_argument("--cache-only", action="store_true")

    common(sub.add_parser("generate", help="generate and validate histories"))
    common(sub.add_parser("build", help="build memory stores with the writer model"))
    a = sub.add_parser("answer", help="ask the questions with one or more answering models")
    common(a)
    a.add_argument("--models", default=C.MODELS[0])
    a.add_argument("--systems", default="")
    a.add_argument("--tag", default="")
    a.add_argument("--no-k", action="store_true", help="skip the extra k levels")
    s = sub.add_parser("show", help="print a history, its memories and answers")
    s.add_argument("--user", type=int, default=0)
    s.add_argument("--condition", default="base")
    s.add_argument("--model", default=C.MODELS[0])
    s.add_argument("--attr", default="")
    sub.add_parser("report", help="write reports/results.md and figures")
    hc = sub.add_parser("handcheck", help="sample judgments for hand labelling, or score the labels")
    hc.add_argument("action", choices=["sample", "score"])
    hc.add_argument("--n", type=int, default=50)
    args = p.parse_args(argv)
    fn = {"generate": cmd_generate, "build": cmd_build, "answer": cmd_answer, "show": cmd_show,
          "report": cmd_report, "handcheck": cmd_handcheck}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
