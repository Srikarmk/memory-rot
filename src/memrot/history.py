"""Seeded generation of synthetic multi-session user histories.

One user gets six attributes, each with a chain of values (A, then B, then C).
A history states every attribute's first value `rep` times, then updates each
attribute once (A to B) or twice (A to B to C), with filler sessions in between.
Each attribute is asked about exactly `gap` sessions after its last update.

Randomness is drawn from named streams keyed by (user, phrasing seed, purpose),
so changing one factor leaves everything else identical. Conditions that differ
only in the gap produce histories that are prefixes of one another.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import random
from dataclasses import asdict, dataclass, field

from . import facts as F

CONDITIONS: dict[str, dict] = {
    "base": dict(rep=3, gap=5, style="explicit", chain=1, phr=0),
    "rep1": dict(rep=1, gap=5, style="explicit", chain=1, phr=0),
    "rep10": dict(rep=10, gap=5, style="explicit", chain=1, phr=0),
    "gap0": dict(rep=3, gap=0, style="explicit", chain=1, phr=0),
    "gap20": dict(rep=3, gap=20, style="explicit", chain=1, phr=0),
    "implied": dict(rep=3, gap=5, style="implied", chain=1, phr=0),
    "chain2": dict(rep=3, gap=5, style="explicit", chain=2, phr=0),
    # same design as base, reworded with a different phrasing and filler draw
    "aa": dict(rep=3, gap=5, style="explicit", chain=1, phr=1),
}

# condition -> (factor, level) for reporting; base is the shared reference level
FACTOR = {
    "base": ("baseline", "rep3/gap5/explicit/one update"),
    "rep1": ("repetition", "1"), "rep10": ("repetition", "10"),
    "gap0": ("recency", "0"), "gap20": ("recency", "20"),
    "implied": ("update style", "implied"), "chain2": ("update chain", "A->B->C"),
    "aa": ("A/A rewording", "same design, new wording"),
}

N_FILLERS_A = 6
N_BETWEEN_CHAIN = 2
MAX_POST = 24
START_DATE = dt.date(2026, 1, 5)


def rng(*key) -> random.Random:
    h = hashlib.sha256(repr(key).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


@dataclass
class Session:
    idx: int
    date: str
    role: str  # statement, update, filler_task, filler_personal, filler_near
    text: str
    facts: list[dict] = field(default_factory=list)


@dataclass
class Question:
    attr: str
    text: str
    asked_after: int  # memory holds sessions 0..asked_after inclusive
    date: str
    current: str
    old: list[str]
    update_session: int


@dataclass
class History:
    user: int
    condition: str
    params: dict
    user_name: str
    project: str
    chains: dict[str, list[str]]
    near: dict[str, list[str]]
    sessions: list[Session]
    questions: list[Question]

    @property
    def hid(self) -> str:
        return f"u{self.user:02d}-{self.condition}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hid"] = self.hid
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "History":
        d = dict(d)
        d.pop("hid", None)
        d["sessions"] = [Session(**s) for s in d["sessions"]]
        d["questions"] = [Question(**q) for q in d["questions"]]
        return cls(**d)


def _user_values(user: int, seed: int) -> dict:
    r = rng("values", seed, user)
    persons = [f"{a} {b}" for a, b in zip(r.sample(F.FIRST_NAMES, 6), r.sample(F.LAST_NAMES, 6))]
    months = r.sample(F.MONTHS, 5)
    dates = [f"{m} {r.randint(3, 27)}" for m in months]
    pools = {
        "city": r.sample(F.CITIES, 5),
        "manager": persons[:5],
        "deadline": dates,
        "notes_tool": r.sample(F.NOTES_TOOLS, 5),
        "diet": r.sample(F.DIETS, 5),
        "meeting_day": r.sample(F.DAYS, 5),
    }
    # first three are the chain A, B, C; the last two are near-miss values
    return {
        "user_name": F.USER_NAMES[user % len(F.USER_NAMES)],
        "project": r.choice(F.PROJECTS),
        "chains": {a: pools[a][:3] for a in F.ATTRIBUTES},
        "near": {a: pools[a][3:5] for a in F.ATTRIBUTES},
    }


def _filler_pool(user: int, seed: int, phr: int, near: dict) -> list[tuple[str, str]]:
    pool: list[tuple[str, str]] = [("filler_task", t) for t in F.TASKS]
    pool += [("filler_personal", t) for t in F.PERSONAL]
    for a in F.ATTRIBUTES:
        r = rng("near", seed, user, phr, a)
        for x, tmpl in zip(near[a], r.sample(F.NEAR_MISS[a], 2)):
            pool.append(("filler_near", tmpl.format(x=x)))
    r = rng("fillers", seed, user, phr)
    tasks = [p for p in pool if p[0] == "filler_task"]
    other = [p for p in pool if p[0] != "filler_task"]
    r.shuffle(tasks)
    r.shuffle(other)
    # alternate personal and task fillers so every stretch has both kinds
    out = []
    while tasks or other:
        if other:
            out.append(other.pop())
        if tasks:
            out.append(tasks.pop())
    return out


def _compose(r: random.Random, sentences: list[str], wrap: bool) -> str:
    parts = [r.choice(F.OPENERS)] + sentences
    if wrap:
        parts.append(r.choice(F.WRAPPERS))
    return " ".join(p for p in parts if p)


def generate(user: int, condition: str, seed: int = 0) -> History:
    p = CONDITIONS[condition]
    rep, gap, style, chain, phr = p["rep"], p["gap"], p["style"], p["chain"], p["phr"]
    vals = _user_values(user, seed)
    chains, near, proj = vals["chains"], vals["near"], vals["project"]
    fillers = _filler_pool(user, seed, phr, near)
    filler_a, filler_post = fillers[:N_FILLERS_A], fillers[N_FILLERS_A:N_FILLERS_A + MAX_POST]

    blocks: list[tuple[str, list[str], list[dict]]] = []  # (role, sentences, facts)

    stmt_sessions = []
    for j in range(rep):
        order = list(F.ATTRIBUTES)
        rng("order", seed, user, phr, j).shuffle(order)
        for k in range(0, len(order), 2):
            sents, fs = [], []
            for a in order[k:k + 2]:
                t = rng("stmt", seed, user, phr, a, j).choice(F.STATEMENT[a])
                sents.append(t.format(v=chains[a][0], p=proj))
                fs.append({"attr": a, "value": chains[a][0], "kind": "statement"})
            stmt_sessions.append(("statement", sents, fs))
    # fillers go at random slots in the learning phase, never after its last statement
    slots = sorted(rng("slotsA", seed, user, phr, rep).choices(range(len(stmt_sessions)), k=N_FILLERS_A))
    fi = 0
    for i, s in enumerate(stmt_sessions):
        while fi < len(slots) and slots[fi] == i:
            blocks.append((filler_a[fi][0], [filler_a[fi][1]], []))
            fi += 1
        blocks.append(s)

    def update_round(n: int):
        order = list(F.ATTRIBUTES)
        rng("uorder", seed, user, phr, n).shuffle(order)
        for a in order:
            new, old = chains[a][n], chains[a][n - 1]
            pool = F.EXPLICIT_UPDATE[a] if style == "explicit" else F.STATEMENT[a]
            t = rng("upd", seed, user, phr, a, n, style).choice(pool)
            blocks.append(("update", [t.format(v=new, old=old, p=proj)],
                           [{"attr": a, "value": new, "kind": "update", "style": style, "old": old}]))

    update_round(1)
    post = list(filler_post)
    if chain == 2:
        for role, text in post[:N_BETWEEN_CHAIN]:
            blocks.append((role, [text], []))
        post = post[N_BETWEEN_CHAIN:]
        update_round(2)
    for role, text in post[:gap]:
        blocks.append((role, [text], []))

    dr = rng("dates", seed, user)
    day = START_DATE
    sessions = []
    for i, (role, sents, fs) in enumerate(blocks):
        if i:
            day += dt.timedelta(days=dr.randint(2, 4))
        wr = rng("wrap", seed, user, phr, role, tuple(sents))
        text = _compose(wr, sents, wrap=role in ("statement", "update") and wr.random() < 0.6)
        sessions.append(Session(i, day.isoformat(), role, text, fs))

    questions = []
    final = {f["attr"]: s.idx for s in sessions for f in s.facts if f["kind"] == "update"}
    for a in F.ATTRIBUTES:
        u = final[a]
        after = u + gap
        qdate = dt.date.fromisoformat(sessions[after].date) + dt.timedelta(days=1)
        qt = rng("q", seed, user, phr, a).choice(F.QUESTION[a]).format(p=proj)
        questions.append(Question(a, qt, after, qdate.isoformat(), chains[a][chain], chains[a][:chain], u))

    return History(user, condition, dict(p), vals["user_name"], proj, chains, near, sessions, questions)


def generate_all(n_users: int, conditions: list[str], seed: int = 0) -> list[History]:
    return [generate(u, c, seed) for c in conditions for u in range(n_users)]


def validate(h: History) -> list[str]:
    """Problems with a history; an empty list means it is valid."""
    errs = []
    for q in h.questions:
        stated = [f["value"] for s in h.sessions[:q.asked_after + 1] for f in s.facts if f["attr"] == q.attr]
        if not stated or stated[-1] != q.current:
            errs.append(f"{q.attr}: last stated value {stated[-1:] } != current {q.current}")
        if len(set(stated)) != len(q.old) + 1:
            errs.append(f"{q.attr}: stated values {set(stated)} do not match chain")
        n_old = sum(1 for s in h.sessions for f in s.facts if f["attr"] == q.attr and f["kind"] == "statement")
        if n_old != h.params["rep"]:
            errs.append(f"{q.attr}: {n_old} statements, expected {h.params['rep']}")
        between = q.asked_after - q.update_session
        if between != h.params["gap"]:
            errs.append(f"{q.attr}: gap {between}, expected {h.params['gap']}")
        if q.asked_after >= len(h.sessions):
            errs.append(f"{q.attr}: asked after session {q.asked_after} which does not exist")
    # probe values may appear only in sessions that carry a fact about that attribute
    for a in F.ATTRIBUTES:
        for v in h.chains[a][:h.params["chain"] + 1]:
            for s in h.sessions:
                if v in s.text and not any(f["attr"] == a for f in s.facts):
                    errs.append(f"value {v!r} of {a} leaks into session {s.idx}")
        if set(h.near[a]) & set(h.chains[a]):
            errs.append(f"{a}: near-miss values overlap the chain")
    return errs
