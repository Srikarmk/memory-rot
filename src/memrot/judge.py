"""Deterministic judge.

Each attribute value is a distinctive string, so the judge finds which of the
attribute's values (current and old) an answer mentions. When both appear it
decides from nearby cue words whether each is presented as current or as past.

Outcomes:
  correct  the current value is given as the answer (an old value may appear,
           framed as past: "you moved from Denver to Boston")
  stale    an old value is given as the answer
  hedged   both appear and neither is clearly the current one
  neither  no value of the attribute appears (or only old ones framed as past)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import facts as F

MONTH_NUM = {m: i + 1 for i, m in enumerate(F.MONTHS)}
DIET_PATTERNS = {
    "gluten-free": r"gluten[- ]?free", "dairy-free": r"dairy[- ]?free", "keto": r"\bketo(?:genic)?\b",
    "pescatarian": r"\bpesc[ae]tarian\b", "vegetarian": r"\bvegetarian\b", "halal": r"\bhalal\b",
    "kosher": r"\bkosher\b",
}


def patterns(attr: str, value: str) -> list[str]:
    if attr == "manager":
        first, last = value.split(" ", 1)
        return [rf"\b{re.escape(value)}\b", rf"\b{re.escape(first)}\b", rf"\b{re.escape(last)}\b"]
    if attr == "deadline":
        month, day = value.split(" ")
        m, ab, n = re.escape(month), re.escape(F.MONTH_ABBR[month]), MONTH_NUM[month]
        sfx = r"(?:st|nd|rd|th)?(?!\d)"
        return [rf"\b{m}\s+{day}{sfx}", rf"\b{ab}\.?\s+{day}{sfx}", rf"\b{day}{sfx}\s+(?:of\s+)?{m}\b",
                rf"(?<![\d/]){n}/{day}(?![\d])", rf"(?<![\d/])0{n}/{int(day):02d}(?![\d])"]
    if attr == "diet":
        return [DIET_PATTERNS[value]]
    if attr == "meeting_day":
        return [rf"\b{re.escape(value)}s?\b"]
    return [rf"\b{re.escape(value)}\b"]


def find_mentions(attr: str, value: str, text: str) -> list[tuple[int, int]]:
    spans = []
    for p in patterns(attr, value):
        spans += [(m.start(), m.end()) for m in re.finditer(p, text, re.I)]
    spans.sort(key=lambda x: (x[0], -x[1]))
    merged = []
    for s in spans:
        if merged and s[0] < merged[-1][1]:
            continue
        merged.append(s)
    return merged


# Clause boundaries: sentence ends, semicolons, brackets, dashes and contrastive joins.
BOUNDARY = re.compile(r"[.;!?()—]|\s-\s|,?\s+(?=but\b|however\b|whereas\b|although\b|while\b)"
                      r"|,\s+(?=not\b|now\b|and now\b|instead\b)", re.I)
PAST_WORDS = {"previously", "formerly", "former", "originally", "earlier", "before", "prior", "old",
              "from", "not", "no", "never", "was", "were", "lived", "reported", "followed", "ate", "met",
              "replaced", "past", "anymore", "longer", "until", "had", "used", "previous", "stopped"}
CURRENT_WORDS = {"now", "currently", "new", "latest", "is", "are", "'s", "today", "recent", "updated",
                 "moved", "switched", "changed", "relocated", "pushed", "rescheduled", "migrated",
                 "moving", "switching", "changing"}
STRONG_CURRENT = {"still", "currently", "now", "presently"}
CHANGE_VERBS = {"moved", "switched", "changed", "relocated", "pushed", "rescheduled", "migrated", "moving",
                "switching", "changing", "updated", "shifted"}
POST_PAST = {"before", "previously", "earlier", "anymore", "originally", "was", "were", "previous",
             "former", "formerly", "prior"}
TOKEN = re.compile(r"n't|'s|[a-z]+", re.I)


def _clause(text: str, start: int, end: int) -> tuple[str, str]:
    left = 0
    for m in BOUNDARY.finditer(text, 0, start):
        left = m.end()
    m = BOUNDARY.search(text, end)
    right = m.start() if m else len(text)
    return text[left:start], text[end:right]


def framing(text: str, span: tuple[int, int]) -> str:
    """'past', 'current', 'strong' (current, marked by still/currently/now) or 'change'
    (reported as the value something changed to) for one mention, from cue words in its clause."""
    pre, post = _clause(text, *span)
    toks = [t.lower() for t in TOKEN.findall(pre)]
    has_from = "from" in toks
    for i in range(len(toks) - 1, max(-1, len(toks) - 9), -1):
        t = toks[i]
        if t == "to":
            if i > 0 and toks[i - 1] == "used":
                return "past"
            if has_from or (i > 0 and toks[i - 1] in CHANGE_VERBS):
                return "change"
            if i > 0 and toks[i - 1] in CURRENT_WORDS:
                return "current"
            continue
        if t == "n't":
            return "past"
        if t == "used" and i + 1 < len(toks) and toks[i + 1] == "to":
            return "past"
        if t in PAST_WORDS:
            if t == "had" or t == "used":
                continue
            if t == "no" and not (i + 1 < len(toks) and toks[i + 1] == "longer"):
                continue
            return "past"
        if t in STRONG_CURRENT:
            return "strong"
        if t in CURRENT_WORDS:
            return "current"
    ptoks = [t.lower() for t in TOKEN.findall(post)][:6]
    if any(t in POST_PAST for t in ptoks) or "n't" in ptoks:
        return "past"
    return "current"


CHANGE = re.compile(r"\b(moved|relocated|changed|switched|updated|rescheduled|pushed|migrated|no longer|"
                    r"used to|previously|formerly|anymore|instead of|recently|new (?:manager|boss|deadline|"
                    r"place|city|app|diet|day))\b", re.I)


STRONG_PAST = re.compile(r"no longer|anymore|used to|previously|formerly|n't|\bnot\b|(?:moved|switched|changed)"
                         r" (?:away )?from", re.I)


@dataclass
class Verdict:
    outcome: str
    mentions_current: bool
    mentions_old: bool
    change_noted: bool
    framing: dict


def judge(attr: str, answer: str, current: str, old: list[str]) -> Verdict:
    cur = [(s, framing(answer, s)) for s in find_mentions(attr, current, answer)]
    olds = [(v, s, framing(answer, s)) for v in old for s in find_mentions(attr, v, answer)]
    mc, mo = bool(cur), bool(olds)
    old_all_past = all(f == "past" for _, _, f in olds)
    live = ("current", "change", "strong")
    cur_any_current = any(f in live for _, f in cur)
    # "it is on Monday, but it was rescheduled to Thursday": the change wins, unless the old
    # value is also asserted as still holding ("you still live in Denver")
    if any(f == "change" for _, f in cur) and not any(f in ("change", "strong") for _, _, f in olds):
        old_all_past = True
    if mc and not mo:
        out = "correct"
    elif mo and not mc:
        # an old value only, but explicitly negated or marked as past: the answer knows it changed
        out = "neither" if old_all_past and STRONG_PAST.search(answer) else "stale"
    elif mc and mo:
        if old_all_past and cur_any_current:
            out = "correct"
        elif not cur_any_current and any(f in live for _, _, f in olds):
            out = "stale"
        else:
            out = "hedged"
    else:
        out = "neither"
    noted = (mo and any(f == "past" for _, _, f in olds)) or bool(CHANGE.search(answer))
    fr = {"current": [f for _, f in cur], "old": [f"{v}:{f}" for v, _, f in olds]}
    return Verdict(out, mc, mo, noted, fr)


def context_has(attr: str, items: list[str], value: str) -> bool:
    return any(find_mentions(attr, value, t) for t in items)
