"""Hand check of the judge.

`sample` writes handcheck/sample.csv without the judge's verdicts: a simple random
sample of answers, plus an extra sample of answers that mention both the current
and an old value (where the cue-word rules do the work). A person fills in the
`label` column (correct, stale, hedged, neither) and saves it as
handcheck/labels.csv. `score` compares the labels with the judge.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import config as C
from .judge import judge
from .report import load_answers, main_rows

DIR = os.path.join(C.ROOT, "handcheck")
COLS = ["id", "subset", "model", "system", "condition", "attr", "question", "current", "old", "answer"]


def sample(n: int = 50, n_both: int = 25, seed: int = 11) -> None:
    os.makedirs(DIR, exist_ok=True)
    df = main_rows(load_answers()).reset_index(drop=True)
    rand = df.sample(n=min(n, len(df)), random_state=seed).assign(subset="random")
    rest = df.drop(rand.index)
    def mentions_both(r):
        v = judge(r.attr, r.answer, r.current, r.old)
        return v.mentions_current and v.mentions_old

    both = rest[rest.apply(mentions_both, axis=1)]
    both = both.sample(n=min(n_both, len(both)), random_state=seed).assign(subset="both values")
    s = pd.concat([rand, both]).reset_index(drop=True)
    s["id"] = range(1, len(s) + 1)
    s["old"] = s.old.apply(lambda o: " | ".join(o))
    s[COLS].assign(label="").to_csv(os.path.join(DIR, "sample.csv"), index=False)
    print(f"wrote {len(s)} rows to handcheck/sample.csv")


def score() -> dict:
    lab = pd.read_csv(os.path.join(DIR, "labels.csv"))
    lab["old_list"] = lab.old.apply(lambda o: [x.strip() for x in str(o).split("|")])
    lab["judge"] = lab.apply(lambda r: judge(r.attr, r.answer, r.current, r.old_list).outcome, axis=1)
    lab["agree"] = lab.judge == lab.label.str.strip()
    out = {}
    parts = []
    for sub, g in lab.groupby("subset", sort=False):
        out[sub] = {"n": int(len(g)), "agree": int(g.agree.sum()), "rate": float(g.agree.mean())}
        parts.append(f"{sub}: {int(g.agree.sum())}/{len(g)} ({g.agree.mean():.0%})")
    dis = lab[~lab.agree]
    out["disagreements"] = [{"id": int(r.id), "label": r.label, "judge": r.judge, "answer": r.answer}
                            for r in dis.itertuples()]
    conf = pd.crosstab(lab.label, lab.judge)
    out["confusion"] = {str(k): {str(j): int(x) for j, x in v.items()} for k, v in conf.to_dict().items()}
    table = ["| label \\ judge | " + " | ".join(conf.columns) + " |", "|---|" + "---|" * len(conf.columns)]
    table += ["| " + i + " | " + " | ".join(str(int(x)) for x in row) + " |" for i, row in conf.iterrows()]
    out["summary"] = ("Labels against the deterministic judge, made without seeing the judge's verdicts (one pass, no second labeller yet). "
                      + "; ".join(parts) + ". Rows: label, columns: judge.\n\n" + "\n".join(table))
    with open(os.path.join(DIR, "agreement.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(out["summary"])
    return out
