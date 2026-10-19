"""reports/results.md and reports/figures/, generated from results/ only."""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .history import CONDITIONS, FACTOR  # noqa: E402
from .judge import judge  # noqa: E402
from .memory import SYSTEM_LABEL, SYSTEMS  # noqa: E402
from .stats import exceeds, mean_ci, paired_diff  # noqa: E402

OUTCOMES = ["correct", "hedged", "stale", "neither"]
SYS_COLOR = {"a": "#2a78d6", "b": "#eb6834", "c": "#1baf7a", "d": "#eda100", "e": "#e87ba4", "f": "#008300",
             "a_rule": "#6250d6"}
OUT_COLOR = {"correct": "#2a78d6", "hedged": "#eda100", "stale": "#eb6834", "neither": "#b5b3ad"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
PLOT_SYSTEMS = ["a", "b", "c", "d", "e", "f"]


def load_answers() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(C.RESULTS, "answers", "*.jsonl"))):
        with open(p) as f:
            rows += [json.loads(line) for line in f]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df.user.isin(C.USERS)].copy()
    # re-judge with the current judge, so every stored answer is scored by the same rules
    verdicts = [judge(r.attr, r.answer, r.current, r.old) for r in df.itertuples()]
    df["outcome"] = [v.outcome for v in verdicts]
    df["change_noted"] = [v.change_noted for v in verdicts]
    for o in OUTCOMES:
        df[o] = (df.outcome == o).astype(float)
    df["noted"] = df.change_noted.astype(float)
    return df


def load_costs() -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(C.RESULTS, "builds", "*.json")):
        with open(p) as f:
            b = json.load(f)
        for c in b["costs"]:
            rows.append({"hid": b["hid"], "condition": b["hid"].split("-", 1)[1], **c})
    return pd.DataFrame(rows)


def main_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep = (df.tag == "") & ((df.k == C.K_DEFAULT) | (df.k == 0))
    return df[keep]


def models_in(df: pd.DataFrame) -> list[str]:
    return [m for m in C.MODELS if m in set(df.model)]


def mlabel(m: str) -> str:
    return f"{m} (thinking)" if m in C.THINKING_ONLY else m


def pct(x: float) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:.0f}%"


def pp(d: dict) -> str:
    if not d or np.isnan(d["diff"]):
        return "n/a"
    return f"{100 * d['diff']:+.0f} pp [{100 * d['lo']:+.0f}, {100 * d['hi']:+.0f}]"


def ci(t: tuple) -> str:
    return f"{pct(t[0])} [{pct(t[1])}, {pct(t[2])}]"


def cell(df, model, system, condition):
    return df[(df.model == model) & (df.system == system) & (df.condition == condition)]


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out + [""]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_factor(df, models, conds, levels, xlabel, name, title):
    models = [m for m in models if all(len(cell(df, m, "a", c)) for c in conds)]
    fig, axes = plt.subplots(1, len(models), figsize=(max(3.2 * len(models) + 0.6, 7.5), 3.6), sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(axes[0], models):
        style_axes(ax)
        for s in PLOT_SYSTEMS:
            ys = [cell(df, m, s, c).stale.mean() if len(cell(df, m, s, c)) else np.nan for c in conds]
            ax.plot(range(len(conds)), ys, color=SYS_COLOR[s], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=SYSTEM_LABEL[s])
        ax.set_xticks(range(len(conds)), levels)
        ax.set_title(mlabel(m), fontsize=9, color=INK)
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
        ax.set_ylim(-0.02, 1.0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0][0].set_ylabel("stale answers", fontsize=8, color=INK2)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, frameon=False, labelcolor=INK2)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(os.path.join(C.FIGURES, name), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_fixes(df, models, name):
    systems = ["a", "b", "c", "d", "e", "f"]
    fig, axes = plt.subplots(2, len(models), figsize=(3.0 * len(models) + 1.2, 4.6), sharex=True,
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, m in enumerate(models):
        for i, metric in enumerate(["stale", "correct"]):
            ax = axes[i][j]
            style_axes(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.6)
            ax.grid(axis="y", visible=False)
            for y, s in enumerate(systems):
                d = cell(df, m, s, "base")
                if d.empty:
                    continue
                mu, lo, hi = mean_ci(d, metric)
                ax.plot([lo, hi], [y, y], color=SYS_COLOR[s], linewidth=2, solid_capstyle="round")
                ax.plot([mu], [y], "o", color=SYS_COLOR[s], markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.5)
            ax.set_yticks(range(len(systems)), [SYSTEM_LABEL[s] for s in systems])
            ax.invert_yaxis()
            ax.set_xlim(-0.02, 1.02)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            if i == 0:
                ax.set_title(mlabel(m), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{metric} answers", fontsize=8, color=INK2)
    fig.suptitle("Baseline histories: share of stale and correct answers per memory system (95% CI)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(C.FIGURES, name), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_outcomes(df, models, name):
    systems = ["a_rule", "a", "b", "c", "d", "e", "f"]
    fig, axes = plt.subplots(1, len(models), figsize=(3.0 * len(models) + 1.4, 3.0), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(axes[0], models):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for y, s in enumerate(systems):
            d = cell(df, m, s, "base")
            left = 0.0
            for o in OUTCOMES:
                w = d[o].mean() if len(d) else 0
                ax.barh(y, w, left=left, color=OUT_COLOR[o], height=0.7, edgecolor=SURFACE, linewidth=1,
                        label=o if y == 0 else None)
                left += w
        ax.set_yticks(range(len(systems)), [SYSTEM_LABEL[s] for s in systems])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(mlabel(m), fontsize=9, color=INK)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7, frameon=False, labelcolor=INK2)
    fig.suptitle("Baseline histories: outcome of every answer", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(os.path.join(C.FIGURES, name), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_cost(cost_rows, name):
    systems = [r["system"] for r in cost_rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, lab in [(axes[0], "write_tokens", "writer tokens per session (prompt + output)"),
                         (axes[1], "read_tokens", "answer prompt tokens per question")]:
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for y, r in enumerate(cost_rows):
            ax.barh(y, r[key], color=SYS_COLOR[r["system"]], height=0.6)
            ax.text(r[key], y, f" {r[key]:.0f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(range(len(systems)), [SYSTEM_LABEL[s] for s in systems])
        ax.invert_yaxis()
        ax.set_xlabel(lab, fontsize=8, color=INK2)
    fig.suptitle("What each memory system costs (baseline histories, qwen3:8b answering)", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(C.FIGURES, name), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def store_contents(condition: str) -> pd.DataFrame:
    from .history import generate
    from .judge import context_has
    from .memory import Build
    rows = []
    for u in C.USERS:
        h = generate(u, condition)
        p = os.path.join(C.RESULTS, "builds", f"{h.hid}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            b = Build.from_dict(json.load(f))
        for q in h.questions:
            t = q.asked_after
            stores = {"rule (append)": [e.text for e in b.rule if e.active(t)],
                      "llm (append)": [e.text for e in b.llm if e.active(t)],
                      "overwrite": [e.text for e in b.overwrite if e.active(t)],
                      "summary": [b.summaries[t]]}
            for name, items in stores.items():
                rows.append({"user": u, "attr": q.attr, "store": name,
                             "has_current": context_has(q.attr, items, q.current),
                             "has_old": any(context_has(q.attr, items, o) for o in q.old)})
    return pd.DataFrame(rows)


def cost_table(df: pd.DataFrame, costs: pd.DataFrame, model: str) -> list[dict]:
    base_c = costs[costs.condition == "base"]
    n_sessions = base_c.groupby("hid").session.max().add(1).sum() if not base_c.empty else np.nan
    kinds = {"a_rule": [], "a": ["extract"], "b": ["extract"], "c": ["extract", "overwrite"], "d": ["extract"],
             "e": ["summary"], "f": []}
    rows = []
    for s in SYSTEMS:
        w = base_c[base_c.kind.isin(kinds[s])]
        r = cell(df, model, s, "base")
        rows.append({
            "system": s,
            "write_calls": len(w) / n_sessions if n_sessions else np.nan,
            "write_tokens": (w.prompt_tokens.sum() + w.output_tokens.sum()) / n_sessions if n_sessions else np.nan,
            "write_seconds": w.seconds.sum() / n_sessions if n_sessions else np.nan,
            "read_tokens": r.prompt_tokens.mean() if len(r) else np.nan,
            "read_seconds": r.seconds.mean() if len(r) else np.nan,
        })
    return rows


def write_report() -> None:
    os.makedirs(C.FIGURES, exist_ok=True)
    df_all = load_answers()
    if df_all.empty:
        print("no answers yet")
        return
    df = main_rows(df_all)
    costs = load_costs()
    models = models_in(df)
    meta = {}
    if os.path.exists(os.path.join(C.RESULTS, "models.json")):
        with open(os.path.join(C.RESULTS, "models.json")) as f:
            meta = json.load(f)
    n_users = df.user.nunique()
    L = ["# Results", "",
         "Generated by `uv run memrot report` from the cached runs in `results/`. Every number below comes from "
         "that code. Intervals are 95% percentile bootstrap intervals that resample users (each user is one "
         f"history family with six probed facts), {C.BOOT} draws. pp = percentage points.", ""]

    # coverage
    L += ["## Runs", ""]
    rows = []
    for m in models:
        d = df[df.model == m]
        conds = [c for c in CONDITIONS if c in set(d.condition)]
        info = meta.get(m, {})
        rows.append([m, info.get("digest", "")[:12], d.user.nunique(), len(d), ", ".join(conds),
                     ", ".join(s for s in SYSTEMS if s in set(d.system))])
    L += md_table(["answering model", "digest", "users", "answers", "conditions", "systems"], rows)
    for m in models:
        if m in C.THINKING_ONLY:
            d = df[df.model == m]
            trunc = int(d.truncated.fillna(False).sum()) if "truncated" in d else 0
            L += [f"`{m}` is the thinking-only Qwen3 release in this Ollama install: `think: false` does not "
                  f"stop its reasoning, so it runs with `think: true` and `num_predict` {C.THINKING_NUM_PREDICT}, "
                  f"and only its final answer is judged. It is a reasoning model here, unlike the other three. "
                  f"Answers cut off by the length cap: {trunc} of {len(d)}. Its grid is smaller (time).", ""]
    if os.path.exists(C.CACHE_DB):
        import sqlite3
        db = sqlite3.connect(C.CACHE_DB)
        tot = db.execute("select model, count(*), sum(json_extract(response, '$.seconds')), "
                         "sum(json_extract(response, '$.wall_seconds')) from calls group by model").fetchall()
        L += ["Model calls in the cache (including the excluded pilot users), with server compute time and "
              "wall time per call summed (wall time includes queueing behind the other workload sharing the "
              "GPU):", ""]
        L += md_table(["model", "calls", "compute hours", "wall hours"],
                      [[m, n, f"{(sec or 0) / 3600:.2f}", f"{(w or 0) / 3600:.2f}"] for m, n, sec, w in tot])
    L += [f"Writer (memory extraction, overwrite decisions, summaries) for every store: `{C.WRITER}`. "
          f"Options for every call: `{json.dumps(C.BASE_OPTIONS)}` plus a per-call `num_predict` "
          f"(`{json.dumps(C.NUM_PREDICT)}`). `think: false` is sent to qwen3:8b (and `think: true` to the thinking-only qwen3:30b-a3b, see above); other models get no think "
          f"field. Embeddings: `{C.EMBED_MODEL}`. Retrieval k = {C.K_DEFAULT} unless stated.", ""]

    # headline table
    L += ["## Headline numbers", "",
          "Stale rate (share of questions answered with an old value), 95% CI over users. Append = the plain "
          "append-only vector store (system a).", ""]
    rows = []
    for m in models:
        def st(sys_, cond):
            d = cell(df, m, sys_, cond)
            return ci(mean_ci(d, "stale")) if len(d) else "n/a"
        rows.append([m, st("a", "base"), st("a", "rep1"), st("a", "rep10"), st("a", "gap0"), st("a", "gap20"),
                     st("b", "base"), st("c", "base"), st("d", "base"), st("e", "base"), st("f", "base"),
                     st("f", "implied")])
    L += md_table(["model", "append, base", "append, stated 1x", "append, stated 10x", "append, gap 0",
                   "append, gap 20", "timestamps", "overwrite", "recency", "summary", "full context",
                   "full context, implied update"], rows)

    # headline
    L += ["## 1. Baseline: which answer comes back?", "",
          "Baseline histories: each old value stated 3 times, one explicit update, 5 sessions between the update "
          "and the question.", ""]
    for m in models:
        rows = []
        for s in SYSTEMS:
            d = cell(df, m, s, "base")
            if d.empty:
                continue
            rows.append([SYSTEM_LABEL[s], ci(mean_ci(d, "correct")), ci(mean_ci(d, "stale")), pct(d.hedged.mean()),
                         pct(d.neither.mean()), pct(d.noted.mean()), len(d)])
        L += [f"**{m}**", ""] + md_table(["memory", "correct", "stale", "hedged", "neither", "mentions change", "n"],
                                         rows)
    fig_outcomes(df, models, "outcomes_base.png")
    L += ["![outcomes](figures/outcomes_base.png)", ""]

    # A/A
    L += ["## 2. Noise floor", "",
          "Two A/A comparisons. **Rewording**: the baseline design regenerated with a different phrasing seed and "
          "filler draw (same users, same values, same schedule); every effect below is compared with this gap. "
          "**Rerun**: the baseline questions asked again with the cache bypassed (same prompts, same seed), which "
          "measures server nondeterminism at temperature 0.", ""]
    aa = {}
    rows = []
    for m in models:
        for s in SYSTEMS:
            b, a2 = cell(df, m, s, "base"), cell(df, m, s, "aa")
            if b.empty or a2.empty:
                continue
            aa[(m, s, "stale")] = paired_diff(a2, b, "stale")
            aa[(m, s, "correct")] = paired_diff(a2, b, "correct")
            aa[(m, s, "neither")] = paired_diff(a2, b, "neither")
            rr = df_all[(df_all.tag == "rerun") & (df_all.model == m) & (df_all.system == s)
                        & (df_all.condition == "base") & ((df_all.k == C.K_DEFAULT) | (df_all.k == 0))]
            rerun = "n/a"
            if len(rr):
                mm = rr.merge(b, on=["user", "attr"], suffixes=("_r", ""))
                rerun = f"{(mm.answer_r != mm.answer).mean():.0%} answers differ, " \
                        f"{(mm.outcome_r != mm.outcome).mean():.0%} outcomes differ"
            rows.append([m, SYSTEM_LABEL[s], pp(aa[(m, s, "stale")]), pp(aa[(m, s, "correct")]), rerun])
    L += md_table(["model", "memory", "rewording gap, stale", "rewording gap, correct", "rerun"], rows)

    def effects_section(title, conds, note, metrics=("stale", "correct")):
        out = [title, "", note, ""]
        for metric in metrics:
            rows = []
            for m in models:
                for s in SYSTEMS:
                    b = cell(df, m, s, "base")
                    if b.empty:
                        continue
                    r = [m, SYSTEM_LABEL[s], pct(b[metric].mean())]
                    for c in conds:
                        x = cell(df, m, s, c)
                        if x.empty:
                            r += ["n/a", "n/a"]
                            continue
                        e = paired_diff(x, b, metric)
                        flag = " *" if exceeds(e, aa.get((m, s, metric))) else ""
                        r += [pct(x[metric].mean()), pp(e) + flag]
                    gap = aa.get((m, s, metric))
                    r.append(pp(gap) if gap else "n/a")
                    rows.append(r)
            hdr = ["model", "memory", f"base {metric}"]
            for c in conds:
                hdr += [f"{FACTOR[c][1]}", f"{FACTOR[c][1]} minus base"]
            hdr.append("A/A gap")
            out += [f"*{metric} rate*", ""] + md_table(hdr, rows)
        out += ["`*` marks an effect whose interval excludes zero and whose size exceeds the rewording A/A gap "
                "for the same model and memory. Where both rates sit near zero the A/A gap is often exactly 0, "
                "so small effects (around 5 pp) can be marked; read those as weak.", ""]
        return out

    # pre-registered primary contrasts
    L += ["## Primary contrasts (fixed in HYPOTHESES.md)", "",
          "Stale rate. Repetition: 10 statements minus 1. Recency: 20 sessions since the update minus 0. "
          "Paired by user and attribute.", ""]
    rows = []
    for m in models:
        for s in SYSTEMS:
            r = [m, SYSTEM_LABEL[s]]
            for hi_c, lo_c in [("rep10", "rep1"), ("gap20", "gap0")]:
                x, y = cell(df, m, s, hi_c), cell(df, m, s, lo_c)
                if x.empty or y.empty:
                    r += ["n/a"]
                    continue
                e = paired_diff(x, y, "stale")
                r += [pp(e) + (" *" if exceeds(e, aa.get((m, s, "stale"))) else "")]
            r.append(pp(aa[(m, s, "stale")]) if (m, s, "stale") in aa else "n/a")
            if any(v != "n/a" for v in r[2:4]):
                rows.append(r)
    L += md_table(["model", "memory", "rep10 minus rep1", "gap20 minus gap0", "A/A gap"], rows)

    L += effects_section("## 3. Repetition: how often the old value was stated", ["rep1", "rep10"],
                         "Old value stated 1, 3 (base) or 10 times before the update; everything else as base.")
    fig_factor(df, models, ["rep1", "base", "rep10"], ["1", "3", "10"], "times the old value was stated",
               "stale_by_repetition.png", "Stale answers by how often the old value was stated")
    L += ["![repetition](figures/stale_by_repetition.png)", ""]
    L += effects_section("## 4. Recency: sessions between the update and the question", ["gap0", "gap20"],
                         "0, 5 (base) or 20 sessions between the update and the question.",
                         metrics=("stale", "correct", "neither"))
    fig_factor(df, models, ["gap0", "base", "gap20"], ["0", "5", "20"], "sessions since the update",
               "stale_by_recency.png", "Stale answers by sessions since the update")
    L += ["![recency](figures/stale_by_recency.png)", ""]
    L += effects_section("## 5. Update style and update chains", ["implied", "chain2"],
                         "Implied: the new value is mentioned the way an established fact would be "
                         "(\"the commute here in Austin\"), with nothing saying it changed. Chain: two updates "
                         "(A to B to C); stale counts either older value.")

    # fixes
    L += ["## 6. Which memory designs fix it, and what they cost", "",
          "Each system compared with the plain append-only vector store (a) on the same baseline questions, "
          "paired by question.", ""]
    rows = []
    for m in models:
        a = cell(df, m, "a", "base")
        for s in SYSTEMS:
            if s == "a":
                continue
            x = cell(df, m, s, "base")
            if x.empty or a.empty:
                continue
            es, ec = paired_diff(x, a, "stale"), paired_diff(x, a, "correct")
            rows.append([m, SYSTEM_LABEL[s], pp(es), pp(ec)])
    L += md_table(["model", "memory", "stale minus append", "correct minus append"], rows)
    fig_fixes(df, models, "fixes_base.png")
    L += ["![fixes](figures/fixes_base.png)", ""]
    ref_model = C.MODELS[0] if C.MODELS[0] in models else models[0]
    crow = cost_table(df, costs, ref_model)
    L += [f"Cost per system (baseline histories). Write side: the writer's calls per session; tokens and "
          f"server compute seconds come from Ollama's counters. Read side: the answer prompt for `{ref_model}`. "
          "Seconds were measured while another workload shared the GPU, so compare them across systems, not "
          "with other hardware.", ""]
    L += md_table(["memory", "writer calls / session", "writer tokens / session", "writer s / session",
                   "answer prompt tokens", "answer s"],
                  [[SYSTEM_LABEL[r["system"]], f"{r['write_calls']:.2f}", f"{r['write_tokens']:.0f}",
                    f"{r['write_seconds']:.2f}", f"{r['read_tokens']:.0f}", f"{r['read_seconds']:.2f}"] for r in crow])
    fig_cost(crow, "cost.png")
    L += ["![cost](figures/cost.png)", ""]

    # k
    L += ["## 7. Retrieval depth k (append store, baseline)", ""]
    rows = []
    for m in models:
        d = df_all[(df_all.tag == "") & (df_all.model == m) & (df_all.system == "a") & (df_all.condition == "base")]
        for k in sorted(d.k.unique()):
            x = d[d.k == k]
            rows.append([m, k, ci(mean_ci(x, "stale")), ci(mean_ci(x, "correct")), pct(x.ctx_current.mean()),
                         pct(x.ctx_old.mean()), f"{x.prompt_tokens.mean():.0f}"])
    L += md_table(["model", "k", "stale", "correct", "current value retrieved", "old value retrieved",
                   "prompt tokens"], rows)

    # diagnostics
    L += ["## 8. Retrieval or reading? What the answering model saw", "",
          "For the vector stores, whether the retrieved entries contained the current value, an old value, or "
          "both, and what the model answered in each case (baseline questions; the retrieval is identical across "
          "models, the answers are not).", ""]
    rows = []
    vec = df[(df.condition == "base") & df.system.isin(["a_rule", "a", "b", "c", "d"])].copy()
    vec["ctx"] = np.select([vec.ctx_current & vec.ctx_old, vec.ctx_current, vec.ctx_old],
                           ["both", "current only", "old only"], "neither")
    for (m, s), g in vec.groupby(["model", "system"]):
        for c in ["both", "current only", "old only", "neither"]:
            x = g[g.ctx == c]
            if len(x):
                rows.append([m, SYSTEM_LABEL[s], c, f"{len(x) / len(g):.0%}", pct(x.correct.mean()),
                             pct(x.stale.mean()), pct(x.hedged.mean())])
    L += md_table(["model", "memory", "retrieved", "share of questions", "correct", "stale", "hedged"], rows)

    # store contents at question time (no model calls: read from the builds)
    L += ["## 8b. What the memory held at question time", "",
          "Independent of retrieval and of the answering model: whether the store (all active entries) or the "
          "summary contained the current value and an old value when the question was asked. For the "
          "overwrite store, holding an old value means the writer failed to delete it, and missing the current "
          "value means it deleted or never wrote the new entry.", ""]
    rows = []
    for cond in ["base", "rep10", "implied", "chain2"]:
        st = store_contents(cond)
        if st.empty:
            continue
        for s, g in st.groupby("store", sort=False):
            rows.append([cond, s, pct(g.has_current.mean()), pct(g.has_old.mean()),
                         pct((g.has_old & ~g.has_current).mean()), len(g)])
    L += md_table(["condition", "store", "holds current", "holds an old value", "old only", "questions"], rows)

    # per attribute
    L += ["## 9. By attribute (baseline, append store)", "", "Ten questions per cell, so these are indicative only.",
          ""]
    rows = []
    for m in models:
        d = cell(df, m, "a", "base")
        r = [m] + [pct(d[d.attr == a].stale.mean()) for a in sorted(d.attr.unique())]
        rows.append(r)
    attrs = sorted(df.attr.unique())
    L += md_table(["model"] + [f"stale: {a}" for a in attrs], rows)

    # hand check
    hc = os.path.join(C.ROOT, "handcheck", "agreement.json")
    if os.path.exists(hc):
        with open(hc) as f:
            h = json.load(f)
        L += ["## 10. Judge check", "", h["summary"], ""]
        lab_p = os.path.join(C.ROOT, "handcheck", "labels.csv")
        if os.path.exists(lab_p):
            comp = pd.read_csv(lab_p).model.value_counts()
            L += ["Sampled answers by model: " + ", ".join(f"{m} {n}" for m, n in comp.items())
                  + ". The sample was drawn before most `qwen3:30b-a3b` answers existed.", ""]

    L += ["## Limits", "",
          "- Small local models (8B to 30B, 4-bit quantised) at temperature 0; hosted models may behave differently.",
          "- Synthetic, templated histories about one fictional user each; real conversations are messier and "
          "facts are rarely this clean.",
          f"- {n_users} users x 6 facts per condition; intervals are wide, and effects inside the A/A gap are "
          "not distinguishable from rewording noise.",
          f"- One writer model (`{C.WRITER}`) builds every store, so differences between answering models are "
          "differences in reading the same memory, not in writing it.",
          "- The judge matches value strings and uses cue words to decide which value is presented as current; "
          "its agreement with one blind labelling pass is reported above.", ""]
    with open(os.path.join(C.REPORTS, "results.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote", os.path.join(C.REPORTS, "results.md"))
