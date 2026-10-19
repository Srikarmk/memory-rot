from conftest import FakeClient

from memrot.history import History, Question, Session
from memrot.memory import (Entry, answer_messages, build, memory_block, parse_facts, parse_numbers,
                           retrieve, rule_extract)


def tiny_history():
    texts = ["I live in Denver.", "How do I boil an egg?", "My sister lives in Phoenix.",
             "I moved to Boston last weekend.", "I keep my notes in Obsidian."]
    sessions = [Session(i, f"2026-01-{i + 1:02d}", "x", t, []) for i, t in enumerate(texts)]
    q = Question("city", "Which city do I live in?", 4, "2026-01-06", "Boston", ["Denver"], 3)
    return History(0, "base", {}, "Avery", "Halcyon", {}, {}, sessions, [q])


def test_rule_extract_keeps_first_person_sentences():
    assert rule_extract("Hi again. I live in Denver. What is a tuple?") == ["I live in Denver."]


def test_parsers():
    assert parse_facts("- The user lives in Denver.\n2. The user eats keto.\nNONE") == [
        "The user lives in Denver.", "The user eats keto."]
    assert parse_facts("NONE") == []
    assert parse_numbers("2, 5", 4) == [2]
    assert parse_numbers("NONE", 4) == []


def test_append_store_keeps_everything(emb):
    b = build(tiny_history(), FakeClient(), emb, writer="w")
    assert [e.session for e in b.llm] == [0, 2, 3, 4]
    assert all(e.removed_at is None for e in b.llm)
    assert [e.text for e in b.rule] == ["I live in Denver.", "How do I boil an egg?",
                                        "My sister lives in Phoenix.", "I moved to Boston last weekend.",
                                        "I keep my notes in Obsidian."]


def test_overwrite_deletes_named_entries(emb):
    def reply(prompt):
        lines = prompt.split("Existing memories:\n", 1)[1].split("\n\n", 1)[0].splitlines()
        new = prompt.split("latest conversation:\n", 1)[1]
        if "Boston" in new:
            return ",".join(l.split(".")[0] for l in lines if "Denver" in l)
        return "NONE"

    b = build(tiny_history(), FakeClient(reply), emb, writer="w")
    denver = [e for e in b.overwrite if "Denver" in e.text][0]
    assert denver.removed_at == 3
    assert denver.active(2) and not denver.active(3)
    assert all(e.removed_at is None for e in b.overwrite if "Denver" not in e.text)


def test_summary_is_rewritten_each_session(emb):
    b = build(tiny_history(), FakeClient(), emb, writer="w")
    assert len(b.summaries) == 5
    assert b.summaries[0] == "I live in Denver."
    assert "Boston" in b.summaries[3] and "Obsidian" not in b.summaries[3]


def test_retrieve_similarity_and_snapshot(emb):
    es = [Entry(0, "The user lives in Denver.", 0, "d0"), Entry(1, "The user keeps notes in Obsidian.", 1, "d1"),
          Entry(2, "The user lives in Boston.", 5, "d5")]
    q = emb.query("Where the user lives")
    assert [e.eid for e in retrieve(es, 9, q, emb, k=2)][0] in (0, 2)
    assert {e.eid for e in retrieve(es, 9, q, emb, k=2)} == {0, 2}
    assert [e.eid for e in retrieve(es, 4, q, emb, k=3)] == [0, 1]  # entry 2 is in the future
    es[0].removed_at = 5
    assert 0 not in [e.eid for e in retrieve(es, 9, q, emb, k=3)]


def test_recency_breaks_toward_newer(emb):
    es = [Entry(0, "The user lives in Denver.", 0, "d0"), Entry(1, "The user lives in Boston.", 8, "d8"),
          Entry(2, "The user likes tea.", 9, "d9")]
    q = emb.query("Where the user lives")
    top = retrieve(es, 10, q, emb, k=1, recency=True)
    assert top[0].eid == 1


def test_answer_prompts(emb):
    h = tiny_history()
    b = build(h, FakeClient(), emb, writer="w")
    q = h.questions[0]
    blk, items = memory_block("b", h, b, q, emb, 5)
    assert "[2026-01-01]" in blk and len(items) == 4
    blk, items = memory_block("f", h, b, q, emb, 5)
    assert len(items) == 5 and "Obsidian" in blk
    blk, _ = memory_block("e", h, b, q, emb, 5)
    assert "Obsidian" in blk
    msgs, _ = answer_messages("a", h, b, q, emb, 2)
    assert msgs[1]["content"] == q.text and "Today is 2026-01-06" in msgs[0]["content"]
    assert msgs[0]["content"].count("\n- ") == 2
