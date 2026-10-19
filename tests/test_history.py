from memrot import facts as F
from memrot.history import CONDITIONS, generate, validate


def test_reproducible():
    for c in CONDITIONS:
        assert generate(3, c).to_dict() == generate(3, c).to_dict()


def test_all_valid():
    for c in CONDITIONS:
        for u in range(12):
            assert validate(generate(u, c)) == [], (u, c)


def test_current_is_last_stated_value():
    for c in CONDITIONS:
        h = generate(1, c)
        for q in h.questions:
            seen = [f["value"] for s in h.sessions[:q.asked_after + 1] for f in s.facts if f["attr"] == q.attr]
            assert seen[-1] == q.current
            assert q.current not in q.old
            assert set(seen) == set(q.old) | {q.current}
            assert q.current in h.sessions[q.update_session].text


def test_factor_levels():
    for c, p in CONDITIONS.items():
        h = generate(0, c)
        for q in h.questions:
            n = sum(1 for s in h.sessions for f in s.facts if f["attr"] == q.attr and f["kind"] == "statement")
            assert n == p["rep"]
            assert q.asked_after - q.update_session == p["gap"]
            assert len(q.old) == p["chain"]


def test_gap_conditions_share_a_prefix():
    g0, base, g20 = generate(2, "gap0"), generate(2, "base"), generate(2, "gap20")
    assert [s.text for s in g0.sessions] == [s.text for s in base.sessions[:len(g0.sessions)]]
    assert [s.text for s in base.sessions] == [s.text for s in g20.sessions[:len(base.sessions)]]


def test_one_factor_at_a_time():
    base, imp = generate(4, "base"), generate(4, "implied")
    assert base.chains == imp.chains
    upd = [s for s in imp.sessions if s.role == "update"]
    assert all(f["style"] == "implied" for s in upd for f in s.facts)
    # implied updates are ordinary statements: none of the explicit templates' change words
    for s in upd:
        assert not any(w in s.text.lower() for w in ("moved", "switched", "changed", "new manager", "anymore"))
    assert [s.text for s in base.sessions[:10]] == [s.text for s in imp.sessions[:10]]


def test_aa_rewords_but_keeps_values():
    base, aa = generate(5, "aa"), generate(5, "base")
    assert base.chains == aa.chains
    assert [s.text for s in base.sessions] != [s.text for s in aa.sessions]
    assert [q.current for q in base.questions] == [q.current for q in aa.questions]


def test_users_differ_and_values_are_distinct_strings():
    h0, h1 = generate(0, "base"), generate(1, "base")
    assert h0.chains != h1.chains
    for a in F.ATTRIBUTES:
        vals = h0.chains[a] + h0.near[a]
        assert len(set(vals)) == len(vals)
