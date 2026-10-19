import pytest

from memrot.judge import context_has, find_mentions, judge

CASES = [
    ("city", "You live in Boston.", "correct", False),
    ("city", "You moved from Denver to Boston.", "correct", True),
    ("city", "You live in Denver.", "stale", False),
    ("city", "You used to live in Denver, but now you live in Boston.", "correct", True),
    ("city", "You live in either Denver or Boston.", "hedged", False),
    ("city", "You no longer live in Denver.", "neither", True),
    ("city", "I don't have that information.", "neither", False),
    ("city", "You live in Phoenix.", "neither", False),
    ("manager", "Your manager is Dana; Priya was your previous manager.", "correct", True),
    ("manager", "You report to Priya Okonkwo.", "stale", False),
    ("manager", "Your manager is Whitfield.", "correct", False),
    ("deadline", "Project Quillon is due on August 19th (moved from July 7).", "correct", True),
    ("deadline", "It is due July 17.", "neither", False),
    ("deadline", "It's due on 7 July.", "stale", False),
    ("meeting_day", "Your team meeting is on Thursdays, not Mondays.", "correct", True),
    ("meeting_day", "Your weekly team meeting is on Monday mornings, but it has been rescheduled to Thursday.",
     "correct", True),
    ("meeting_day", "It is on Monday.", "stale", False),
    ("notes_tool", "You use Zettlr before switching to Trilium.", "correct", True),
    ("diet", "You follow a gluten free diet.", "correct", False),
    ("city", "You recently moved to Boston, but you still live in Denver.", "hedged", True),
    ("city", "You currently live in Denver, but you moved to Boston last weekend.", "hedged", True),
    ("notes_tool", "You currently use Trilium, as you changed from Zettlr.", "correct", True),
    ("diet", "You follow a keto diet and also eat gluten-free food.", "hedged", False),
]
CUR = {"city": "Boston", "manager": "Dana Whitfield", "deadline": "August 19", "meeting_day": "Thursday",
       "notes_tool": "Trilium", "diet": "gluten-free"}
OLD = {"city": ["Denver"], "manager": ["Priya Okonkwo"], "deadline": ["July 7"], "meeting_day": ["Monday"],
       "notes_tool": ["Zettlr"], "diet": ["keto"]}


@pytest.mark.parametrize("attr,answer,outcome,noted", CASES)
def test_hand_cases(attr, answer, outcome, noted):
    v = judge(attr, answer, CUR[attr], OLD[attr])
    assert v.outcome == outcome
    assert v.change_noted == noted


def test_chain_old_values():
    v = judge("city", "You live in Austin.", "Boston", ["Denver", "Austin"])
    assert v.outcome == "stale"


def test_date_boundaries():
    assert find_mentions("deadline", "July 7", "due July 17") == []
    assert find_mentions("deadline", "July 7", "due 7/7")
    assert find_mentions("deadline", "July 7", "due Jul. 7th")


def test_context_has():
    assert context_has("city", ["The user lives in Boston."], "Boston")
    assert not context_has("city", ["The user lives in Bostonia."], "Boston")
