import pytest

from games.game import ACTIONS, N_ACTIONS, bayes_belief, clue_text, intro, question_text


def action(attr, value, negated, source):
    return ACTIONS.index((attr, value, negated, source))


def test_action_count():
    assert N_ACTIONS == 32


def test_prior_is_uniform():
    assert bayes_belief([]) == [0.5] * 4


def test_single_clue_equals_reliability():
    assert bayes_belief([action(0, 0, False, 0)])[0] == pytest.approx(0.9)
    assert bayes_belief([action(0, 1, False, 1)])[0] == pytest.approx(0.4)


def test_negation_is_equivalent():
    assert bayes_belief([action(1, 0, True, 0)]) == bayes_belief([action(1, 0, False, 0)])


def test_contradiction():
    # Alice says red (0.9), Bob says blue (0.6): odds 9 * (2/3) = 6, p = 6/7.
    belief = bayes_belief([action(0, 0, False, 0), action(0, 1, False, 1)])
    assert belief[0] == pytest.approx(6 / 7)
    assert belief[1:] == [0.5] * 3


def test_texts():
    assert "Alice's statements are correct 90% of the time." in intro()
    assert clue_text(action(0, 0, False, 0)) == "\nAlice says the object is red."
    assert clue_text(action(0, 0, True, 1)) == "\nBob says the object is not blue."
    assert question_text(2).endswith("Is the object big or small?\nAnswer: The object is")


def test_templates_keep_the_claim():
    red_by_alice = action(0, 0, False, 0)
    not_blue_by_bob = action(0, 0, True, 1)
    assert clue_text(red_by_alice, 8) == "\nAlice denies that the object is not red."
    assert clue_text(not_blue_by_bob, 8) == "\nBob denies that the object is blue."
    assert clue_text(red_by_alice, 1) == "\nAccording to Alice, the object is red."


def test_trust_theme():
    from games import game
    game.use("trust")
    try:
        assert game.N_ACTIONS == 32 and game.SUBJECT == "the request"
        text = game.intro()
        assert "The audit tool's statements are correct 90% of the time." in text
        assert "accepting an illegal request costs 10" in text
        assert game.clue_text(game.ACTIONS.index((0, 0, False, 1))) == "\nThe user says the request is legal."
        assert game.clue_text(game.ACTIONS.index((0, 0, True, 0)), 8) == "\nThe audit tool denies that the request is illegal."
        assert game.question_text(0).endswith("Is the request legal or illegal?\nAnswer: The request is")
        # accept only above 10/11: one clue from the tool (0.9) is not enough, two are
        tool_says_legal = game.ACTIONS.index((0, 0, False, 0))
        assert game.decision_threshold() == pytest.approx(10 / 11)
        assert game.bayes_decision([]) == 0
        assert game.bayes_decision([tool_says_legal]) == 0
        assert game.bayes_decision([tool_says_legal] * 2) == 1
        game.use("trust_reversed")
        assert game.SOURCES["the user"] == 0.9
    finally:
        game.use("clue")
    assert game.bayes_decision([]) is None
    assert clue_text(action(0, 0, False, 0)) == "\nAlice says the object is red."


def test_stated_prior():
    # Prior 0.8 red, then Alice says blue (0.9): odds 4 / 9, p = 4/13.
    assert bayes_belief([action(0, 1, False, 0)], [0.8, 0.5, 0.5, 0.5])[0] == pytest.approx(4 / 13)
    assert bayes_belief([], [0.3, 0.5, 0.5, 0.9]) == pytest.approx([0.3, 0.5, 0.5, 0.9])
    assert "chances" not in intro() and "50% each" in intro([0.5] * 4)
    assert "the chances are: red 70%, blue 30%; round 50%, square 50%; big 20%, small 80%; wooden 90%, metal 10%." \
        in intro([0.7, 0.5, 0.2, 0.9])
