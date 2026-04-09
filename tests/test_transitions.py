"""Lock-in tests for the transition graph.

These tests assert the exact set of allowed edges and action names so that
any accidental change to transitions.py is caught immediately.
"""

import pytest
from aideator.models import PostType
from aideator.transitions import TRANSITIONS, ACTION_NAMES, validate_transition, get_allowed_children


# ---------------------------------------------------------------------------
# The canonical edge list (source of truth for these tests)
# ---------------------------------------------------------------------------
EXPECTED_EDGES: set[tuple[PostType, PostType]] = {
    (PostType.MISSION,      PostType.STAKEHOLDER),
    (PostType.STAKEHOLDER,  PostType.GOAL),
    (PostType.GOAL,         PostType.BARRIER),
    (PostType.GOAL,         PostType.SOLUTION),
    (PostType.GOAL,         PostType.ABSTRACTION),
    (PostType.BARRIER,      PostType.CAUSE),
    (PostType.BARRIER,      PostType.SOLUTION),
    (PostType.BARRIER,      PostType.ABSTRACTION),
    (PostType.CAUSE,        PostType.CAUSE),
    (PostType.CAUSE,        PostType.SOLUTION),
    (PostType.CAUSE,        PostType.ABSTRACTION),
    (PostType.ABSTRACTION,  PostType.ANALOGY),
    (PostType.ANALOGY,      PostType.INSPIRATION),
    (PostType.INSPIRATION,  PostType.SOLUTION),
    (PostType.SOLUTION,     PostType.IMPROVEMENT),
    (PostType.SOLUTION,     PostType.BARRIER),
    (PostType.SOLUTION,     PostType.QUESTION),
    (PostType.QUESTION,     PostType.ANSWER),
}


def _all_edges() -> set[tuple[PostType, PostType]]:
    return {
        (parent, child)
        for parent, children in TRANSITIONS.items()
        for child in children
    }


class TestTransitionEdges:
    def test_exact_edge_set(self):
        """TRANSITIONS must contain exactly the expected edges — no more, no less."""
        assert _all_edges() == EXPECTED_EDGES

    def test_no_extra_edges(self):
        assert _all_edges() - EXPECTED_EDGES == set()

    def test_no_missing_edges(self):
        assert EXPECTED_EDGES - _all_edges() == set()


class TestActionNames:
    def test_covers_all_transitions(self):
        """Every edge in TRANSITIONS must have an entry in ACTION_NAMES."""
        missing = _all_edges() - set(ACTION_NAMES.keys())
        assert missing == set(), f"ACTION_NAMES missing entries for: {missing}"

    def test_no_extra_action_names(self):
        """ACTION_NAMES must not contain edges that aren't in TRANSITIONS."""
        extra = set(ACTION_NAMES.keys()) - _all_edges()
        assert extra == set(), f"ACTION_NAMES has entries for non-existent edges: {extra}"

    def test_all_values_are_strings(self):
        for edge, name in ACTION_NAMES.items():
            assert isinstance(name, str) and name, f"Empty/non-string action name for {edge}"


class TestValidateTransition:
    def test_valid_transitions_return_true(self):
        for parent, child in EXPECTED_EDGES:
            assert validate_transition(parent, child), f"Expected valid: {parent} -> {child}"

    def test_invalid_transition_returns_false(self):
        assert validate_transition(PostType.MISSION, PostType.GOAL) is False
        assert validate_transition(PostType.ANSWER, PostType.SOLUTION) is False
        assert validate_transition(PostType.IMPROVEMENT, PostType.BARRIER) is False

    def test_type_with_no_children_returns_empty_list(self):
        assert get_allowed_children(PostType.ANSWER) == []
        assert get_allowed_children(PostType.IMPROVEMENT) == []
