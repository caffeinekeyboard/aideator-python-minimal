"""Tests for experiment_worker parent resolution (pipeline gaps)."""

import pytest

from aideator.models import PostType
from experiment_worker import _resolve_parent_type_for_layer


def test_parent_first_layer_is_mission() -> None:
    p = [(PostType.STAKEHOLDER, 1)]
    assert _resolve_parent_type_for_layer(p, 0) == PostType.MISSION


def test_parent_stakeholder_goal_solution_chain() -> None:
    p = [
        (PostType.STAKEHOLDER, 1),
        (PostType.GOAL, 2),
        (PostType.SOLUTION, 3),
    ]
    assert _resolve_parent_type_for_layer(p, 0) == PostType.MISSION
    assert _resolve_parent_type_for_layer(p, 1) == PostType.STAKEHOLDER
    assert _resolve_parent_type_for_layer(p, 2) == PostType.GOAL


def test_parent_skipped_goal_solution_under_mission() -> None:
    """If GOAL is omitted (branching 0), solutions attach to mission, not stakeholders."""
    p = [
        (PostType.STAKEHOLDER, 1),
        (PostType.SOLUTION, 3),
    ]
    assert _resolve_parent_type_for_layer(p, 0) == PostType.MISSION
    assert _resolve_parent_type_for_layer(p, 1) == PostType.MISSION


def test_parent_mission_only_solution_pipeline() -> None:
    """Pipeline with no stakeholder row (web omits it when goal branching is 0)."""
    p = [(PostType.SOLUTION, 3)]
    assert _resolve_parent_type_for_layer(p, 0) == PostType.MISSION


def test_parent_skipped_barrier_solution_under_goal() -> None:
    p = [
        (PostType.STAKEHOLDER, 1),
        (PostType.GOAL, 2),
        (PostType.SOLUTION, 3),
    ]
    assert _resolve_parent_type_for_layer(p, 2) == PostType.GOAL


@pytest.mark.parametrize(
    ("pipeline", "pi", "expected"),
    [
        (
            [(PostType.STAKEHOLDER, 1), (PostType.GOAL, 1), (PostType.BARRIER, 1)],
            2,
            PostType.GOAL,
        ),
        (
            [
                (PostType.STAKEHOLDER, 1),
                (PostType.GOAL, 1),
                (PostType.BARRIER, 1),
                (PostType.SOLUTION, 1),
            ],
            3,
            PostType.BARRIER,
        ),
    ],
)
def test_parent_contiguous_chain(
    pipeline: list[tuple[PostType, int]], pi: int, expected: PostType
) -> None:
    assert _resolve_parent_type_for_layer(pipeline, pi) == expected
