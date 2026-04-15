from aideator.models import PostType


TRANSITIONS: dict[PostType, list[PostType]] = {
    PostType.MISSION:     [PostType.STAKEHOLDER, PostType.SOLUTION],
    PostType.STAKEHOLDER: [PostType.GOAL],
    PostType.GOAL:        [PostType.BARRIER, PostType.SOLUTION, PostType.ABSTRACTION],
    PostType.BARRIER:     [PostType.CAUSE, PostType.SOLUTION, PostType.ABSTRACTION],
    PostType.CAUSE:       [PostType.CAUSE, PostType.SOLUTION, PostType.ABSTRACTION],
    PostType.ABSTRACTION: [PostType.ANALOGY],
    PostType.ANALOGY:     [PostType.INSPIRATION],
    PostType.INSPIRATION: [PostType.SOLUTION],
    PostType.SOLUTION:    [PostType.QUESTION],
    PostType.QUESTION:    [PostType.ANSWER],
}

ACTION_NAMES: dict[tuple[PostType, PostType], str] = {
    (PostType.MISSION, PostType.STAKEHOLDER): "add-stakeholder",
    (PostType.MISSION, PostType.SOLUTION): "add-solution",
    (PostType.STAKEHOLDER, PostType.GOAL): "add-goal",
    (PostType.GOAL, PostType.BARRIER): "add-barrier",
    (PostType.GOAL, PostType.SOLUTION): "add-solution",
    (PostType.GOAL, PostType.ABSTRACTION): "add-abstraction",
    (PostType.BARRIER, PostType.CAUSE): "add-cause",
    (PostType.BARRIER, PostType.SOLUTION): "add-solution",
    (PostType.BARRIER, PostType.ABSTRACTION): "add-abstraction",
    (PostType.CAUSE, PostType.CAUSE): "add-cause",
    (PostType.CAUSE, PostType.SOLUTION): "add-solution",
    (PostType.CAUSE, PostType.ABSTRACTION): "add-abstraction",
    (PostType.ABSTRACTION, PostType.ANALOGY): "add-analogy",
    (PostType.ANALOGY, PostType.INSPIRATION): "add-inspiration",
    (PostType.INSPIRATION, PostType.SOLUTION): "use-inspiration",
    (PostType.SOLUTION, PostType.QUESTION): "add-question",
    (PostType.QUESTION, PostType.ANSWER): "add-answer",
}


def get_allowed_children(parent_type: PostType) -> list[PostType]:
    """Return the list of child types allowed for a given parent type."""
    return TRANSITIONS.get(parent_type, [])


def validate_transition(parent_type: PostType, child_type: PostType) -> bool:
    """Return True if the parent -> child transition is legal."""
    return child_type in TRANSITIONS.get(parent_type, [])
