from __future__ import annotations

from aideator.models import Post, PostType
from aideator.tree import context, find_first, describe_context


def _json_template(ptype: str, name_hint: str, desc_hint: str) -> str:
    """Build the standard JSON response template appended to every prompt."""
    return (
        "\n\nGive me your response as a JSON structure as follows:"
        "\n```json{"
        f'\n"type": "{ptype}",'
        f'\n"description": <{desc_hint}>,'
        f'\n"name": <{name_hint}>'
        "\n}```"
    )


def _list_existing(existing: list[Post], ptype: str) -> str:
    """Format existing siblings for inclusion in prompt."""
    if not existing:
        return ""
    lines = [f"\n- {e.name} {e.description}" for e in existing]
    return "".join(lines)


def _prompt_mission(purpose: Post, existing: list[Post]) -> str:
    s = ""
    # Fixed LISP bug: line 120 used (format nil ...) discarding this text.
    # Including it here as it was clearly intended.
    s += (
        "\n\nA mission should describe what deliberation we want to have, "
        "including who is the customer, what decision needs to be made, what we hope to accomplish, "
        "the hard constraints on what kinds of solutions are practical (for example, whether or not "
        "it'll be possible to change laws, what the maximum feasible timeline is, and the magnitude "
        "of money that can be spent). Each of those points should appear in their own paragraph "
        "prefixed with the topic expressed as a question."
    )
    s += "\n\nAlso describe the list of stakeholders who should be considered during the deliberation."
    s += _json_template(
        "mission",
        "a 4 to 6 word title for the mission",
        "a 2 or 3 sentence description of your proposed mission",
    )
    return s


def _prompt_stakeholder(purpose: Post, existing: list[Post]) -> str:
    s = ""
    s += (
        f"\n\nA stakeholder is a class of entities whose needs should be considered "
        f"during the deliberation on {purpose.name}. "
        f"You should discuss WHY the stakeholder is important for coming up with a successful "
        f"solution for this deliberation, for example whether its support is critical for the "
        f"solution to succeed. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += "\n\nHere are the stakeholders that have been proposed so far:"
        s += _list_existing(existing, "stakeholder")
    s += "\n\nPick a stakeholder from the following list that has NOT been already proposed:"
    s += _json_template(
        "stakeholder",
        "a 2 to 5 word name for the stakeholder",
        "a 2 or 3 sentence description of your proposed stakeholder",
    )
    return s


def _prompt_goal(purpose: Post, existing: list[Post]) -> str:
    s = ""
    s += (
        f"\n\nA goal represents a state (e.g. clean air) that is important for the success "
        f'of the {purpose.ptype.value} "{purpose.name}". '
        f"You should describe the goal as well as why it is important. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += f"\n\nThe following goals have already been proposed:"
        s += _list_existing(existing, "goal")
    s += "\n\nMake sure your proposed goal includes JUST ONE SIMPLE GOAL."
    s += '\nFor example, DO NOT say "Secure national energy sovereignty AND affordability". These are two different goals!'
    s += '\nPick just one goal e.g. "Secure national energy sovereignty".'
    s += '\nDO NOT say "Insure stable, affordable energy supply". Again, these are two different goals!'
    s += '\nPick just one goal e.g. "Secure stable energy supply".'
    s += _json_template(
        "goal",
        "a 4 to 6 word name for the goal",
        "a 2 or 3 sentence description of your proposed goal",
    )
    return s


def _prompt_barrier(purpose: Post, existing: list[Post]) -> str:
    s = ""
    s += (
        f"\n\nA barrier should represent something that undercuts our ability to succeed "
        f'at the {purpose.ptype.value} "{purpose.name}". '
        f"You should describe the barrier as well as why it is an important impediment "
        f"to the {purpose.ptype.value}. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += "\n\nThe barrier you propose should be as different as possible from these others:"
        s += _list_existing(existing, "barrier")
    s += _json_template(
        "barrier",
        "a 4 to 6 word name for the barrier",
        "a 2 or 3 sentence description of your proposed barrier",
    )
    return s


def _prompt_cause(purpose: Post, existing: list[Post]) -> str:
    s = ""
    s += (
        f"\n\nA cause should describe something that can lead to the "
        f'{purpose.ptype.value} {purpose.name}. '
        f"Describe the cause, how it leads to the {purpose.ptype.value}, "
        f"and WHY it can have a large impact. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += f"\n\nThe cause you propose should be as different as possible from these others:"
        s += _list_existing(existing, "cause")
    s += _json_template(
        "cause",
        "a 4 to 6 word name for the cause",
        "a 2 or 3 sentence description of your proposed cause",
    )
    return s


def _prompt_abstraction(purpose: Post, existing: list[Post]) -> str:
    origin = find_first([PostType.GOAL, PostType.BARRIER, PostType.CAUSE], purpose)
    s = ""
    origin_type = origin.ptype.value if origin else purpose.ptype.value
    origin_name = origin.name if origin else purpose.name
    s += (
        f"\n\nAn abstraction is the first step of using analogy to solve a problem. "
        f'The abstraction should represent a generalization of the {origin_type} "{origin_name}". '
        f"Try to generalize the {origin_type} a LOT so I am more likely to get out of the box "
        f"ideas from the analogy process. "
        f"When you generalize it, maintain the deep structure but abstract away domain-specific "
        f"features e.g. by replacing nouns and verbs with more general ones (hypernyms) that subsume them."
    )
    if existing:
        s += "\n\nThe abstraction you propose should be as different as possible from these others:"
        s += _list_existing(existing, "abstraction")
    s += _json_template(
        "abstraction",
        "a 4 to 6 word name for the abstraction",
        "a 2 or 3 sentence description of your proposed abstraction",
    )
    return s


def _prompt_analogy(purpose: Post, existing: list[Post]) -> str:
    origin = find_first([PostType.GOAL, PostType.BARRIER, PostType.CAUSE], purpose)
    s = ""
    origin_type = origin.ptype.value if origin else purpose.ptype.value
    origin_name = origin.name if origin else purpose.name
    s += (
        f"\n\nAn analogy should describe a problem that comes from a different domain "
        f'than the {origin_type} "{origin_name}", '
        f'but that instantiates the same structural pattern as the abstraction "{purpose.name}".'
    )
    if existing:
        s += "\n\nThe analogy you propose should be as different as possible from these others:"
        s += _list_existing(existing, "analogy")
    s += _json_template(
        "analogy",
        "a 4 to 6 word name for the analogy",
        "a 2 or 3 sentence description of your proposed analogy",
    )
    return s


def _prompt_inspiration(purpose: Post, existing: list[Post]) -> str:
    s = ""
    s += (
        f"\n\nYour inspiring idea should represent a possible solution "
        f'for the {purpose.ptype.value} "{purpose.name}".'
    )
    if existing:
        s += "\n\nThe idea you propose should be as different as possible from these others:"
        s += _list_existing(existing, "inspiration")
    s += _json_template(
        "inspiration",
        "a 4 to 6 word name for the solution",
        "a 2 or 3 sentence description of your proposed solution",
    )
    return s


def _prompt_solution(purpose: Post, existing: list[Post]) -> str:
    origin = find_first([PostType.GOAL, PostType.BARRIER, PostType.CAUSE], purpose)
    s = ""
    origin_type = origin.ptype.value if origin else purpose.ptype.value
    origin_name = origin.name if origin else purpose.name
    s += (
        f'\n\nA solution should address the {origin_type} "{origin_name}". '
        f"You should include the following points, each in their own paragraph:"
        f"\n\nWhat is the solution? <2 to 3 sentence description of the solution>"
        f"\n\nWhy is it a good solution? <2 to 3 sentence description of the advantages of this solution>"
    )
    s += "\n\nJust give me the solution, in a succinct way, without re-iterating the problem it solves."
    if existing:
        s += "\n\nThe solution you propose should be as different as possible from these others:"
        s += _list_existing(existing, "solution")
    s += _json_template(
        "solution",
        "a 4 to 6 word name for the solution",
        "description of solution",
    )
    return s


def _prompt_improvement(purpose: Post, existing: list[Post]) -> str:
    origin = find_first([PostType.GOAL, PostType.BARRIER, PostType.CAUSE], purpose)
    s = ""
    origin_type = origin.ptype.value if origin else purpose.ptype.value
    origin_name = origin.name if origin else purpose.name
    s += (
        f"\n\nAn improvement should describe how the "
        f'{purpose.ptype.value} "{purpose.name}" can be made better '
        f"in terms of addressing the {origin_type} {origin_name}. "
        f"You should discuss HOW it improves upon the {purpose.ptype.value}, "
        f"and WHY it is a good idea to implement it. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += "\n\nThe improvement you propose should be as different as possible from these others:"
        s += _list_existing(existing, "improvement")
    s += _json_template(
        "improvement",
        "a 4 to 6 word name for the improvement",
        "a 2 or 3 sentence description of your proposed improvement",
    )
    return s


def _prompt_question(purpose: Post, existing: list[Post]) -> str:
    s = ""
    s += (
        f'\n\nYour question should be one whose reply will help make the solution "{purpose.name}" '
        f"more complete. "
        f"A good question could for example identify a possible failure mode with the current "
        f"solution and ask how to avoid that failure mode. "
        f"It could ask what sub-components are needed to make the solution into a reality. "
        f"Make sure that the question's name has the structure: how can we X?"
    )
    if existing:
        s += f"\n\nThe question you propose should be as different as possible from these others:"
        s += _list_existing(existing, "question")
    s += _json_template(
        "question",
        "a 4 to 6 word name for the question",
        "a 2 or 3 sentence description of your proposed question",
    )
    return s


def _prompt_answer(purpose: Post, existing: list[Post]) -> str:
    origin = find_first([PostType.ANSWER, PostType.SOLUTION], purpose)
    s = ""
    origin_name = origin.name if origin else purpose.name
    # Fixed LISP bug: line 293 used (format nil ...) discarding this text.
    # Including it here as it was clearly intended.
    s += (
        f'\n\nThe answer should help make the solution "{origin_name}" more robust and complete. '
        f"You should describe the answer as well as why it is a good one. "
        f"Each of those points should appear in their own paragraph prefixed with the topic "
        f"expressed as a question."
    )
    if existing:
        s += f"\n\nThe answer you propose should be as different as possible from these others:"
        s += _list_existing(existing, "answer")
    s += _json_template(
        "answer",
        "a 4 to 6 word name for the answer",
        "a 2 or 3 sentence description of your proposed answer",
    )
    return s


_PROMPT_BUILDERS = {
    PostType.MISSION: _prompt_mission,
    PostType.STAKEHOLDER: _prompt_stakeholder,
    PostType.GOAL: _prompt_goal,
    PostType.BARRIER: _prompt_barrier,
    PostType.CAUSE: _prompt_cause,
    PostType.SOLUTION: _prompt_solution,
    PostType.ABSTRACTION: _prompt_abstraction,
    PostType.ANALOGY: _prompt_analogy,
    PostType.INSPIRATION: _prompt_inspiration,
    PostType.QUESTION: _prompt_question,
    PostType.ANSWER: _prompt_answer,
    PostType.IMPROVEMENT: _prompt_improvement,
}


def build_prompt(ptype: PostType, purpose: Post) -> str:
    """Build the full LLM prompt for proposing a new post of the given type.

    Faithfully reproduces the prompt construction from the LISP propose-achiever function.
    """
    # Gather existing achievers of the same type
    existing = [a for a in purpose.achievers if a.ptype == ptype]

    # Common preamble
    prompt = f"\n\nPlease propose a new {ptype.value}"
    if ptype != PostType.MISSION:
        prompt += f" for the {purpose.ptype.value}: {purpose.name}."
    if ptype not in (PostType.ANALOGY, PostType.INSPIRATION):
        prompt += (
            f"\n\nDo your best to make the {ptype.value} suited to the customer's "
            f"specific context: {describe_context(purpose)}"
        )

    # Per-type body
    builder = _PROMPT_BUILDERS.get(ptype)
    if builder is None:
        raise ValueError(f"No prompt template for post type: {ptype}")
    prompt += builder(purpose, existing)

    return prompt
