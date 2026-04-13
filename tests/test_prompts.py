"""Unit tests for aideator/prompts.py.

Tests prompt structure invariants without checking full text — asserts that
key clauses are present or absent based on post type.
All tests are network-free.
"""

import pytest
from aideator.models import Post, PostType
from aideator.prompts import build_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mission():
    return Post(ptype=PostType.MISSION, name="Urban Mobility", description="Reduce congestion.")

def _stakeholder(purpose=None):
    m = purpose or _mission()
    return Post(ptype=PostType.STAKEHOLDER, name="Citizens", description="City residents.", purpose=m)

def _goal(purpose=None):
    s = purpose or _stakeholder()
    return Post(ptype=PostType.GOAL, name="Faster Commute", description="Reduce travel time.", purpose=s)

def _barrier(purpose=None):
    g = purpose or _goal()
    return Post(ptype=PostType.BARRIER, name="Poor Infrastructure", description="Bad roads.", purpose=g)

def _cause(purpose=None):
    b = purpose or _barrier()
    return Post(ptype=PostType.CAUSE, name="Underfunding", description="Lack of budget.", purpose=b)

def _abstraction(purpose=None):
    g = purpose or _goal()
    return Post(ptype=PostType.ABSTRACTION, name="Resource Flow", description="Abstract.", purpose=g)

def _analogy(purpose=None):
    a = purpose or _abstraction()
    return Post(ptype=PostType.ANALOGY, name="Water Pipes", description="Analogy.", purpose=a)

def _inspiration(purpose=None):
    a = purpose or _analogy()
    return Post(ptype=PostType.INSPIRATION, name="Pipeline Model", description="Inspiration.", purpose=a)

def _solution(purpose=None):
    g = purpose or _goal()
    return Post(ptype=PostType.SOLUTION, name="Bus Rapid Transit", description="Fast buses.", purpose=g)

def _question(purpose=None):
    s = purpose or _solution()
    return Post(ptype=PostType.QUESTION, name="How can we fund this?", description="Funding question.", purpose=s)


# ---------------------------------------------------------------------------
# Common preamble
# ---------------------------------------------------------------------------

class TestBuildPromptPreamble:
    def test_contains_requested_ptype(self):
        prompt = build_prompt(PostType.STAKEHOLDER, _mission())
        assert "stakeholder" in prompt.lower()

    def test_contains_purpose_name(self):
        mission = _mission()
        prompt = build_prompt(PostType.STAKEHOLDER, mission)
        assert mission.name in prompt

    def test_non_mission_includes_for_the_purpose(self):
        mission = _mission()
        prompt = build_prompt(PostType.STAKEHOLDER, mission)
        assert f"for the {PostType.MISSION.value}" in prompt


# ---------------------------------------------------------------------------
# Context inclusion rules
# ---------------------------------------------------------------------------

class TestContextInclusion:
    """Context description should appear for all types EXCEPT analogy and inspiration."""

    def test_stakeholder_includes_context(self):
        prompt = build_prompt(PostType.STAKEHOLDER, _mission())
        assert "The mission is:" in prompt

    def test_goal_includes_context(self):
        prompt = build_prompt(PostType.GOAL, _stakeholder())
        assert "The mission is:" in prompt

    def test_barrier_includes_context(self):
        prompt = build_prompt(PostType.BARRIER, _goal())
        assert "The mission is:" in prompt

    def test_solution_includes_context(self):
        prompt = build_prompt(PostType.SOLUTION, _goal())
        assert "The mission is:" in prompt

    def test_analogy_omits_context(self):
        prompt = build_prompt(PostType.ANALOGY, _abstraction())
        assert "The mission is:" not in prompt

    def test_inspiration_omits_context(self):
        prompt = build_prompt(PostType.INSPIRATION, _analogy())
        assert "The mission is:" not in prompt


# ---------------------------------------------------------------------------
# JSON template
# ---------------------------------------------------------------------------

class TestJsonTemplate:
    def test_prompt_ends_with_json_template(self):
        for ptype, parent in [
            (PostType.STAKEHOLDER, _mission()),
            (PostType.GOAL, _stakeholder()),
            (PostType.BARRIER, _goal()),
            (PostType.SOLUTION, _goal()),
        ]:
            prompt = build_prompt(ptype, parent)
            assert '"name"' in prompt
            assert '"description"' in prompt
            assert '"type"' in prompt

    def test_json_template_type_matches_ptype(self):
        prompt = build_prompt(PostType.GOAL, _stakeholder())
        assert '"type": "goal"' in prompt

    def test_json_template_uses_backtick_fences(self):
        prompt = build_prompt(PostType.GOAL, _stakeholder())
        assert "```json" in prompt


# ---------------------------------------------------------------------------
# Existing siblings
# ---------------------------------------------------------------------------

class TestExistingSiblings:
    def test_existing_siblings_appear_in_prompt(self):
        stakeholder = _stakeholder()
        existing_goal = Post(
            ptype=PostType.GOAL, name="Existing Goal",
            description="Already there.", purpose=stakeholder
        )
        stakeholder.achievers = [existing_goal]
        prompt = build_prompt(PostType.GOAL, stakeholder)
        assert "Existing Goal" in prompt

    def test_no_existing_siblings_no_list(self):
        stakeholder = _stakeholder()
        stakeholder.achievers = []
        prompt = build_prompt(PostType.GOAL, stakeholder)
        # Should not mention "already been proposed" siblings list
        assert "Existing Goal" not in prompt


# ---------------------------------------------------------------------------
# Stakeholder prompt fix (issue 5)
# ---------------------------------------------------------------------------

class TestStakeholderPrompt:
    def test_no_empty_list_instruction(self):
        """The old broken instruction referenced a list that was never provided."""
        prompt = build_prompt(PostType.STAKEHOLDER, _mission())
        assert "Pick a stakeholder from the following list" not in prompt

    def test_has_clear_dedup_instruction(self):
        prompt = build_prompt(PostType.STAKEHOLDER, _mission())
        assert "NOT been already proposed" in prompt


# ---------------------------------------------------------------------------
# Per-type specific checks
# ---------------------------------------------------------------------------

class TestPerTypePrompts:
    def test_abstraction_mentions_generalization(self):
        prompt = build_prompt(PostType.ABSTRACTION, _goal())
        assert "abstraction" in prompt.lower()
        assert "generali" in prompt.lower()

    def test_analogy_mentions_different_domain(self):
        prompt = build_prompt(PostType.ANALOGY, _abstraction())
        assert "domain" in prompt.lower()

    def test_solution_mentions_goal_or_barrier_or_cause(self):
        prompt = build_prompt(PostType.SOLUTION, _goal())
        # Should reference the origin (goal/barrier/cause) that the solution addresses
        assert "goal" in prompt.lower() or "barrier" in prompt.lower() or "cause" in prompt.lower()

    def test_question_instructs_how_can_we_format(self):
        prompt = build_prompt(PostType.QUESTION, _solution())
        assert "how can we" in prompt.lower()

    def test_improvement_mentions_solution(self):
        prompt = build_prompt(PostType.IMPROVEMENT, _solution())
        assert "improvement" in prompt.lower()

    def test_answer_references_solution(self):
        prompt = build_prompt(PostType.ANSWER, _question())
        assert "answer" in prompt.lower()
