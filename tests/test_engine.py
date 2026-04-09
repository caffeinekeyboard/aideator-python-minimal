"""Unit tests for IdeaEngine.

Uses a FakeLLM stub — no GEMINI_API_KEY required.
"""

import pytest
import types, sys

# Patch google.genai before importing engine
google_mod = types.ModuleType("google")
genai_mod = types.ModuleType("google.genai")
google_mod.genai = genai_mod
sys.modules.setdefault("google", google_mod)
sys.modules.setdefault("google.genai", genai_mod)

from aideator.engine import IdeaEngine
from aideator.models import Post, PostType


# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------

class FakeLLM:
    """Returns deterministic JSON responses for testing."""

    def __init__(self, ptype: str = "stakeholder", name: str = "Test Node", description: str = "A test description."):
        self._ptype = ptype
        self._name = name
        self._description = description

    def ask(self, prompt: str) -> str:
        return (
            f'{{"type":"{self._ptype}",'
            f'"name":"{self._name}",'
            f'"description":"{self._description}"}}'
        )


class MismatchedTypeLLM:
    """Returns a response with a type that doesn't match what was requested."""

    def ask(self, prompt: str) -> str:
        return '{"type":"goal","name":"Wrong Type","description":"This should fail."}'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateMission:
    def test_returns_mission_post(self):
        engine = IdeaEngine(llm_client=FakeLLM())
        root = engine.create_mission("Urban Mobility", "A description.")
        assert root.ptype == PostType.MISSION
        assert root.name == "Urban Mobility"
        assert root.description == "A description."

    def test_mission_has_no_parent(self):
        engine = IdeaEngine(llm_client=FakeLLM())
        root = engine.create_mission("Urban Mobility", "A description.")
        assert root.purpose is None

    def test_mission_has_no_children(self):
        engine = IdeaEngine(llm_client=FakeLLM())
        root = engine.create_mission("Urban Mobility", "A description.")
        assert root.achievers == []


class TestProposeAchiever:
    def test_attaches_child_to_parent(self):
        engine = IdeaEngine(llm_client=FakeLLM(ptype="stakeholder", name="Citizens", description="City residents."))
        root = engine.create_mission("Urban Mobility", "A description.")
        child = engine.propose_achiever(PostType.STAKEHOLDER, root)
        assert child in root.achievers
        assert child.purpose is root

    def test_returns_correct_ptype(self):
        engine = IdeaEngine(llm_client=FakeLLM(ptype="stakeholder", name="Citizens", description="City residents."))
        root = engine.create_mission("Urban Mobility", "A description.")
        child = engine.propose_achiever(PostType.STAKEHOLDER, root)
        assert child.ptype == PostType.STAKEHOLDER

    def test_returns_correct_name_and_description(self):
        engine = IdeaEngine(llm_client=FakeLLM(ptype="stakeholder", name="Citizens", description="City residents."))
        root = engine.create_mission("Urban Mobility", "A description.")
        child = engine.propose_achiever(PostType.STAKEHOLDER, root)
        assert child.name == "Citizens"
        assert child.description == "City residents."

    def test_invalid_transition_raises_value_error(self):
        engine = IdeaEngine(llm_client=FakeLLM())
        root = engine.create_mission("Urban Mobility", "A description.")
        with pytest.raises(ValueError, match="Cannot add"):
            engine.propose_achiever(PostType.GOAL, root)

    def test_invalid_transition_error_lists_allowed_types(self):
        engine = IdeaEngine(llm_client=FakeLLM())
        root = engine.create_mission("Urban Mobility", "A description.")
        with pytest.raises(ValueError, match="stakeholder"):
            engine.propose_achiever(PostType.GOAL, root)

    def test_type_mismatch_raises_value_error(self):
        """LLM returning wrong type should be rejected."""
        engine = IdeaEngine(llm_client=MismatchedTypeLLM())
        root = engine.create_mission("Urban Mobility", "A description.")
        with pytest.raises(ValueError, match="LLM returned type"):
            engine.propose_achiever(PostType.STAKEHOLDER, root)

    def test_accepts_any_llm_protocol_object(self):
        """Engine should work with any object that has ask()."""
        class MinimalStub:
            def ask(self, prompt):
                return '{"type":"stakeholder","name":"X","description":"Y"}'

        engine = IdeaEngine(llm_client=MinimalStub())
        root = engine.create_mission("Test", "Desc.")
        child = engine.propose_achiever(PostType.STAKEHOLDER, root)
        assert child.name == "X"

    def test_dedup_prevents_duplicate_children(self):
        """Proposing same name twice should not add a second child."""
        engine = IdeaEngine(llm_client=FakeLLM(ptype="stakeholder", name="Citizens", description="City residents."))
        root = engine.create_mission("Urban Mobility", "A description.")
        child1 = engine.propose_achiever(PostType.STAKEHOLDER, root)
        child2 = engine.propose_achiever(PostType.STAKEHOLDER, root)
        assert len(root.achievers) == 1
        assert child1 is child2
