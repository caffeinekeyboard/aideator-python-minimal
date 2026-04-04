from __future__ import annotations

from aideator.models import Post, PostType
from aideator.tree import build_post
from aideator.transitions import validate_transition, get_allowed_children
from aideator.prompts import build_prompt
from aideator.llm import LLMClient


class IdeaEngine:
    """Core engine for building idea trees using LLM-generated proposals."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def create_mission(self, name: str, description: str) -> Post:
        """Create the root mission post from user-supplied values."""
        return Post(
            ptype=PostType.MISSION,
            name=name,
            description=description,
        )

    def propose_achiever(self, ptype: PostType, purpose: Post) -> Post:
        """Propose a new post of the given type as a child of purpose.

        Builds a prompt, calls the LLM, parses the response, and attaches
        the new post to the tree.

        Raises:
            ValueError: If the transition from purpose.ptype to ptype is not allowed.
        """
        if not validate_transition(purpose.ptype, ptype):
            allowed = get_allowed_children(purpose.ptype)
            allowed_str = ", ".join(t.value for t in allowed)
            raise ValueError(
                f"Cannot add {ptype.value} to {purpose.ptype.value}. "
                f"Allowed child types: {allowed_str}"
            )

        prompt = build_prompt(ptype, purpose)
        response = self.llm.ask(prompt)
        name, description = LLMClient.parse_response(response)
        return build_post(purpose, ptype, name, description)
