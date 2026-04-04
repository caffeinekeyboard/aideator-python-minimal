from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PostType(str, Enum):
    MISSION = "mission"
    STAKEHOLDER = "stakeholder"
    GOAL = "goal"
    BARRIER = "barrier"
    CAUSE = "cause"
    SOLUTION = "solution"
    ABSTRACTION = "abstraction"
    ANALOGY = "analogy"
    INSPIRATION = "inspiration"
    QUESTION = "question"
    ANSWER = "answer"
    IMPROVEMENT = "improvement"


class Post(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ptype: PostType
    name: str
    description: str
    purpose: Optional[Post] = None
    achievers: list[Post] = Field(default_factory=list)


Post.model_rebuild()
