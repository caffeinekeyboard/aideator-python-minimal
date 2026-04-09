from __future__ import annotations

from typing import Optional

from aideator.models import Post, PostType


def context(post: Post) -> list[Post]:
    """Returns the ancestor chain: [post, parent, grandparent, ..., root].
    
    Raises:
        ValueError: If a circular reference is detected.
    """
    chain: list[Post] = []
    visited: set[str] = set() #<--to detect loops
    current: Optional[Post] = post
    
    while current is not None:
        if current.id in visited:
            raise ValueError(f"Circular reference detected: {current.id}")
        visited.add(current.id)
        chain.append(current)
        current = current.purpose
    
    return chain


def find_first(ptypes: list[PostType] | PostType, post: Post) -> Optional[Post]:
    """Find the first post in the context chain with one of the given types."""
    if isinstance(ptypes, PostType):
        ptypes = [ptypes]
    for p in context(post):
        if p.ptype in ptypes:
            return p
    return None


def build_post(purpose: Post, ptype: PostType, name: str, description: str) -> Post:
    """Create a new post and attach it to the purpose's achievers list.

    Deduplication is content-based: a sibling with the same ptype and
    normalised name (case-insensitive, stripped) is considered a duplicate
    and the existing post is returned without adding a new one.
    """
    normalised = name.strip().lower()
    for existing in purpose.achievers:
        if existing.ptype == ptype and existing.name.strip().lower() == normalised:
            return existing

    new_post = Post(
        ptype=ptype,
        name=name,
        description=description,
        purpose=purpose,
    )
    purpose.achievers.append(new_post)
    return new_post


def describe_context(post: Post) -> str:
    """Return a human-readable description of the ancestor chain from root to post."""
    chain = list(reversed(context(post)))
    parts: list[str] = []
    previous: Optional[Post] = None
    for item in chain:
        if item.ptype == PostType.MISSION:
            parts.append(f"\n\nThe mission is: {item.name}. {item.description}")
        else:
            prev_type = previous.ptype.value if previous else ""
            parts.append(f"\n\nOne {item.ptype.value} for this {prev_type} is: {item.name}. {item.description}")
        previous = item
    return "".join(parts)
