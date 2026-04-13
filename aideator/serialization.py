from __future__ import annotations

import json
from typing import Optional

from aideator.models import Post, PostType


def tree_to_dict(post: Post) -> dict:
    """Recursively convert a post tree to a JSON-serializable dict.

    Omits the 'purpose' back-reference to break circularity;
    parent-child relationships are implied by nesting.
    """
    return {
        "id": post.id,
        "type": post.ptype.value,
        "name": post.name,
        "description": post.description,
        "achievers": [tree_to_dict(a) for a in post.achievers],
    }


def dict_to_tree(data: dict, purpose: Optional[Post] = None) -> Post:
    """Reconstruct a Post tree from a dict, re-linking purpose references."""
    for field in ("id", "type", "name", "description"):
        if field not in data:
            raise ValueError(
                f"Invalid tree data: missing required field '{field}'. "
                f"Got keys: {list(data.keys())}"
            )
    try:
        ptype = PostType(data["type"])
    except ValueError:
        valid = [t.value for t in PostType]
        raise ValueError(
            f"Invalid post type '{data['type']}'. Valid types are: {valid}"
        )
    post = Post(
        id=data["id"],
        ptype=ptype,
        name=data["name"],
        description=data["description"],
        purpose=purpose,
    )
    for child_data in data.get("achievers", []):
        child = dict_to_tree(child_data, purpose=post)
        post.achievers.append(child)
    return post


def export_json(root: Post, filepath: str) -> None:
    """Write the idea tree to a JSON file."""
    data = tree_to_dict(root)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def import_json(filepath: str) -> Post:
    """Read an idea tree from a JSON file and return the root Post."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return dict_to_tree(data)


def print_tree(post: Post, indent: int = 0, index: dict[int, Post] | None = None,
               counter: list[int] | None = None) -> str:
    """Return a text-based tree visualization with numbered posts.

    Args:
        post: The root post to display.
        indent: Current indentation level.
        index: Optional dict to populate with {number: post} mapping.
        counter: Internal counter for numbering (do not pass externally).

    Returns:
        A string representation of the tree.
    """
    if counter is None:
        counter = [1]

    num = counter[0]
    counter[0] += 1

    if index is not None:
        index[num] = post

    prefix = "  " * indent
    line = f"{prefix}{num}. [{post.ptype.value}] {post.name}\n"

    for achiever in post.achievers:
        line += print_tree(achiever, indent + 1, index, counter)

    return line
