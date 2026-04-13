"""Tests for print_tree traversal order and index mapping.

Verifies that node numbering follows preorder (parent before children)
and that the index dict maps numbers to the correct Post objects.
Any change to traversal order will break these tests.
"""

from aideator.models import Post, PostType
from aideator.serialization import print_tree


def _make_tree() -> tuple[Post, Post, Post, Post]:
    """Build a small tree: mission -> stakeholder -> goal1, goal2."""
    mission = Post(ptype=PostType.MISSION, name="Mission", description="Root")
    stakeholder = Post(ptype=PostType.STAKEHOLDER, name="Stakeholder", description="S", purpose=mission)
    goal1 = Post(ptype=PostType.GOAL, name="Goal One", description="G1", purpose=stakeholder)
    goal2 = Post(ptype=PostType.GOAL, name="Goal Two", description="G2", purpose=stakeholder)
    mission.achievers = [stakeholder]
    stakeholder.achievers = [goal1, goal2]
    return mission, stakeholder, goal1, goal2


class TestPrintTreeIndexMapping:
    def test_preorder_numbering(self):
        """Nodes are numbered in preorder: root=1, then depth-first left-to-right."""
        mission, stakeholder, goal1, goal2 = _make_tree()
        index: dict[int, Post] = {}
        print_tree(mission, index=index)

        assert index[1] is mission
        assert index[2] is stakeholder
        assert index[3] is goal1
        assert index[4] is goal2

    def test_index_covers_all_nodes(self):
        """Index must contain exactly as many entries as there are nodes in the tree."""
        mission, _, _, _ = _make_tree()
        index: dict[int, Post] = {}
        print_tree(mission, index=index)
        assert len(index) == 4

    def test_numbering_starts_at_one(self):
        mission, _, _, _ = _make_tree()
        index: dict[int, Post] = {}
        print_tree(mission, index=index)
        assert min(index.keys()) == 1

    def test_output_contains_node_names(self):
        mission, _, _, _ = _make_tree()
        output = print_tree(mission)
        assert "Mission" in output
        assert "Stakeholder" in output
        assert "Goal One" in output
        assert "Goal Two" in output

    def test_output_contains_ptype_labels(self):
        mission, _, _, _ = _make_tree()
        output = print_tree(mission)
        assert "[mission]" in output
        assert "[stakeholder]" in output
        assert "[goal]" in output

    def test_single_node_tree(self):
        root = Post(ptype=PostType.MISSION, name="Solo", description="Alone")
        index: dict[int, Post] = {}
        print_tree(root, index=index)
        assert len(index) == 1
        assert index[1] is root
