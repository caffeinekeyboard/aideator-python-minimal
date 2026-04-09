"""Unit tests for aideator/tree.py.

Covers: context(), find_first(), build_post(), describe_context().
All tests are network-free.
"""

import pytest
from aideator.models import Post, PostType
from aideator.tree import context, find_first, build_post, describe_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mission(name="Test Mission"):
    return Post(ptype=PostType.MISSION, name=name, description="A mission.")

def _stakeholder(purpose, name="Stakeholders"):
    return Post(ptype=PostType.STAKEHOLDER, name=name, description="A stakeholder.", purpose=purpose)

def _goal(purpose, name="A Goal"):
    return Post(ptype=PostType.GOAL, name=name, description="A goal.", purpose=purpose)

def _barrier(purpose, name="A Barrier"):
    return Post(ptype=PostType.BARRIER, name=name, description="A barrier.", purpose=purpose)


# ---------------------------------------------------------------------------
# context()
# ---------------------------------------------------------------------------

class TestContext:
    def test_single_node_returns_self(self):
        root = _mission()
        assert context(root) == [root]

    def test_chain_order_is_post_to_root(self):
        root = _mission()
        s = _stakeholder(root)
        g = _goal(s)
        chain = context(g)
        assert chain == [g, s, root]

    def test_chain_length(self):
        root = _mission()
        s = _stakeholder(root)
        g = _goal(s)
        assert len(context(g)) == 3

    def test_root_has_no_purpose(self):
        root = _mission()
        chain = context(root)
        assert chain[-1].purpose is None

    def test_cycle_detection_raises(self):
        root = _mission()
        s = _stakeholder(root)
        # Manually create a cycle
        root.purpose = s
        with pytest.raises(ValueError, match="Circular reference"):
            context(s)
        # Clean up
        root.purpose = None


# ---------------------------------------------------------------------------
# find_first()
# ---------------------------------------------------------------------------

class TestFindFirst:
    def test_finds_direct_match(self):
        root = _mission()
        s = _stakeholder(root)
        assert find_first(PostType.STAKEHOLDER, s) is s

    def test_finds_ancestor(self):
        root = _mission()
        s = _stakeholder(root)
        g = _goal(s)
        assert find_first(PostType.MISSION, g) is root

    def test_returns_none_if_not_found(self):
        root = _mission()
        assert find_first(PostType.GOAL, root) is None

    def test_accepts_list_of_types(self):
        root = _mission()
        s = _stakeholder(root)
        g = _goal(s)
        b = _barrier(g)
        # find_first should return goal (nearest match in [goal, barrier, cause])
        result = find_first([PostType.GOAL, PostType.BARRIER, PostType.CAUSE], b)
        assert result is b  # barrier is the post itself, matches first

    def test_list_of_types_finds_first_in_chain(self):
        root = _mission()
        s = _stakeholder(root)
        g = _goal(s)
        result = find_first([PostType.GOAL, PostType.BARRIER], g)
        assert result is g

    def test_single_type_as_non_list(self):
        root = _mission()
        assert find_first(PostType.MISSION, root) is root


# ---------------------------------------------------------------------------
# build_post()
# ---------------------------------------------------------------------------

class TestBuildPost:
    def test_attaches_new_post_to_parent(self):
        root = _mission()
        s = build_post(root, PostType.STAKEHOLDER, "Citizens", "All city residents.")
        assert s in root.achievers
        assert s.purpose is root

    def test_returns_new_post(self):
        root = _mission()
        s = build_post(root, PostType.STAKEHOLDER, "Citizens", "All city residents.")
        assert s.ptype == PostType.STAKEHOLDER
        assert s.name == "Citizens"

    def test_dedup_same_name_same_ptype(self):
        root = _mission()
        s1 = build_post(root, PostType.STAKEHOLDER, "Citizens", "Description A.")
        s2 = build_post(root, PostType.STAKEHOLDER, "Citizens", "Description B.")
        assert len(root.achievers) == 1
        assert s1 is s2

    def test_dedup_case_insensitive(self):
        root = _mission()
        s1 = build_post(root, PostType.STAKEHOLDER, "Citizens", "Desc.")
        s2 = build_post(root, PostType.STAKEHOLDER, "CITIZENS", "Desc.")
        assert len(root.achievers) == 1
        assert s1 is s2

    def test_dedup_strips_whitespace(self):
        root = _mission()
        s1 = build_post(root, PostType.STAKEHOLDER, "Citizens", "Desc.")
        s2 = build_post(root, PostType.STAKEHOLDER, "  Citizens  ", "Desc.")
        assert len(root.achievers) == 1
        assert s1 is s2

    def test_different_names_both_attach(self):
        root = _mission()
        build_post(root, PostType.STAKEHOLDER, "Citizens", "Desc.")
        build_post(root, PostType.STAKEHOLDER, "Businesses", "Desc.")
        assert len(root.achievers) == 2

    def test_same_name_different_ptype_both_attach(self):
        """Same name but different ptype should both be allowed."""
        root = _mission()
        s = build_post(root, PostType.STAKEHOLDER, "Transport", "Desc.")
        # Now attach a goal with same name to the stakeholder (different parent, different ptype)
        g = build_post(s, PostType.GOAL, "Transport", "Desc.")
        assert len(s.achievers) == 1
        assert g.ptype == PostType.GOAL


# ---------------------------------------------------------------------------
# describe_context()
# ---------------------------------------------------------------------------

class TestDescribeContext:
    def test_mission_line_format(self):
        root = _mission("Clean City")
        desc = describe_context(root)
        assert "The mission is:" in desc
        assert "Clean City" in desc

    def test_chain_includes_all_ancestors(self):
        root = _mission("Clean City")
        s = _stakeholder(root, "Residents")
        g = _goal(s, "Fresh Air")
        desc = describe_context(g)
        assert "Clean City" in desc
        assert "Residents" in desc
        assert "Fresh Air" in desc

    def test_non_mission_uses_one_x_for_this_y_format(self):
        root = _mission("Clean City")
        s = _stakeholder(root, "Residents")
        desc = describe_context(s)
        assert "One stakeholder for this mission is" in desc

    def test_root_only_no_crash(self):
        root = _mission()
        desc = describe_context(root)
        assert len(desc) > 0
