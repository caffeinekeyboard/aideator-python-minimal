"""Unit tests for aideator/serialization.py.

Covers: tree_to_dict, dict_to_tree, export_json, import_json, and round-trip integrity.
All tests are network-free.
"""

import json
import os
import tempfile
import pytest

from aideator.models import Post, PostType
from aideator.serialization import tree_to_dict, dict_to_tree, export_json, import_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_small_tree() -> tuple[Post, Post, Post]:
    """mission -> stakeholder -> goal"""
    mission = Post(ptype=PostType.MISSION, name="Urban Mobility", description="Reduce congestion.")
    stakeholder = Post(ptype=PostType.STAKEHOLDER, name="Citizens", description="City residents.", purpose=mission)
    goal = Post(ptype=PostType.GOAL, name="Faster Commute", description="Reduce travel time.", purpose=stakeholder)
    mission.achievers = [stakeholder]
    stakeholder.achievers = [goal]
    return mission, stakeholder, goal


# ---------------------------------------------------------------------------
# tree_to_dict
# ---------------------------------------------------------------------------

class TestTreeToDict:
    def test_root_fields_present(self):
        mission, _, _ = _build_small_tree()
        d = tree_to_dict(mission)
        assert d["id"] == mission.id
        assert d["type"] == "mission"
        assert d["name"] == "Urban Mobility"
        assert d["description"] == "Reduce congestion."

    def test_no_purpose_field_in_dict(self):
        """purpose back-reference must not appear (would cause circular JSON)."""
        mission, _, _ = _build_small_tree()
        d = tree_to_dict(mission)
        assert "purpose" not in d

    def test_achievers_nested(self):
        mission, stakeholder, goal = _build_small_tree()
        d = tree_to_dict(mission)
        assert len(d["achievers"]) == 1
        assert d["achievers"][0]["name"] == "Citizens"
        assert d["achievers"][0]["achievers"][0]["name"] == "Faster Commute"

    def test_leaf_node_has_empty_achievers(self):
        mission, _, goal = _build_small_tree()
        d = tree_to_dict(mission)
        leaf = d["achievers"][0]["achievers"][0]
        assert leaf["achievers"] == []

    def test_json_serializable(self):
        mission, _, _ = _build_small_tree()
        d = tree_to_dict(mission)
        # Should not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
# dict_to_tree
# ---------------------------------------------------------------------------

class TestDictToTree:
    def test_restores_root_fields(self):
        mission, _, _ = _build_small_tree()
        restored = dict_to_tree(tree_to_dict(mission))
        assert restored.name == mission.name
        assert restored.ptype == PostType.MISSION
        assert restored.id == mission.id

    def test_restores_nested_structure(self):
        mission, stakeholder, goal = _build_small_tree()
        restored = dict_to_tree(tree_to_dict(mission))
        assert len(restored.achievers) == 1
        assert restored.achievers[0].name == stakeholder.name
        assert restored.achievers[0].achievers[0].name == goal.name

    def test_restores_purpose_links(self):
        mission, _, _ = _build_small_tree()
        restored = dict_to_tree(tree_to_dict(mission))
        child = restored.achievers[0]
        grandchild = child.achievers[0]
        assert child.purpose is restored
        assert grandchild.purpose is child

    def test_root_purpose_is_none(self):
        mission, _, _ = _build_small_tree()
        restored = dict_to_tree(tree_to_dict(mission))
        assert restored.purpose is None

    def test_missing_field_raises_friendly_error(self):
        bad = {"id": "1", "type": "goal", "name": "X"}  # missing description
        with pytest.raises(ValueError, match="description"):
            dict_to_tree(bad)

    def test_missing_id_raises_friendly_error(self):
        bad = {"type": "goal", "name": "X", "description": "Y"}
        with pytest.raises(ValueError, match="id"):
            dict_to_tree(bad)

    def test_invalid_type_raises_friendly_error(self):
        bad = {"id": "1", "type": "INVALID", "name": "X", "description": "Y"}
        with pytest.raises(ValueError, match="Invalid post type"):
            dict_to_tree(bad)

    def test_invalid_type_error_lists_valid_types(self):
        bad = {"id": "1", "type": "INVALID", "name": "X", "description": "Y"}
        with pytest.raises(ValueError, match="mission"):
            dict_to_tree(bad)


# ---------------------------------------------------------------------------
# Round-trip integrity
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_roundtrip_preserves_all_ptypes(self):
        mission, stakeholder, goal = _build_small_tree()
        restored = dict_to_tree(tree_to_dict(mission))
        assert restored.ptype == PostType.MISSION
        assert restored.achievers[0].ptype == PostType.STAKEHOLDER
        assert restored.achievers[0].achievers[0].ptype == PostType.GOAL

    def test_roundtrip_preserves_ids(self):
        mission, stakeholder, goal = _build_small_tree()
        restored = dict_to_tree(tree_to_dict(mission))
        assert restored.id == mission.id
        assert restored.achievers[0].id == stakeholder.id

    def test_roundtrip_preserves_node_count(self):
        mission, _, _ = _build_small_tree()
        restored = dict_to_tree(tree_to_dict(mission))
        def count(post):
            return 1 + sum(count(c) for c in post.achievers)
        assert count(restored) == count(mission)


# ---------------------------------------------------------------------------
# export_json / import_json
# ---------------------------------------------------------------------------

class TestFileIO:
    def test_export_creates_file(self):
        mission, _, _ = _build_small_tree()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_json(mission, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_import_restores_tree(self):
        mission, stakeholder, goal = _build_small_tree()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_json(mission, path)
            loaded = import_json(path)
            assert loaded.name == mission.name
            assert loaded.achievers[0].name == stakeholder.name
            assert loaded.achievers[0].achievers[0].name == goal.name
        finally:
            os.unlink(path)

    def test_exported_file_is_valid_json(self):
        mission, _, _ = _build_small_tree()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_json(mission, path)
            with open(path) as f:
                data = json.load(f)
            assert data["name"] == "Urban Mobility"
        finally:
            os.unlink(path)
