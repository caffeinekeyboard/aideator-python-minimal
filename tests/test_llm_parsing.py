"""Unit tests for LLMClient.parse_response() and _extract_json_candidates().

All tests are network-free — no Gemini API key required.
"""

import pytest
from unittest.mock import patch

# Import only the static/standalone parts to avoid triggering the live client
import importlib, sys

# We test parse_response and _extract_json_candidates by importing after
# patching the google.genai import so no API key is needed at import time.
import types

# Patch google.genai before importing llm
google_mod = types.ModuleType("google")
genai_mod = types.ModuleType("google.genai")
google_mod.genai = genai_mod
sys.modules.setdefault("google", google_mod)
sys.modules.setdefault("google.genai", genai_mod)

from aideator.llm import LLMClient


# ---------------------------------------------------------------------------
# _extract_json_candidates
# ---------------------------------------------------------------------------

class TestExtractJsonCandidates:
    def test_single_object(self):
        result = LLMClient._extract_json_candidates('{"a": 1}')
        assert result == ['{"a": 1}']

    def test_multiple_top_level_objects(self):
        result = LLMClient._extract_json_candidates('{"a":1} some text {"b":2}')
        assert len(result) == 2
        assert '{"a":1}' in result
        assert '{"b":2}' in result

    def test_nested_object(self):
        result = LLMClient._extract_json_candidates('{"a": {"b": 1}}')
        assert len(result) == 1
        assert result[0] == '{"a": {"b": 1}}'

    def test_no_objects_returns_empty(self):
        result = LLMClient._extract_json_candidates("no braces here")
        assert result == []

    def test_ignores_unclosed_brace(self):
        result = LLMClient._extract_json_candidates('{"unclosed": 1')
        assert result == []


# ---------------------------------------------------------------------------
# _clean_json
# ---------------------------------------------------------------------------

class TestCleanJson:
    def test_removes_trailing_comma_before_brace(self):
        result = LLMClient._clean_json('{"a": 1,}')
        assert result == '{"a": 1}'

    def test_removes_trailing_comma_before_bracket(self):
        result = LLMClient._clean_json('{"a": [1, 2,]}')
        assert result == '{"a": [1, 2]}'

    def test_no_change_when_clean(self):
        s = '{"a": 1}'
        assert LLMClient._clean_json(s) == s


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_fenced_json(self):
        response = '```json\n{"type":"goal","name":"Clean Air","description":"Fresh air for all"}\n```'
        name, desc, ptype = LLMClient.parse_response(response)
        assert name == "Clean Air"
        assert desc == "Fresh air for all"
        assert ptype == "goal"

    def test_fenced_json_preferred_over_raw(self):
        """When fenced block exists it should be used, not the raw block."""
        response = (
            '{"type":"wrong","name":"Bad","description":"Should not pick this"}\n'
            '```json\n{"type":"goal","name":"Clean Air","description":"Correct"}\n```'
        )
        name, desc, ptype = LLMClient.parse_response(response)
        assert name == "Clean Air"

    def test_raw_json_with_surrounding_text(self):
        response = 'Sure! Here: {"type":"goal","name":"Clean Air","description":"Fresh air"} done.'
        name, desc, _ = LLMClient.parse_response(response)
        assert name == "Clean Air"
        assert desc == "Fresh air"

    def test_multiple_objects_picks_first_with_required_keys(self):
        response = '{"status":"ok"} {"type":"goal","name":"Clean Air","description":"Fresh air"}'
        name, desc, _ = LLMClient.parse_response(response)
        assert name == "Clean Air"

    def test_trailing_comma_handled(self):
        response = '{"type":"goal","name":"Clean Air","description":"Fresh air",}'
        name, desc, _ = LLMClient.parse_response(response)
        assert name == "Clean Air"

    def test_strips_whitespace_from_name_and_desc(self):
        response = '{"type":"goal","name":"  Clean Air  ","description":"  Fresh air  "}'
        name, desc, _ = LLMClient.parse_response(response)
        assert name == "Clean Air"
        assert desc == "Fresh air"

    def test_returns_type_field(self):
        response = '{"type":"barrier","name":"Traffic","description":"Heavy congestion"}'
        _, _, ptype = LLMClient.parse_response(response)
        assert ptype == "barrier"

    def test_returns_none_type_when_absent(self):
        response = '{"name":"Traffic","description":"Heavy congestion"}'
        _, _, ptype = LLMClient.parse_response(response)
        assert ptype is None

    def test_raises_when_no_json(self):
        with pytest.raises(ValueError, match="Could not find JSON"):
            LLMClient.parse_response("No JSON here at all.")

    def test_raises_when_name_missing(self):
        with pytest.raises(ValueError, match="name"):
            LLMClient.parse_response('{"type":"goal","description":"Fresh air"}')

    def test_raises_when_description_missing(self):
        with pytest.raises(ValueError, match="name"):
            LLMClient.parse_response('{"type":"goal","name":"Clean Air"}')

    def test_raises_when_fields_empty_string(self):
        with pytest.raises(ValueError):
            LLMClient.parse_response('{"type":"goal","name":"","description":""}')
