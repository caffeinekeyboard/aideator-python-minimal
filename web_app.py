"""web_app.py — Streamlit web UI for Aideator.

Run with:
    .venv/bin/python -m streamlit run web_app.py

Default port is 7500 (`.streamlit/config.toml`). Override: --server.port PORT

Two tabs:
  - Interactive Builder  : build an idea tree node-by-node, guided by the LLM
  - Experiment Runner    : automated background pipeline, persistent across
                           page refreshes, with full experiment history
"""

from __future__ import annotations

import ast
import html as _html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from aideator.engine import IdeaEngine
from aideator.models import Post, PostType
from aideator.serialization import dict_to_tree, import_json, tree_to_dict
from aideator.transitions import get_allowed_children
from aideator.tree import build_post, context as _node_context

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aideator",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Card component ──────────────────────────────────────────────────────── */
.aid-card {
    background: var(--secondary-background-color, #f8fafc);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
}

/* ── Metric pill ─────────────────────────────────────────────────────────── */
.metric-pill {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 20px;
    background: var(--secondary-background-color, #f8fafc);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    min-width: 100px;
}
.metric-pill .mp-val {
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
}
.metric-pill .mp-lbl {
    font-size: 11px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-top: 4px;
}

/* ── Section heading ─────────────────────────────────────────────────────── */
.aid-section-title {
    font-size: 13px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: 10px;
    margin-top: 4px;
}

/* ── Subtle dividers ─────────────────────────────────────────────────────── */
hr { border-color: #e2e8f0 !important; margin: 14px 0 !important; }

/* ── Condition preset cards ──────────────────────────────────────────────── */
.preset-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 13px 13px 11px 17px;
    margin-bottom: 4px;
    background: var(--secondary-background-color, #f8fafc);
    position: relative;
    overflow: hidden;
    min-height: 88px;
}
.preset-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    border-radius: 10px 0 0 10px;
}
.pc-c1::before { background: #16a34a; }
.pc-c2::before { background: #2563eb; }
.pc-c3::before { background: #7c3aed; }
.pc-c4::before { background: #ea580c; }
.preset-card.pc-active { background: #f8faff; border-color: #bfdbfe; }
.pc-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.pc-c1 .pc-label { color: #16a34a; }
.pc-c2 .pc-label { color: #2563eb; }
.pc-c3 .pc-label { color: #7c3aed; }
.pc-c4 .pc-label { color: #ea580c; }
.pc-name {
    font-size: 12px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 7px;
}
.pc-steps {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    align-items: center;
}
.pc-step {
    font-size: 9px;
    font-weight: 600;
    color: #475569;
    background: #f1f5f9;
    border-radius: 4px;
    padding: 2px 5px;
    white-space: nowrap;
}
.pc-arrow { font-size: 9px; color: #cbd5e1; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR  = Path(__file__).parent / "experiments"
SAVED_TREES_DIR  = Path(__file__).parent / "saved_trees"

PTYPE_ICON: dict[PostType, str] = {
    PostType.MISSION:     "🎯",
    PostType.STAKEHOLDER: "👥",
    PostType.GOAL:        "🏆",
    PostType.BARRIER:     "🚧",
    PostType.CAUSE:       "🔍",
    PostType.SOLUTION:    "✅",
    PostType.ABSTRACTION: "💡",
    PostType.ANALOGY:     "🔄",
    PostType.INSPIRATION: "⚡",
    PostType.QUESTION:    "❓",
    PostType.ANSWER:      "💬",
    PostType.IMPROVEMENT: "⬆️",
}

PTYPE_COLOR: dict[PostType, str] = {
    PostType.MISSION:     "#1565C0",
    PostType.STAKEHOLDER: "#2E7D32",
    PostType.GOAL:        "#E65100",
    PostType.BARRIER:     "#B71C1C",
    PostType.CAUSE:       "#6A1B9A",
    PostType.SOLUTION:    "#1B5E20",
    PostType.ABSTRACTION: "#F57F17",
    PostType.ANALOGY:     "#00695C",
    PostType.INSPIRATION: "#F9A825",
    PostType.QUESTION:    "#0277BD",
    PostType.ANSWER:      "#33691E",
    PostType.IMPROVEMENT: "#4A148C",
}

STATE_ICON = {
    "starting": "⟳",
    "running":  "⟳",
    "complete": "✓",
    "failed":   "✗",
}

STATE_COLOR = {
    "starting": "#2563eb",
    "running":  "#2563eb",
    "complete": "#16a34a",
    "failed":   "#dc2626",
    "unknown":  "#94a3b8",
}

# ── Condition presets ─────────────────────────────────────────────────────────
SOLUTIONS_SLIDER_MAX = 100

CONDITION_PRESETS: list[dict] = [
    {
        "id": 1,
        "label": "C1",
        "name": "Mission → Solutions",
        "desc": "Mission → Solution (no stakeholders/goals, solutions on any challenge)",
        "steps": ["Solution"],
        "sliders": dict(b_stakeholder=0, b_goal=0, b_barrier=0, b_cause=0,
                        b_abs=0, b_analogy=0, b_insp=0,
                        b_solution_per_challenge=96, b_question=0, b_answer=0),
        "solution_parent_types": [PostType.GOAL.value, PostType.BARRIER.value, PostType.CAUSE.value],
    },
    {
        "id": 2,
        "label": "C2",
        "name": "+ Problem Analysis",
        "desc": "… → Goal/Barrier/Cause → Solution (solutions on any challenge)",
        "steps": ["Goal", "Barrier", "Cause", "Solution"],
        "sliders": dict(b_stakeholder=1, b_goal=2, b_barrier=2, b_cause=2,
                        b_abs=0, b_analogy=0, b_insp=0,
                        b_solution_per_challenge=3, b_question=0, b_answer=0),
        # When set, the worker will generate solutions under all listed types
        # (in addition to the standard layer chaining).
        "solution_parent_types": [PostType.GOAL.value, PostType.BARRIER.value, PostType.CAUSE.value],
    },
    {
        "id": 3,
        "label": "C3",
        "name": "+ Analogical Ideation",
        "desc": "… → Cause → Abstraction → Analogy → Inspiration → Solution (solutions on any challenge)",
        "steps": ["Cause", "Abstraction", "Analogy", "Inspiration", "Solution"],
        "sliders": dict(b_stakeholder=1, b_goal=2, b_barrier=2, b_cause=2,
                        b_abs=2, b_analogy=4, b_insp=2,
                        b_solution_per_challenge=3, b_question=0, b_answer=0),
        "solution_parent_types": [PostType.GOAL.value, PostType.BARRIER.value, PostType.CAUSE.value],
    },
    {
        "id": 4,
        "label": "C4",
        "name": "+ Pregnant Question",
        "desc": "… → Solution → Question → Answer (solutions on any challenge)",
        "steps": ["Solution", "Question", "Answer"],
        "sliders": dict(b_stakeholder=1, b_goal=2, b_barrier=2, b_cause=2,
                        b_abs=2, b_analogy=4, b_insp=2,
                        b_solution_per_challenge=3, b_question=2, b_answer=1),
        "solution_parent_types": [PostType.GOAL.value, PostType.BARRIER.value, PostType.CAUSE.value],
    },
]

# ── Description renderer ──────────────────────────────────────────────────────
def _render_description(description: str) -> None:
    """Render a node description.

    The LLM sometimes returns a Python-dict-formatted string like
    ``{'What is the solution?': '...', 'Why is it a good solution?': '...'}``.
    When detected, each key is rendered as a bold label followed by its value.
    Otherwise the description is rendered as plain text.
    """
    text = description.strip()
    parsed: dict | None = None
    if text.startswith("{"):
        try:
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, dict):
                parsed = None
        except (ValueError, SyntaxError):
            parsed = None

    if parsed:
        for key, value in parsed.items():
            st.markdown(f"**{_html.escape(str(key))}**")
            st.text(str(value))
    else:
        st.markdown(text)


# ── Session state ──────────────────────────────────────────────────────────────
def _init_state() -> None:
    # Backward compatibility for previously persisted Streamlit sessions.
    if (
        "runner_b_solution_per_challenge" not in st.session_state
        and "runner_b_solution" in st.session_state
    ):
        st.session_state["runner_b_solution_per_challenge"] = st.session_state["runner_b_solution"]

    defaults: dict = {
        "root":               None,   # builder: active tree root
        "selected_id":        None,   # builder: selected node id
        "node_num":           1,      # builder: last node number chosen
        "engine":             None,   # builder: IdeaEngine instance
        "viewed_exp_id":      None,   # runner:  experiment currently on display
        "hist_runs_collapsed": False, # history: hide the "All runs" list for more detail space
        # Edit tab for completed experiments
        "exp_edit_root":        None,   # editable Post tree for the open experiment
        "exp_edit_exp_id":      None,   # which experiment the above tree belongs to
        "exp_edit_selected_id": None,   # selected node within edit view
        "exp_edit_node_num":    1,      # nav counter for edit view
        "exp_edit_engine":      None,   # IdeaEngine for "Propose with AI" in edit tab
        # Runner branching sliders (bound by key; do not pass duplicate defaults on widgets)
        "runner_b_goal":     2,
        "runner_b_stakeholder": 1,
        "runner_b_barrier":  0,
        "runner_b_cause":    0,
        "runner_b_abs":      0,
        "runner_b_analogy":  0,
        "runner_b_insp":     0,
        "runner_b_solution_per_challenge": 3,
        "runner_b_question": 0,
        "runner_b_answer":   0,
        "runner_solution_parent_types": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _estimate_theoretical_max_solutions(bg: dict) -> int:
    """Estimate solution upper bound implied by current branching factors.

    Uses the same logic as launch validation, including C2-style "solutions on
    any challenge" mode where solutions can be attached under goals/barriers/causes.
    """
    _stakeholder = int(bg.get("runner_b_stakeholder", 1))
    _goal = int(bg.get("runner_b_goal", 2))
    _pipe_steps: list[tuple[PostType, int]] = []
    if _stakeholder > 0:
        _pipe_steps.append((PostType.STAKEHOLDER, _stakeholder))
    _pipe_steps.extend(
        [
            (PostType.GOAL, _goal),
            (PostType.BARRIER, int(bg.get("runner_b_barrier", 0))),
            (PostType.CAUSE, int(bg.get("runner_b_cause", 0))),
            (PostType.ABSTRACTION, int(bg.get("runner_b_abs", 0))),
            (PostType.ANALOGY, int(bg.get("runner_b_analogy", 0))),
            (PostType.INSPIRATION, int(bg.get("runner_b_insp", 0))),
            (PostType.SOLUTION, int(bg.get("runner_b_solution_per_challenge", 3))),
            (PostType.QUESTION, int(bg.get("runner_b_question", 0))),
            (PostType.ANSWER, int(bg.get("runner_b_answer", 0))),
        ]
    )
    pipeline = [step for step in _pipe_steps if step[1] > 0]

    est = 1
    for _, b in pipeline:
        est *= b

    solution_parent_types = bg.get("runner_solution_parent_types")
    if solution_parent_types and PostType.SOLUTION in [pt for pt, _ in pipeline]:
        goals = _stakeholder * int(bg.get("runner_b_goal", 0))
        barriers = goals * int(bg.get("runner_b_barrier", 0))
        causes = barriers * int(bg.get("runner_b_cause", 0))
        total_challenges = 0
        if PostType.GOAL.value in solution_parent_types:
            total_challenges += goals
        if PostType.BARRIER.value in solution_parent_types:
            total_challenges += barriers
        if PostType.CAUSE.value in solution_parent_types:
            total_challenges += causes
        est = max(est, total_challenges * int(bg.get("runner_b_solution_per_challenge", 0)))

    return est


# ── Saved-tree helpers ────────────────────────────────────────────────────────
def _list_saved_trees() -> list[Path]:
    """Return saved tree files sorted newest-first."""
    if not SAVED_TREES_DIR.exists():
        return []
    return sorted(
        SAVED_TREES_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _save_tree(root: Post, save_name: str) -> Path:
    """Write the current tree to `saved_trees/<save_name>.json`. Returns the path."""
    SAVED_TREES_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^\w\s-]", "", save_name.strip()).strip()
    slug = re.sub(r"\s+", "_", slug)[:50] or "tree"
    path = SAVED_TREES_DIR / f"{slug}.json"
    path.write_text(json.dumps(tree_to_dict(root), indent=2))
    return path


@st.dialog("Save Tree")
def _save_tree_dialog(root: Post) -> None:
    mission_name = root.name if root else "tree"
    save_name = st.text_input(
        "Save name",
        value=mission_name,
        placeholder="Give this tree a name…",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save", type="primary", use_container_width=True):
            if not save_name.strip():
                st.error("Please enter a name.")
            else:
                path = _save_tree(root, save_name.strip())
                st.session_state.pop("pending_save_tree", None)
                st.toast(f'Saved as "{path.stem}"', icon="💾")
                st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("pending_save_tree", None)
            st.rerun()


def _maybe_show_save_dialog() -> None:
    if st.session_state.get("pending_save_tree"):
        _save_tree_dialog(st.session_state.root)


# ── Tree helpers ───────────────────────────────────────────────────────────────
def _all_nodes(root: Post) -> list[tuple[int, Post, int]]:
    """Return [(number, post, depth), ...] in preorder."""
    result: list[tuple[int, Post, int]] = []
    counter = [1]

    def _walk(post: Post, depth: int) -> None:
        result.append((counter[0], post, depth))
        counter[0] += 1
        for child in post.achievers:
            _walk(child, depth + 1)

    _walk(root, 0)
    return result


def _find_by_id(root: Post, node_id: str) -> Post | None:
    for _, post, _ in _all_nodes(root):
        if post.id == node_id:
            return post
    return None


def _get_selected() -> Post | None:
    root = st.session_state.root
    sid  = st.session_state.selected_id
    if root and sid:
        return _find_by_id(root, sid)
    return None


# ── HTML tree renderer ─────────────────────────────────────────────────────────
def _tree_html(
    root: Post,
    selected_id: str | None = None,
    scrollable: bool = False,
) -> tuple[str, dict[int, Post]]:
    """Render the tree as styled HTML with box-drawing connectors. Returns (html, {num: Post})."""
    index: dict[int, Post] = {}
    rows: list[str] = []
    counter = [1]

    def _row(post: Post, prefix: str, connector: str) -> None:
        num = counter[0]
        counter[0] += 1
        index[num] = post

        icon    = PTYPE_ICON.get(post.ptype, "•")
        color   = PTYPE_COLOR.get(post.ptype, "#333")
        is_sel  = post.id == selected_id
        row_bg  = (
            "background:#dbeafe;border-left:3px solid var(--primary-color,#2563eb);"
            if is_sel else ""
        )
        name_wt = "font-weight:700;" if is_sel else "font-weight:500;"
        safe_name = _html.escape(post.name)

        pre_html = (
            f'<span style="font-family:\'SF Mono\',\'Fira Code\',monospace;'
            f'font-size:12px;color:#cbd5e1;white-space:pre;flex-shrink:0;">'
            f'{_html.escape(prefix)}{_html.escape(connector)}'
            f'</span>'
        ) if (prefix or connector) else ""

        rows.append(
            f'<div style="display:flex;align-items:center;padding:3px 6px;'
            f'border-radius:5px;margin:1px 0;{row_bg}">'
            + pre_html +
            f'<span style="margin-right:6px;font-size:14px;flex-shrink:0;">{icon}</span>'
            f'<span style="font-size:10px;font-weight:700;color:{color};'
            f'background:{color}18;padding:1px 6px;border-radius:3px;'
            f'margin-right:7px;white-space:nowrap;flex-shrink:0;">{post.ptype.value}</span>'
            f'<span style="{name_wt}font-size:13px;">'
            f'<span style="opacity:0.4;margin-right:5px;font-size:11px;">{num}</span>'
            f'{safe_name}'
            f'</span>'
            f'</div>'
        )

        children = post.achievers
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            child_connector = "└─ " if is_last else "├─ "
            # Extend the prefix: last children leave a gap, others leave a vertical bar
            if not connector:          # root: no continuation indent
                child_prefix = ""
            elif connector.startswith("└"):
                child_prefix = prefix + "   "
            else:
                child_prefix = prefix + "│  "
            _row(child, child_prefix, child_connector)

    _row(root, "", "")

    scroll = "max-height:500px;overflow-y:auto;" if scrollable else ""
    wrapper = (
        f'<div style="font-family:system-ui,sans-serif;line-height:1.85;'
        f'padding:12px 10px;background:var(--secondary-background-color,#f8fafc);'
        f'border-radius:10px;border:1px solid #e2e8f0;{scroll}">'
        + "".join(rows)
        + "</div>"
    )
    return wrapper, index


# ═══════════════════════════════════════════════════════════════════════════════
# BUILDER TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_builder() -> None:
    if st.session_state.root is None:
        col_new, col_load = st.columns(2, gap="large")

        with col_new:
            st.markdown('<div class="aid-section-title">New Mission</div>', unsafe_allow_html=True)
            with st.form("new_mission_form"):
                name = st.text_input("Mission name", placeholder="e.g. Urban Transportation Reform")
                desc = st.text_area("Description", placeholder="Scope, constraints, goals…", height=120)
                if st.form_submit_button("✨ Create Mission", type="primary", use_container_width=True):
                    if not name.strip() or not desc.strip():
                        st.error("Name and description are required.")
                    else:
                        try:
                            engine = IdeaEngine()
                            root   = engine.create_mission(name.strip(), desc.strip())
                            st.session_state.engine      = engine
                            st.session_state.root        = root
                            st.session_state.selected_id = root.id
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to initialise engine: {e}")

        with col_load:
            st.markdown('<div class="aid-section-title">Load Tree</div>', unsafe_allow_html=True)

            # ── Saved trees ───────────────────────────────────────────────────
            saved = _list_saved_trees()
            if saved:
                st.markdown("**From saved trees**")
                saved_labels = {p.stem: p for p in saved}
                chosen_label = st.selectbox(
                    "Saved tree", options=list(saved_labels.keys()),
                    label_visibility="collapsed",
                )
                c_load_saved, c_del_saved = st.columns([3, 1])
                with c_load_saved:
                    if st.button("📂 Load", use_container_width=True, key="load_saved_btn"):
                        try:
                            root = import_json(str(saved_labels[chosen_label]))
                            st.session_state.engine      = IdeaEngine()
                            st.session_state.root        = root
                            st.session_state.selected_id = root.id
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load: {e}")
                with c_del_saved:
                    if st.button("🗑️", use_container_width=True, key="del_saved_btn",
                                 help="Delete this saved tree from disk"):
                        try:
                            saved_labels[chosen_label].unlink()
                            st.toast(f'Deleted "{chosen_label}"', icon="🗑️")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not delete: {e}")
                st.markdown("---")

            # ── Upload from file ──────────────────────────────────────────────
            st.markdown("**From file upload**")
            with st.form("load_form"):
                uploaded = st.file_uploader("Upload a JSON tree file", type="json")
                submitted = st.form_submit_button("📂 Load Tree", use_container_width=True)
                if submitted and uploaded:
                    try:
                        data = json.load(uploaded)
                        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                            json.dump(data, f)
                            tmp = f.name
                        root = import_json(tmp)
                        os.unlink(tmp)
                        st.session_state.engine      = IdeaEngine()
                        st.session_state.root        = root
                        st.session_state.selected_id = root.id
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {e}")
                elif submitted:
                    st.warning("Please upload a JSON file before clicking Load Tree.")
        return

    root     = st.session_state.root
    selected = _get_selected()
    col_tree, col_panel = st.columns([3, 2], gap="large")

    with col_tree:
        st.markdown('<div class="aid-section-title">Idea Tree</div>', unsafe_allow_html=True)
        tree_html, index = _tree_html(root, st.session_state.selected_id, scrollable=False)
        st.markdown(tree_html, unsafe_allow_html=True)
        # ── Node navigation ───────────────────────────────────────────────────
        total = len(index)
        # Resolve the number of the currently selected node from the index
        cur_num = next(
            (n for n, p in index.items() if p.id == st.session_state.selected_id), 1
        )

        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

        # Primary nav: ◀ Prev  |  counter label  |  Next ▶
        c_prev, c_counter, c_next = st.columns([1, 2, 1])
        with c_prev:
            if st.button(
                "◀ Prev", use_container_width=True,
                disabled=(cur_num <= 1),
                help="Select previous node",
            ):
                new_num = cur_num - 1
                st.session_state.node_num    = new_num
                st.session_state.selected_id = index[new_num].id
                st.rerun()
        with c_counter:
            st.markdown(
                f'<div style="text-align:center;padding:6px 0;font-size:13px;'
                f'font-weight:600;opacity:0.7;">node {cur_num} / {total}</div>',
                unsafe_allow_html=True,
            )
        with c_next:
            if st.button(
                "Next ▶", use_container_width=True,
                disabled=(cur_num >= total),
                help="Select next node",
            ):
                new_num = cur_num + 1
                st.session_state.node_num    = new_num
                st.session_state.selected_id = index[new_num].id
                st.rerun()

        # Secondary nav: jump directly to any node by number
        with st.expander("Jump to node…", expanded=False):
            c_jump_input, c_jump_btn = st.columns([3, 1])
            with c_jump_input:
                jump_num = st.number_input(
                    "Node number", min_value=1, max_value=total,
                    step=1, value=cur_num, label_visibility="collapsed",
                )
            with c_jump_btn:
                if st.button("Go", use_container_width=True):
                    st.session_state.node_num    = int(jump_num)
                    st.session_state.selected_id = index[int(jump_num)].id
                    st.rerun()

    with col_panel:
        if selected:
            color = PTYPE_COLOR.get(selected.ptype, "#333")
            icon  = PTYPE_ICON.get(selected.ptype, "")

            # ── Ancestor breadcrumb ───────────────────────────────────────────
            try:
                ancestors = list(reversed(_node_context(selected)))  # root → selected
            except ValueError:
                ancestors = [selected]
            if len(ancestors) > 1:
                crumb_parts = []
                for anc in ancestors[:-1]:  # all except selected itself
                    anc_icon  = PTYPE_ICON.get(anc.ptype, "")
                    anc_color = PTYPE_COLOR.get(anc.ptype, "#94a3b8")
                    crumb_parts.append(
                        f'<span style="color:{anc_color};font-weight:600;font-size:12px;">'
                        f'{anc_icon} {_html.escape(anc.name)}</span>'
                    )
                crumb_html = ' <span style="color:#cbd5e1;font-size:11px;">›</span> '.join(crumb_parts)
                st.markdown(
                    f'<div style="font-size:12px;margin-bottom:6px;opacity:0.8;">'
                    f'{crumb_html}'
                    f' <span style="color:#cbd5e1;font-size:11px;">›</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div style="padding:14px 16px;background:#f0f9ff;border-radius:10px;'
                f'border-left:4px solid {color};margin-bottom:12px;">'
                f'<div style="font-size:11px;color:{color};font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.05em;">{icon} {selected.ptype.value}</div>'
                f'<div style="font-size:17px;font-weight:700;color:#0f172a;margin-top:4px;">'
                f'{_html.escape(selected.name)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _render_description(selected.description)

        allowed = get_allowed_children(selected.ptype) if selected else []
        if allowed:
            st.markdown('<div class="aid-section-title" style="margin-top:16px;">Add a child node</div>', unsafe_allow_html=True)
            child_type = st.selectbox(
                "Child type", options=allowed,
                format_func=lambda t: f"{PTYPE_ICON.get(t, '')}  {t.value}",
                label_visibility="collapsed",
            )

            tab_ai, tab_manual = st.tabs(["🤖 Propose with AI", "✏️ Add Manually"])

            with tab_ai:
                if st.button("🤖 Propose with AI", type="primary", use_container_width=True):
                    with st.spinner(f"Asking the LLM to propose a {child_type.value}…"):
                        try:
                            new_post = st.session_state.engine.propose_achiever(child_type, selected)
                            st.session_state.selected_id = new_post.id
                            st.toast(f'Created: "{new_post.name}"', icon="✅")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

            with tab_manual:
                with st.form("manual_node_form", clear_on_submit=True):
                    manual_name = st.text_input(
                        "Name",
                        placeholder=f"A short name for the {child_type.value}…",
                    )
                    manual_desc = st.text_area(
                        "Description",
                        placeholder="Describe it in a few sentences…",
                        height=100,
                    )
                    if st.form_submit_button("➕ Add Node", use_container_width=True):
                        if not manual_name.strip():
                            st.error("Name is required.")
                        elif not manual_desc.strip():
                            st.error("Description is required.")
                        else:
                            new_post = build_post(
                                selected, child_type,
                                manual_name.strip(), manual_desc.strip(),
                            )
                            st.session_state.selected_id = new_post.id
                            st.toast(f'Added: "{new_post.name}"', icon="✅")
                            st.rerun()

        elif selected:
            color = PTYPE_COLOR.get(selected.ptype, "#94a3b8")
            st.markdown(
                f'<div style="padding:10px;background:#f1f5f9;border-radius:8px;'
                f'font-size:13px;color:#64748b;">No children allowed for '
                f'<b style="color:{color}">{selected.ptype.value}</b>.</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        c_save, c_dl = st.columns(2)
        with c_save:
            if st.button("💾 Save Tree", use_container_width=True):
                st.session_state.pending_save_tree = True
                st.rerun()
        with c_dl:
            dl_base = re.sub(r"\s+", "_", root.name[:30])
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(tree_to_dict(root), indent=2),
                file_name=f"{dl_base}.json",
                mime="application/json",
                use_container_width=True,
            )
        if st.button("🗑️ Reset Tree", use_container_width=True):
            st.session_state.pending_reset_tree = True
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER — persistence helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _exp_slug(name: str) -> str:
    slug = re.sub(r"[^\w\s]", "", name.lower()).strip()
    slug = re.sub(r"\s+", "_", slug)[:30]
    return slug or "experiment"


@st.cache_data
def _read_json_file_cached(path_str: str, mtime_ns: int) -> Any:
    """Read+parse a JSON file once per path/version."""
    _ = mtime_ns  # cache key only; not used in function body
    return json.loads(Path(path_str).read_text())


def _read_json_file(path: Path) -> Any | None:
    """Version-aware JSON reader (cache key = absolute path + mtime)."""
    if not path.exists():
        return None
    try:
        return _read_json_file_cached(str(path), path.stat().st_mtime_ns)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=4)
def _list_experiments() -> list[dict]:
    """Return all experiment metadata sorted newest-first.

    Cached with a 4-second TTL so repeated reruns (e.g. auto-refresh while a
    run is in progress) do not re-read every JSON file on every tick.
    Call ``_list_experiments.clear()`` after any mutation (delete / new run).
    """
    if not EXPERIMENTS_DIR.exists():
        return []
    results = []
    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        config_f = exp_dir / "config.json"
        status_f = exp_dir / "status.json"
        if not config_f.exists():
            continue
        config = _read_json_file(config_f)
        status = _read_json_file(status_f) if status_f.exists() else {"state": "unknown"}
        if not isinstance(config, dict) or not isinstance(status, dict):
            continue
        results.append({
            "id":              exp_dir.name,
            "path":            str(exp_dir),
            "mission_name":    config.get("mission_name", "Unknown"),
            "mission_desc":    config.get("mission_desc", ""),
            "started_at":      config.get("started_at", ""),
            "state":           status.get("state", "unknown"),
            "total_solutions": status.get("total_solutions", 0),
            "nodes_generated": status.get("nodes_generated", 0),
            "layer":           status.get("layer", 0),
            "total_layers":    status.get("total_layers", 0),
        })
    return sorted(results, key=lambda x: x["started_at"], reverse=True)


def _read_status(exp_id: str) -> dict:
    f = EXPERIMENTS_DIR / exp_id / "status.json"
    data = _read_json_file(f)
    if not isinstance(data, dict):
        return {"state": "unknown"}
    return data


def _read_log(exp_id: str) -> list[dict]:
    f = EXPERIMENTS_DIR / exp_id / "log.jsonl"
    if not f.exists():
        return []
    entries = []
    for line in f.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


def _read_log_tail(exp_id: str, max_lines: int = 800) -> list[dict]:
    """Parse the last ``max_lines`` non-empty lines of log.jsonl (for live progress polling)."""
    f = EXPERIMENTS_DIR / exp_id / "log.jsonl"
    if not f.exists():
        return []
    try:
        size = f.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []
    chunk_size = min(size, max(65536, max_lines * 400))
    with open(f, "rb") as fh:
        fh.seek(-chunk_size, os.SEEK_END)
        raw = fh.read().decode("utf-8", errors="replace")
    lines = raw.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    entries: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


def _read_results(exp_id: str) -> dict | None:
    f = EXPERIMENTS_DIR / exp_id / "results.json"
    data = _read_json_file(f)
    if not isinstance(data, dict):
        return None
    return data


def _launch_experiment(
    mission_name: str,
    mission_desc: str,
    pipeline: list[tuple[PostType, int]],
    max_concurrent: int = 4,
    extra_config: dict | None = None,
) -> str:
    """Write config, create experiment directory, launch worker subprocess."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    exp_id  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_exp_slug(mission_name)}"
    exp_dir = EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir()

    config = {
        "mission_name":    mission_name,
        "mission_desc":    mission_desc,
        "pipeline":        [[pt.value, b] for pt, b in pipeline],
        "started_at":      datetime.now().isoformat(),
        "experiment_id":   exp_id,
        "max_concurrent":  max_concurrent,
    }
    if extra_config:
        config.update(extra_config)
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "status.json").write_text(json.dumps({"state": "starting"}, indent=2))
    (exp_dir / "log.jsonl").write_text("")
    _read_json_file_cached.clear()
    _list_experiments.clear()

    worker = Path(__file__).parent / "experiment_worker.py"
    stderr_log = open(exp_dir / "worker_stderr.log", "w")  # noqa: WPS515
    env = {**os.environ, "EXPERIMENT_MAX_CONCURRENT": str(max_concurrent)}
    subprocess.Popen(
        [sys.executable, str(worker), str(exp_dir)],
        cwd=str(Path(__file__).parent),
        stdout=subprocess.DEVNULL,
        stderr=stderr_log,
        env=env,
    )
    return exp_id


def _rerun_experiment(source_exp_id: str) -> str:
    """Start a new experiment using mission + pipeline from an existing run."""
    config_f = EXPERIMENTS_DIR / source_exp_id / "config.json"
    if not config_f.exists():
        raise FileNotFoundError(f"No config for experiment {source_exp_id!r}")
    config = _read_json_file(config_f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config for experiment {source_exp_id!r}")
    raw_pipeline = config.get("pipeline")
    if not raw_pipeline:
        raise ValueError("Saved experiment has no pipeline; cannot rerun.")
    pipeline = [(PostType(pt), int(b)) for pt, b in raw_pipeline]
    max_concurrent = int(config.get("max_concurrent", 4))
    extra = {}
    if config.get("solution_parent_types"):
        extra["solution_parent_types"] = config["solution_parent_types"]
    return _launch_experiment(
        str(config["mission_name"]).strip(),
        str(config["mission_desc"]).strip(),
        pipeline,
        max_concurrent=max_concurrent,
        extra_config=extra or None,
    )


def _resume_experiment_worker(exp_id: str) -> None:
    """Start (or restart) the background worker for an existing experiment directory.

    Use when status is starting/running but the process died, or after a failure.
    Re-runs the full pipeline in that folder (same as a fresh worker launch).
    """
    exp_dir = EXPERIMENTS_DIR / exp_id
    if not (exp_dir / "config.json").exists():
        raise FileNotFoundError(f"No config for experiment {exp_id!r}")
    config = _read_json_file(exp_dir / "config.json")
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config for experiment {exp_id!r}")
    max_concurrent = int(config.get("max_concurrent", 4))
    worker = Path(__file__).parent / "experiment_worker.py"
    stderr_log = open(exp_dir / "worker_stderr.log", "a")  # noqa: WPS515
    env = {**os.environ, "EXPERIMENT_MAX_CONCURRENT": str(max_concurrent)}
    subprocess.Popen(
        [sys.executable, str(worker), str(exp_dir.resolve())],
        cwd=str(Path(__file__).parent.resolve()),
        stdout=subprocess.DEVNULL,
        stderr=stderr_log,
        env=env,
    )


def _delete_experiment_dir(exp_id: str) -> None:
    """Remove an experiment directory and clear selection if it was open."""
    p = EXPERIMENTS_DIR / exp_id
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
    if st.session_state.get("viewed_exp_id") == exp_id:
        st.session_state.viewed_exp_id = None
    _read_json_file_cached.clear()
    _list_experiments.clear()


@st.dialog("Delete experiment?")
def _delete_experiment_dialog(exp_id: str) -> None:
    st.markdown(f"Remove **`{exp_id}`** and all files under it? This cannot be undone.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Delete permanently", type="primary", use_container_width=True):
            _delete_experiment_dir(exp_id)
            st.session_state.pop("pending_delete_exp", None)
            st.toast("Experiment deleted.", icon="🗑️")
            st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("pending_delete_exp", None)
            st.rerun()


def _experiment_can_resume(state: str) -> bool:
    return state in ("starting", "failed")


def _maybe_show_delete_dialog() -> None:
    """Open delete confirmation when `pending_delete_exp` was set by a Delete button."""
    exp_id = st.session_state.get("pending_delete_exp")
    if exp_id:
        _delete_experiment_dialog(exp_id)


@st.dialog("Reset tree?")
def _reset_tree_dialog() -> None:
    st.markdown("Discard the current idea tree? **This cannot be undone.**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset", type="primary", use_container_width=True):
            st.session_state.root        = None
            st.session_state.selected_id = None
            st.session_state.node_num    = 1
            st.session_state.engine      = None
            st.session_state.pop("pending_reset_tree", None)
            st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("pending_reset_tree", None)
            st.rerun()


def _maybe_show_reset_dialog() -> None:
    if st.session_state.get("pending_reset_tree"):
        _reset_tree_dialog()


# ── Live experiment progress (autorefresh full rerun — avoids orphaned st.fragment) ─

def _should_autorefresh_running_experiment(tab: str) -> bool:
    """True when the main view is showing a run that is still active (poll disk every few seconds)."""
    if tab not in ("runner", "history"):
        return False
    vid = st.session_state.get("viewed_exp_id")
    if not vid or not (EXPERIMENTS_DIR / vid).exists():
        return False
    status = _read_status(vid)
    return status.get("state") in ("starting", "running")


def _render_experiment_live_progress(exp_id: str) -> None:
    """Show status/log while worker runs; parent script reruns on an interval via st_autorefresh."""
    status = _read_status(exp_id)
    state = status.get("state", "unknown")
    if state not in ("starting", "running"):
        return

    layer = status.get("layer", 0)
    total_lay = status.get("total_layers", 1)
    nodes = status.get("nodes_generated", 0)
    cur_type = status.get("current_type") or "—"
    progress = layer / total_lay if total_lay else 0

    st.progress(progress, text=f"Layer {layer} / {total_lay}  ·  {cur_type.upper()}")
    st.caption(f"Nodes generated so far: **{nodes}**")

    log_entries = _read_log_tail(exp_id)
    node_entries = [e for e in log_entries if e.get("event") == "node"]

    if node_entries:
        # ── Type count breakdown ──────────────────────────────────────────────
        type_counts: dict[str, int] = {}
        for e in node_entries:
            t = e.get("type", "?")
            type_counts[t] = type_counts.get(t, 0) + 1

        badge_html = ""
        for ptype_str, cnt in type_counts.items():
            try:
                icon  = PTYPE_ICON.get(PostType(ptype_str), "•")
                color = PTYPE_COLOR.get(PostType(ptype_str), "#333")
            except ValueError:
                icon, color = "•", "#888"
            badge_html += (
                f'<span style="display:inline-flex;align-items:center;gap:4px;'
                f'font-size:11px;font-weight:700;color:{color};'
                f'background:{color}18;padding:3px 8px;border-radius:12px;'
                f'border:1px solid {color}30;">'
                f'{icon} {ptype_str} <span style="opacity:0.7;">×{cnt}</span>'
                f'</span>'
            )
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 14px;">'
            f'{badge_html}</div>',
            unsafe_allow_html=True,
        )

        # ── Live node feed ────────────────────────────────────────────────────
        st.markdown("**Live node feed**")
        rows = []
        for e in reversed(node_entries[-40:]):
            ptype_str = e.get("type", "")
            try:
                icon  = PTYPE_ICON.get(PostType(ptype_str), "•")
                color = PTYPE_COLOR.get(PostType(ptype_str), "#333")
            except ValueError:
                icon, color = "•", "#333"
            safe_name = _html.escape(e.get("name", ""))
            rows.append(
                f'<div style="display:flex;align-items:center;gap:7px;'
                f'padding:4px 8px;border-bottom:1px solid #f1f5f9;font-size:13px;">'
                f'<span style="font-size:14px;flex-shrink:0;">{icon}</span>'
                f'<span style="font-size:10px;font-weight:700;color:{color};'
                f'background:{color}18;padding:1px 5px;border-radius:3px;'
                f'white-space:nowrap;flex-shrink:0;">{ptype_str}</span>'
                f'<span style="color:inherit;">{safe_name}</span>'
                f'</div>'
            )
        st.markdown(
            '<div style="background:var(--secondary-background-color,#f8fafc);'
            'border:1px solid #e2e8f0;border-radius:8px;'
            'max-height:260px;overflow-y:auto;">'
            + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )


# ── Experiment Edit tab ────────────────────────────────────────────────────────

def _render_experiment_edit(exp_id: str, results: dict) -> None:
    """Edit tab for a completed experiment: browse/edit nodes, add children, save."""
    # ── Load or reuse editable tree in session state ─────────────────────────
    if st.session_state.exp_edit_exp_id != exp_id:
        root = dict_to_tree(results["tree"])
        st.session_state.exp_edit_root        = root
        st.session_state.exp_edit_exp_id      = exp_id
        st.session_state.exp_edit_selected_id = root.id
        st.session_state.exp_edit_node_num    = 1
        st.session_state.exp_edit_engine      = IdeaEngine()
    else:
        root = st.session_state.exp_edit_root

    edit_engine: IdeaEngine = st.session_state.exp_edit_engine
    all_nodes = _all_nodes(root)
    total     = len(all_nodes)

    # Resolve selected node from id; fall back to root
    sel_id = st.session_state.exp_edit_selected_id
    edit_selected: Post | None = _find_by_id(root, sel_id) if sel_id else root
    if edit_selected is None:
        edit_selected = root
        st.session_state.exp_edit_selected_id = root.id

    # Current number of the selected node (1-based)
    cur_num = next(
        (n for n, p, _ in all_nodes if p.id == edit_selected.id),
        1,
    )
    st.session_state.exp_edit_node_num = cur_num

    col_tree, col_detail = st.columns([3, 2])

    # ── Left: tree + navigation ──────────────────────────────────────────────
    with col_tree:
        tree_html, tree_index = _tree_html(root, selected_id=sel_id, scrollable=False)
        st.markdown(tree_html, unsafe_allow_html=True)

        nav_l, nav_m, nav_r = st.columns([1, 2, 1])
        with nav_l:
            if st.button(
                "◀ Prev",
                key="exp_edit_prev",
                disabled=(cur_num <= 1),
                use_container_width=True,
            ):
                new_num = cur_num - 1
                st.session_state.exp_edit_node_num    = new_num
                st.session_state.exp_edit_selected_id = tree_index[new_num].id
                st.rerun()
        with nav_m:
            st.markdown(
                f'<div style="text-align:center;padding:6px 0;font-size:13px;">'
                f'node <strong>{cur_num}</strong> / {total}</div>',
                unsafe_allow_html=True,
            )
        with nav_r:
            if st.button(
                "Next ▶",
                key="exp_edit_next",
                disabled=(cur_num >= total),
                use_container_width=True,
            ):
                new_num = cur_num + 1
                st.session_state.exp_edit_node_num    = new_num
                st.session_state.exp_edit_selected_id = tree_index[new_num].id
                st.rerun()

        with st.expander("Jump to node…", expanded=False):
            jump_num = st.number_input(
                "Node number",
                min_value=1,
                max_value=total,
                value=cur_num,
                step=1,
                key="exp_edit_jump_input",
            )
            if st.button("Go", key="exp_edit_jump_go", use_container_width=True):
                target = tree_index.get(int(jump_num))
                if target:
                    st.session_state.exp_edit_node_num    = int(jump_num)
                    st.session_state.exp_edit_selected_id = target.id
                    st.rerun()

    # ── Right: node detail + edit + add + save ────────────────────────────────
    with col_detail:
        # Ancestor breadcrumb
        ancestors = list(reversed(_node_context(edit_selected)))
        if len(ancestors) > 1:
            crumb_parts = []
            for anc in ancestors[:-1]:
                anc_icon = PTYPE_ICON.get(anc.ptype, "•")
                crumb_parts.append(
                    f'<span style="opacity:0.6;font-size:11px;">'
                    f'{anc_icon} {_html.escape(anc.name)}'
                    f'</span>'
                )
            st.markdown(
                " › ".join(crumb_parts),
                unsafe_allow_html=True,
            )

        # Node card
        icon  = PTYPE_ICON.get(edit_selected.ptype, "•")
        color = PTYPE_COLOR.get(edit_selected.ptype, "#333")
        st.markdown(
            f'<div style="padding:12px 14px;background:var(--secondary-background-color,#f8fafc);'
            f'border-radius:10px;border-left:4px solid {color};margin-bottom:12px;">'
            f'<div style="font-size:11px;font-weight:700;color:{color};'
            f'background:{color}18;padding:2px 8px;border-radius:10px;'
            f'display:inline-block;margin-bottom:6px;">'
            f'{icon} {edit_selected.ptype.value}</div>'
            f'<div style="font-size:15px;font-weight:700;margin-bottom:4px;">'
            f'{_html.escape(edit_selected.name)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _render_description(edit_selected.description)

        # ── Edit name / description ──────────────────────────────────────────
        with st.expander("✏️ Edit name / description", expanded=False):
            with st.form("exp_edit_node_form"):
                new_name = st.text_input("Name", value=edit_selected.name)
                new_desc = st.text_area(
                    "Description",
                    value=(
                        edit_selected.description
                        if isinstance(edit_selected.description, str)
                        else str(edit_selected.description)
                    ),
                    height=120,
                )
                if st.form_submit_button("Update", use_container_width=True):
                    edit_selected.name        = new_name.strip()
                    edit_selected.description = new_desc.strip()
                    st.toast("Node updated.", icon="✏️")
                    st.rerun()

        # ── Add child node ───────────────────────────────────────────────────
        allowed = get_allowed_children(edit_selected.ptype)
        if allowed:
            with st.expander("➕ Add child node", expanded=False):
                child_type = st.selectbox(
                    "Child type",
                    options=allowed,
                    format_func=lambda pt: f"{PTYPE_ICON.get(pt, '•')} {pt.value}",
                    key="exp_edit_child_type",
                )
                ai_tab, manual_tab = st.tabs(["🤖 Propose with AI", "✏️ Add Manually"])

                with ai_tab:
                    if st.button(
                        f"Propose {child_type.value} with AI",
                        key="exp_edit_propose_ai",
                        use_container_width=True,
                    ):
                        with st.spinner("Asking AI…"):
                            try:
                                new_post = edit_engine.propose_achiever(
                                    child_type, edit_selected
                                )
                                st.session_state.exp_edit_selected_id = new_post.id
                                st.toast(f"Added: {new_post.name}", icon="🤖")
                                st.rerun()
                            except Exception as _exc:
                                st.error(str(_exc))

                with manual_tab:
                    with st.form("exp_edit_manual_child_form"):
                        m_name = st.text_input("Name")
                        m_desc = st.text_area("Description", height=80)
                        if st.form_submit_button("Add node", use_container_width=True):
                            if not m_name.strip():
                                st.error("Name is required.")
                            else:
                                new_post = build_post(
                                    edit_selected, child_type,
                                    m_name.strip(), m_desc.strip(),
                                )
                                st.session_state.exp_edit_selected_id = new_post.id
                                st.toast(f"Added: {new_post.name}", icon="✏️")
                                st.rerun()

        # ── Save changes ─────────────────────────────────────────────────────
        st.divider()
        if st.button("💾 Save Changes", type="primary", use_container_width=True,
                     key="exp_edit_save"):
            all_posts = [p for _, p, _ in _all_nodes(root)]
            sols      = [p for p in all_posts if p.ptype == PostType.SOLUTION]
            updated   = {
                "tree": tree_to_dict(root),
                "solutions": [
                    {"name": s.name, "description": s.description}
                    for s in sols
                ],
                "total_solutions": len(sols),
                "completed_at": results.get("completed_at", ""),
                "edited_at": datetime.now().isoformat(),
            }
            (EXPERIMENTS_DIR / exp_id / "results.json").write_text(
                json.dumps(updated, indent=2)
            )
            _read_json_file_cached.clear()
            _list_experiments.clear()
            st.toast("Saved. You can now use Continue pipeline to fill in missing children.", icon="💾")
            st.rerun()


# ── Experiment detail view ────────────────────────────────────────────────────

def _render_experiment_details(exp_id: str) -> None:
    config_f = EXPERIMENTS_DIR / exp_id / "config.json"
    if not config_f.exists():
        st.warning("Experiment data not found.")
        return

    config = _read_json_file(config_f)
    if not isinstance(config, dict):
        st.warning("Experiment config is invalid.")
        return
    status = _read_status(exp_id)
    state  = status.get("state", "unknown")

    # ── Header ────────────────────────────────────────────────────────────────
    badge = STATE_ICON.get(state, "⚪")
    badge_color = STATE_COLOR.get(state, STATE_COLOR["unknown"])
    started = config.get("started_at", "")[:19].replace("T", " ")
    safe_mission_title = _html.escape(str(config.get("mission_name", "")))
    st.markdown(
        f'<div style="padding:12px 16px;background:var(--secondary-background-color,#f8fafc);'
        f'border-radius:10px;border:1px solid #e2e8f0;margin-bottom:16px;">'
        f'<div style="font-size:18px;font-weight:700;">'
        f'{safe_mission_title}'
        f'<span style="font-size:12px;font-weight:700;color:{badge_color};'
        f'background:{badge_color}18;padding:2px 9px;border-radius:10px;margin-left:10px;">'
        f'{badge} {state}</span></div>'
        f'<div style="font-size:12px;color:#94a3b8;margin-top:4px;">'
        f'Started: {started} &nbsp;·&nbsp; ID: <code>{exp_id}</code></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Load results now so we can check edited_at for the button row
    results_early = _read_results(exp_id)
    has_edited = bool(results_early and results_early.get("edited_at"))

    btn_cols = st.columns(3 if (has_edited or _experiment_can_resume(state)) else 2)
    col_idx = 0
    with btn_cols[col_idx]:
        if _experiment_can_resume(state):
            if st.button(
                "▶ Resume worker",
                key=f"detail_resume_{exp_id}",
                use_container_width=True,
                help="Start the pipeline worker again in this folder (e.g. stuck or failed run).",
            ):
                try:
                    _resume_experiment_worker(exp_id)
                    st.toast("Worker started for this run.", icon="▶️")
                    st.rerun()
                except Exception as ex:
                    st.error(str(ex))
    col_idx += 1
    if has_edited:
        with btn_cols[col_idx]:
            if st.button(
                "▶ Continue pipeline",
                key=f"detail_continue_{exp_id}",
                use_container_width=True,
                type="primary",
                help="Resume the AI pipeline from the saved edited tree — fills in missing children.",
            ):
                try:
                    _resume_experiment_worker(exp_id)
                    st.toast("Worker started — will fill in missing children.", icon="▶️")
                    st.rerun()
                except Exception as ex:
                    st.error(str(ex))
        col_idx += 1
    with btn_cols[col_idx]:
        if st.button(
            "🗑️ Delete run",
            key=f"detail_del_{exp_id}",
            use_container_width=True,
        ):
            st.session_state.pending_delete_exp = exp_id
            st.rerun()

    # ── Running: live progress (refreshed by st_autorefresh when this view is active) ─
    if state in ("starting", "running"):
        _render_experiment_live_progress(exp_id)

    # ── Complete: results view ─────────────────────────────────────────────────
    elif state == "complete":
        nodes     = status.get("nodes_generated", 0)
        solutions = status.get("total_solutions", 0)
        completed = status.get("completed_at", "")[:19].replace("T", " ")

        c1, c2, c3 = st.columns(3)
        c1.metric("Solutions", solutions)
        c2.metric("Total nodes", nodes)
        c3.metric("Completed", completed)

        results = results_early or _read_results(exp_id)
        if results:
            parse_err: str | None = None
            result_root: Post | None = None
            try:
                result_root = dict_to_tree(results["tree"])
            except (KeyError, ValueError) as _err:
                parse_err = str(_err)

            tab_tree, tab_branching, tab_solutions, tab_nodes, tab_dl = st.tabs(
                ["🌳 Tree", "⚙️ Branching Config", "✅ Solutions", "📋 Node Details", "⬇️ Download"]
            )

            with tab_tree:
                if parse_err:
                    st.error(f"Could not render tree: {parse_err}")
                else:
                    assert result_root is not None
                    html, _ = _tree_html(result_root, scrollable=False)
                    st.markdown(html, unsafe_allow_html=True)

            with tab_branching:
                raw_pipeline = config.get("pipeline", [])
                if not raw_pipeline:
                    st.warning("No branching config found in experiment config.")
                else:
                    rows = []
                    for i, layer in enumerate(raw_pipeline, 1):
                        try:
                            ptype_str, branching = layer
                        except (TypeError, ValueError):
                            continue
                        rows.append(
                            {
                                "Layer": i,
                                "Node type": str(ptype_str),
                                "Branching factor": int(branching),
                            }
                        )
                    if rows:
                        st.caption("Configuration used to generate this completed tree.")
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.warning("Branching config exists but could not be parsed.")

                challenge_types = config.get("solution_parent_types")
                if challenge_types:
                    joined = ", ".join(str(t) for t in challenge_types)
                    st.caption(f"Solutions were also attached under challenge types: `{joined}`.")

            with tab_solutions:
                sols = results.get("solutions", [])
                if not sols:
                    st.warning("No solutions found.")
                else:
                    sol_color = PTYPE_COLOR.get(PostType.SOLUTION, "#1B5E20")
                    sol_icon  = PTYPE_ICON.get(PostType.SOLUTION, "✅")
                    for i, sol in enumerate(sols, 1):
                        safe_sol_name = _html.escape(sol.get("name", ""))
                        with st.expander(
                            f"{sol_icon} {i}. {sol.get('name', '')}",
                            expanded=(i <= 3),
                        ):
                            st.markdown(
                                f'<div style="font-size:14px;font-weight:700;'
                                f'color:{sol_color};margin-bottom:10px;">'
                                f'{sol_icon} {safe_sol_name}</div>',
                                unsafe_allow_html=True,
                            )
                            _render_description(sol.get("description", ""))

            with tab_nodes:
                if parse_err:
                    st.error(f"Could not load nodes: {parse_err}")
                else:
                    assert result_root is not None
                    all_nodes = _all_nodes(result_root)
                    st.caption(f"{len(all_nodes)} nodes in this experiment tree")
                    for num, post, _ in all_nodes:
                        icon  = PTYPE_ICON.get(post.ptype, "•")
                        color = PTYPE_COLOR.get(post.ptype, "#333")
                        with st.expander(f"{icon} {num}. {post.name}", expanded=False):
                            st.markdown(
                                f'<span style="font-size:11px;font-weight:700;'
                                f'color:{color};background:{color}18;'
                                f'padding:2px 8px;border-radius:10px;">'
                                f'{post.ptype.value}</span>',
                                unsafe_allow_html=True,
                            )
                            _render_description(post.description)

            with tab_dl:
                st.download_button(
                    "⬇️ Download Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name=f"{exp_id}_results.json",
                    mime="application/json",
                    use_container_width=True,
                )

    # ── Failed ────────────────────────────────────────────────────────────────
    elif state == "failed":
        reason = status.get("reason") or status.get("error", "Unknown error.")
        st.error(f"Experiment failed: {reason}")
        tb = status.get("traceback")
        if tb:
            with st.expander("Traceback"):
                st.code(tb, language="python")
        stderr_log = EXPERIMENTS_DIR / exp_id / "worker_stderr.log"
        if stderr_log.exists():
            content = stderr_log.read_text().strip()
            if content:
                with st.expander("Worker output (stderr)"):
                    st.code(content)

    else:
        st.info(f"Status: {state}")


# ── Experiment History (list + inline detail panel) ────────────────────────────

_HISTORY_PAGE_SIZE = 15


def _render_experiment_history_cards(key_prefix: str) -> None:
    """Experiment list with Open / Rerun / Resume / Delete."""
    experiments = _list_experiments()
    if not experiments:
        st.caption("No experiments yet. Go to **Experiment Runner** and launch one.")
        return

    # ── Pagination ────────────────────────────────────────────────────────────
    total = len(experiments)
    page_key = f"{key_prefix}page"
    page = st.session_state.get(page_key, 0)
    max_page = max(0, (total - 1) // _HISTORY_PAGE_SIZE)
    page = min(page, max_page)
    page_exps = experiments[page * _HISTORY_PAGE_SIZE : (page + 1) * _HISTORY_PAGE_SIZE]

    for exp in page_exps:
        icon = STATE_ICON.get(exp["state"], "⚪")
        started = exp.get("started_at", "")[:19].replace("T", " ")
        state = exp["state"]
        suffix = ""
        if state == "complete":
            suffix = f" · {exp['total_solutions']} solutions"
        elif state == "running" or state == "starting":
            suffix = f" · layer {exp['layer']}/{exp['total_layers']}"

        badge_color = STATE_COLOR.get(state, STATE_COLOR["unknown"])
        is_viewing = exp["id"] == st.session_state.get("viewed_exp_id")
        eid = exp["id"]
        with st.container(border=True):
            safe_hist_name = _html.escape(str(exp.get("mission_name", "")))
            st.markdown(
                f'<span style="font-weight:700;">{safe_hist_name}</span>'
                f'<span style="font-size:11px;font-weight:700;color:{badge_color};'
                f'background:{badge_color}18;padding:1px 7px;border-radius:10px;'
                f'margin-left:8px;">{icon} {state}{suffix}</span>',
                unsafe_allow_html=True,
            )
            desc = exp.get("mission_desc", "")
            if desc:
                preview = desc[:90] + ("…" if len(desc) > 90 else "")
                st.caption(preview)
            cap = f"{started} · `{eid}`"
            if is_viewing:
                cap = f"👁 **Shown in panel →** · {cap}"
            st.caption(cap)

            r = st.columns(4)
            with r[0]:
                if st.button("Open", key=f"{key_prefix}open_{eid}", use_container_width=True):
                    st.session_state.viewed_exp_id = eid
                    st.rerun()
            with r[1]:
                if st.button("Rerun", key=f"{key_prefix}rerun_{eid}", use_container_width=True):
                    try:
                        new_id = _rerun_experiment(eid)
                        st.session_state.viewed_exp_id = new_id
                        st.toast(f"Rerun started: {new_id}", icon="🚀")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not rerun: {e}")
            with r[2]:
                if _experiment_can_resume(state):
                    help_txt = (
                        "Start the worker again for this folder (stuck running, failed, or never started). "
                        "Re-runs the full pipeline in place."
                    )
                    if st.button(
                        "▶ Resume",
                        key=f"{key_prefix}resume_{eid}",
                        use_container_width=True,
                        help=help_txt,
                    ):
                        try:
                            _resume_experiment_worker(eid)
                            st.toast("Worker started for this run.", icon="▶️")
                            st.rerun()
                        except Exception as ex:
                            st.error(str(ex))
                else:
                    st.caption("")
            with r[3]:
                if st.button(
                    "🗑️",
                    key=f"{key_prefix}del_{eid}",
                    use_container_width=True,
                    help="Delete this run from disk",
                ):
                    st.session_state.pending_delete_exp = eid
                    st.rerun()

    # ── Pagination controls ───────────────────────────────────────────────────
    if max_page > 0:
        pc1, pc2, pc3 = st.columns([1, 2, 1])
        with pc1:
            if page > 0 and st.button("← Prev", key=f"{key_prefix}prev", use_container_width=True):
                st.session_state[page_key] = page - 1
                st.rerun()
        with pc2:
            st.caption(f"Page {page + 1} / {max_page + 1} · {total} runs")
        with pc3:
            if page < max_page and st.button("Next →", key=f"{key_prefix}next", use_container_width=True):
                st.session_state[page_key] = page + 1
                st.rerun()


def _render_experiment_history_page() -> None:
    """List on the left; **Open** shows the same experiment details on the right."""
    st.markdown('<div class="aid-section-title">Experiment History</div>', unsafe_allow_html=True)
    st.caption("Runs live under `experiments/`. **Open** loads progress, results, and downloads in the panel beside the list.")

    viewed_id = st.session_state.get("viewed_exp_id")
    collapsed = st.session_state.get("hist_runs_collapsed", False)

    # ── Header row: description + collapse toggle ─────────────────────────────
    hdr_left, hdr_right = st.columns([5, 1])
    with hdr_left:
        st.markdown(
            "Latest runs first. **Open** shows that run on this page (right). "
            "**Rerun** clones config into a *new* run. **▶ Resume** starts the worker again "
            "in the same folder if a run is stuck, failed, or never finished. **🗑️** removes "
            "that run from disk."
        )
    with hdr_right:
        # Only show the toggle when an experiment is open
        if viewed_id and (EXPERIMENTS_DIR / viewed_id).exists():
            label = "▶ Show runs" if collapsed else "◀ Hide runs"
            if st.button(label, key="hist_collapse_toggle", use_container_width=True,
                         help="Toggle the runs list to give more space to the detail view"):
                st.session_state.hist_runs_collapsed = not collapsed
                st.rerun()

    # ── Body: list + detail, or full-width detail when collapsed ─────────────
    if collapsed and viewed_id and (EXPERIMENTS_DIR / viewed_id).exists():
        _render_experiment_details(viewed_id)
    else:
        col_list, col_detail = st.columns([1.6, 3.4], gap="large")

        with col_list:
            st.markdown('<div class="aid-section-title">All runs</div>', unsafe_allow_html=True)
            with st.container(
                key="hist_all_runs",
                border=True,
                height=820,
            ):
                _render_experiment_history_cards(key_prefix="histpg_")

        with col_detail:
            st.markdown('<div class="aid-section-title">Experiment</div>', unsafe_allow_html=True)
            if viewed_id and (EXPERIMENTS_DIR / viewed_id).exists():
                _render_experiment_details(viewed_id)
            else:
                st.info(
                    "Click **Open** on a run to show live progress, results, and downloads here."
                )


# ── Runner tab ────────────────────────────────────────────────────────────────

def _render_runner() -> None:
    st.markdown('<div class="aid-section-title">Experiment Runner</div>', unsafe_allow_html=True)
    st.caption(
        "Experiments run as **background processes** and persist across page refreshes "
        "and tab switches. Results are saved to `experiments/` automatically."
    )
    st.info("Use this tab to configure and launch experiments only. View progress/results in **History**.")

    # ── Condition presets ──────────────────────────────────────────────────
    active_preset = st.session_state.get("runner_active_preset", None)
    st.markdown('<div class="aid-section-title">Condition Presets</div>', unsafe_allow_html=True)
    card_cols = st.columns(4)
    for idx, p in enumerate(CONDITION_PRESETS):
        is_active = active_preset == p["id"]
        active_cls = "pc-active" if is_active else ""
        steps_html = " ".join(
            f'<span class="pc-step">{s}</span>'
            + ('' if j == len(p["steps"]) - 1 else '<span class="pc-arrow">›</span>')
            for j, s in enumerate(p["steps"])
        )
        card_html = (
            f'<div class="preset-card pc-c{p["id"]} {active_cls}">'
            f'<div class="pc-label">{p["label"]}</div>'
            f'<div class="pc-name">{p["name"]}</div>'
            f'<div class="pc-steps">{steps_html}</div>'
            f'</div>'
        )
        with card_cols[idx]:
            st.markdown(card_html, unsafe_allow_html=True)

    button_cols = st.columns(4)
    for idx, p in enumerate(CONDITION_PRESETS):
        is_active = active_preset == p["id"]
        with button_cols[idx]:
            if st.button(
                "✓ Active" if is_active else f"Select {p['label']}",
                key=f"preset_btn_{p['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=p["desc"],
            ):
                for k, v in p["sliders"].items():
                    st.session_state[f"runner_{k}"] = v
                st.session_state["runner_solution_parent_types"] = p.get("solution_parent_types")
                st.session_state["runner_active_preset"] = p["id"]
                st.rerun()
    st.markdown("")

    # ── Branching factors (outside form for live estimate updates) ─────────
    with st.expander("➕ New Experiment", expanded=True):
        mission_name = st.text_input(
            "Mission name", value="Next-Gen Grid-Scale Energy Storage"
        )
        mission_desc = st.text_area(
            "Description",
            value=(
                "Develop novel methods to store grid-scale renewable energy "
                "that do not rely on rare-earth lithium-ion batteries, targeting "
                "cost, scalability, and environmental sustainability."
            ),
            height=90,
        )
        st.markdown("**Branching factors**")
        b_stakeholder = st.slider("👥  Stakeholders / Mission", 0, 4, key="runner_b_stakeholder")
        b_goal     = st.slider("🏆  Goals / Stakeholder",     0, 4, key="runner_b_goal")
        b_barrier  = st.slider("🚧  Barriers / Goal",          0, 4, key="runner_b_barrier")
        b_cause    = st.slider("🔍  Causes / Barrier",          0, 4, key="runner_b_cause")
        b_abs      = st.slider("💡  Abstractions / Cause",      0, 4, key="runner_b_abs")
        b_analogy  = st.slider("🔄  Analogies / Abstraction",  0, 6, key="runner_b_analogy")
        b_insp     = st.slider("⚡  Inspirations / Analogy",   0, 4, key="runner_b_insp")
        b_solution_per_challenge = st.slider(
            "✅  Solutions / Challenge", 0, SOLUTIONS_SLIDER_MAX, key="runner_b_solution_per_challenge"
        )
        b_question = st.slider("❓  Questions / Solution",      0, 4, key="runner_b_question")
        b_answer   = st.slider("💬  Answers / Question",        0, 4, key="runner_b_answer")
        est = _estimate_theoretical_max_solutions(st.session_state)
        st.caption(f"Theoretical max solutions: **{est}**")

        st.divider()

        # ── Mission details + launch ───────────────────────────────────────
        with st.form("runner_form"):
            st.markdown("**Performance**")
            max_concurrent = st.slider(
                "⚡  Parallel requests",
                min_value=1, max_value=16, value=4,
                help=(
                    "Max number of parent nodes processed simultaneously. "
                    "Higher = faster, but may hit API rate limits. "
                    "Use 4 for free-tier keys, 8–12 for paid."
                ),
            )

            if st.form_submit_button("🚀 Launch Experiment", type="primary", use_container_width=True):
                if not mission_name.strip() or not mission_desc.strip():
                    st.error("Mission name and description are required.")
                else:
                    _bg = st.session_state
                    _stakeholder = int(_bg.get("runner_b_stakeholder", 1))
                    _goal = int(_bg.get("runner_b_goal", 2))
                    if _goal > 0 and _stakeholder == 0:
                        st.error("Set Stakeholders / Mission to at least 1 when Goals / Stakeholder is above 0.")
                        return
                    _pipe_steps: list[tuple[PostType, int]] = []
                    if _stakeholder > 0:
                        _pipe_steps.append((PostType.STAKEHOLDER, _stakeholder))
                    _pipe_steps.extend(
                        [
                            (PostType.GOAL, _goal),
                            (PostType.BARRIER, int(_bg.get("runner_b_barrier", 0))),
                            (PostType.CAUSE, int(_bg.get("runner_b_cause", 0))),
                            (PostType.ABSTRACTION, int(_bg.get("runner_b_abs", 0))),
                            (PostType.ANALOGY, int(_bg.get("runner_b_analogy", 0))),
                            (PostType.INSPIRATION, int(_bg.get("runner_b_insp", 0))),
                            (PostType.SOLUTION, int(_bg.get("runner_b_solution_per_challenge", 3))),
                            (PostType.QUESTION, int(_bg.get("runner_b_question", 0))),
                            (PostType.ANSWER, int(_bg.get("runner_b_answer", 0))),
                        ]
                    )
                    pipeline = [step for step in _pipe_steps if step[1] > 0]
                    solution_parent_types = _bg.get("runner_solution_parent_types")
                    extra = {}
                    if solution_parent_types:
                        extra["solution_parent_types"] = list(solution_parent_types)
                    try:
                        exp_id = _launch_experiment(
                            mission_name.strip(), mission_desc.strip(), pipeline,
                            max_concurrent=max_concurrent,
                            extra_config=extra or None,
                        )
                        st.session_state.viewed_exp_id = exp_id
                        st.success(f"Experiment launched successfully: `{exp_id}`")
                        st.toast(f"Run started: {exp_id}", icon="🚀")
                    except Exception as ex:
                        st.error(f"Failed to launch experiment: {ex}")

    viewed_id = st.session_state.get("viewed_exp_id")
    if viewed_id and (EXPERIMENTS_DIR / viewed_id).exists():
        status = _read_status(viewed_id)
        state = status.get("state", "unknown")
        if state in ("starting", "running"):
            st.success(f"Current run `{viewed_id}` is {state}. Open **History** for live progress.")
        elif state == "failed":
            st.error(f"Current run `{viewed_id}` failed. Open **History** to inspect the error details.")
        elif state == "complete":
            st.info(f"Current run `{viewed_id}` is complete. Open **History** to review results.")
        else:
            st.warning(f"Current run `{viewed_id}` status is `{state}`. Open **History** for details.")


# ── Metrics page ─────────────────────────────────────────────────────────────

def _render_metrics() -> None:
    """Experiment Metrics — aggregate stats across all runs."""
    experiments = _list_experiments()

    # ── Aggregate numbers ─────────────────────────────────────────────────────
    total_runs      = len(experiments)
    complete_runs   = [e for e in experiments if e["state"] == "complete"]
    failed_runs     = [e for e in experiments if e["state"] == "failed"]
    running_runs    = [e for e in experiments if e["state"] in ("starting", "running")]
    total_solutions = sum(e.get("total_solutions", 0) for e in complete_runs)
    total_nodes     = sum(e.get("nodes_generated", 0) for e in complete_runs)
    avg_solutions   = (total_solutions / len(complete_runs)) if complete_runs else 0

    # ── Summary pills ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="aid-section-title">Overview</div>',
        unsafe_allow_html=True,
    )
    pill_cols = st.columns(6)
    _pills = [
        (total_runs,              "Total Runs"),
        (len(complete_runs),      "Completed"),
        (len(running_runs),       "Running"),
        (len(failed_runs),        "Failed"),
        (total_solutions,         "Solutions"),
        (f"{avg_solutions:.1f}",  "Avg / Run"),
    ]
    for col, (val, lbl) in zip(pill_cols, _pills):
        col.markdown(
            f'<div class="metric-pill">'
            f'<span class="mp-val">{val}</span>'
            f'<span class="mp-lbl">{lbl}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if not experiments:
        st.info("No experiments found. Launch one from the **Runner** page.")
        return

    # ── Per-run table ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="aid-section-title">All Runs</div>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for e in experiments:
        state      = e["state"]
        badge_col  = STATE_COLOR.get(state, STATE_COLOR["unknown"])
        icon       = STATE_ICON.get(state, "⚪")
        started    = e.get("started_at", "")[:16].replace("T", " ")
        sols       = e.get("total_solutions", "—") if state == "complete" else "—"
        nodes      = e.get("nodes_generated", "—") if state == "complete" else "—"
        name       = _html.escape(e["mission_name"])
        rows_html += (
            f'<tr>'
            f'<td style="padding:9px 12px;font-weight:600;">{name}</td>'
            f'<td style="padding:9px 12px;">'
            f'<span style="font-size:11px;font-weight:700;color:{badge_col};'
            f'background:{badge_col}18;padding:2px 8px;border-radius:8px;">'
            f'{icon} {state}</span></td>'
            f'<td style="padding:9px 12px;color:#64748b;font-size:12px;">{started}</td>'
            f'<td style="padding:9px 12px;text-align:center;">{nodes}</td>'
            f'<td style="padding:9px 12px;text-align:center;">{sols}</td>'
            f'</tr>'
        )

    st.markdown(
        '<div style="overflow-x:auto;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;'
        'background:var(--secondary-background-color,#f8fafc);'
        'border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">'
        '<thead><tr style="background:#0f172a;color:#94a3b8;">'
        '<th style="padding:10px 12px;text-align:left;font-weight:700;font-size:11px;'
        'text-transform:uppercase;letter-spacing:.06em;">Mission</th>'
        '<th style="padding:10px 12px;text-align:left;font-weight:700;font-size:11px;'
        'text-transform:uppercase;letter-spacing:.06em;">Status</th>'
        '<th style="padding:10px 12px;text-align:left;font-weight:700;font-size:11px;'
        'text-transform:uppercase;letter-spacing:.06em;">Started</th>'
        '<th style="padding:10px 12px;text-align:center;font-weight:700;font-size:11px;'
        'text-transform:uppercase;letter-spacing:.06em;">Nodes</th>'
        '<th style="padding:10px 12px;text-align:center;font-weight:700;font-size:11px;'
        'text-transform:uppercase;letter-spacing:.06em;">Solutions</th>'
        '</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    # ── Solutions by run bar chart (unique index: mission snippet + experiment id suffix) ─
    if complete_runs:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="aid-section-title">Solutions per Completed Run</div>',
            unsafe_allow_html=True,
        )
        chart_rows = []
        for e in complete_runs:
            mn = e["mission_name"]
            short = mn[:28] + ("…" if len(mn) > 28 else "")
            chart_rows.append(
                {
                    "run": f"{short} ({e['id'][-12:]})",
                    "Solutions": e.get("total_solutions", 0),
                    "Nodes": e.get("nodes_generated", 0),
                }
            )
        chart_df = pd.DataFrame(chart_rows).set_index("run")
        st.bar_chart(chart_df[["Solutions"]], use_container_width=True, height=220)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="aid-section-title">Nodes Generated per Completed Run</div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(chart_df[["Nodes"]], use_container_width=True, height=220)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

_init_state()

st.markdown(
    "<h2 style='margin-bottom:0;'>🌳 Aideator</h2>"
    "<p style='color:#64748b;margin-top:4px;margin-bottom:8px;'>"
    "LLM-powered idea tree builder using structured deliberation.</p>",
    unsafe_allow_html=True,
)

# Use segmented_control (not st.tabs) so only the selected view runs — avoids running
# Builder + Runner + History on every rerun (LLM init side effects, etc.).
_tab = st.segmented_control(
    "Main view",
    options=("builder", "runner", "history", "metrics"),
    format_func=lambda x: {
        "builder": "🌳  Builder",
        "runner":  "🚀  Runner",
        "history": "📜  History",
        "metrics": "📊  Metrics",
    }[x],
    label_visibility="collapsed",
    key="main_view_tab",
    default="builder",
)
st.divider()

_tab = _tab or "builder"

if _should_autorefresh_running_experiment(_tab):
    st_autorefresh(interval=4_000, limit=None, key="running_exp_poll")

if _tab == "builder":
    _render_builder()
elif _tab == "runner":
    _render_runner()
elif _tab == "history":
    _render_experiment_history_page()
elif _tab == "metrics":
    _render_metrics()

_maybe_show_delete_dialog()
_maybe_show_reset_dialog()
_maybe_show_save_dialog()
