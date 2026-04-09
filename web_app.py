"""web_app.py — Streamlit web UI for Aideator.

Run with:
    streamlit run web_app.py

Two modes:
  - Interactive Builder : build an idea tree node-by-node, guided by the LLM
  - Experiment Runner   : automated breadth-first pipeline with live progress
"""

from __future__ import annotations

import json
import os
import tempfile

import streamlit as st

from aideator.engine import IdeaEngine
from aideator.models import Post, PostType
from aideator.serialization import export_json, import_json, tree_to_dict
from aideator.transitions import get_allowed_children, ACTION_NAMES
from experiment_runner import WORKFLOW_PIPELINE, robust_propose_achiever

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aideator",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Type styling ─────────────────────────────────────────────────────────────
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

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "root":          None,   # builder tree root Post
        "selected_id":   None,   # selected node id in builder
        "engine":        None,   # IdeaEngine instance
        "exp_root":      None,   # experiment runner result root
        "exp_solutions": [],     # experiment runner flat solutions list
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Tree helpers ──────────────────────────────────────────────────────────────
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
    sid = st.session_state.selected_id
    if root and sid:
        return _find_by_id(root, sid)
    return None


# ── HTML tree renderer ────────────────────────────────────────────────────────
def _tree_html(root: Post, selected_id: str | None = None, scrollable: bool = False) -> tuple[str, dict[int, Post]]:
    """Render the full tree as styled HTML.

    Returns:
        (html_string, {number: Post}) index mapping.
    """
    index: dict[int, Post] = {}
    rows: list[str] = []

    for num, post, depth in _all_nodes(root):
        index[num] = post
        icon  = PTYPE_ICON.get(post.ptype, "•")
        color = PTYPE_COLOR.get(post.ptype, "#333")
        is_sel = post.id == selected_id

        indent_px = depth * 22
        row_bg    = "background:#dbeafe;border-left:3px solid #2563eb;" if is_sel else ""
        name_style = "font-weight:700;color:#1e293b;" if is_sel else "font-weight:400;color:#1e293b;"

        rows.append(
            f'<div style="display:flex;align-items:center;padding:3px 8px;'
            f'border-radius:5px;margin:2px 0;{row_bg}">'
            f'<span style="min-width:{indent_px}px;display:inline-block;"></span>'
            f'<span style="margin-right:5px;font-size:14px;">{icon}</span>'
            f'<span style="font-size:10px;font-weight:700;color:{color};'
            f'background:{color}18;padding:1px 6px;border-radius:3px;'
            f'margin-right:6px;white-space:nowrap;">{post.ptype.value}</span>'
            f'<span style="{name_style}font-size:13px;">'
            f'<span style="color:#94a3b8;margin-right:4px;">{num}.</span>'
            f'{post.name}</span>'
            f'</div>'
        )

    scroll_style = "max-height:420px;overflow-y:auto;" if scrollable else ""
    html = (
        f'<div style="font-family:\'SF Mono\',\'Fira Code\',monospace;'
        f'line-height:1.7;padding:10px;background:#f8fafc;border-radius:10px;'
        f'border:1px solid #e2e8f0;{scroll_style}">'
        + "".join(rows)
        + "</div>"
    )
    return html, index


# ── Builder tab ───────────────────────────────────────────────────────────────
def _render_builder() -> None:
    # ── No tree yet: show init UI ────────────────────────────────────────────
    if st.session_state.root is None:
        col_new, col_load = st.columns(2, gap="large")

        with col_new:
            st.markdown("#### New Mission")
            with st.form("new_mission_form"):
                name = st.text_input("Mission name", placeholder="e.g. Urban Transportation Reform")
                desc = st.text_area(
                    "Mission description",
                    placeholder="Describe the deliberation scope, constraints, and goals…",
                    height=120,
                )
                if st.form_submit_button("✨ Create Mission", type="primary", use_container_width=True):
                    if not name.strip() or not desc.strip():
                        st.error("Both name and description are required.")
                    else:
                        try:
                            engine = IdeaEngine()
                            root = engine.create_mission(name.strip(), desc.strip())
                            st.session_state.engine = engine
                            st.session_state.root = root
                            st.session_state.selected_id = root.id
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to initialise engine: {e}")

        with col_load:
            st.markdown("#### Load from File")
            with st.form("load_form"):
                uploaded = st.file_uploader("Upload a saved JSON tree", type="json")
                if st.form_submit_button("📂 Load Tree", use_container_width=True) and uploaded:
                    try:
                        data = json.load(uploaded)
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False
                        ) as f:
                            json.dump(data, f)
                            tmp = f.name
                        root = import_json(tmp)
                        os.unlink(tmp)
                        st.session_state.engine = IdeaEngine()
                        st.session_state.root = root
                        st.session_state.selected_id = root.id
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {e}")
        return

    # ── Tree + action panel ──────────────────────────────────────────────────
    root     = st.session_state.root
    selected = _get_selected()

    col_tree, col_panel = st.columns([3, 2], gap="large")

    with col_tree:
        st.markdown("##### Idea Tree")
        tree_html, index = _tree_html(root, st.session_state.selected_id, scrollable=True)
        st.markdown(tree_html, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        c_input, c_btn = st.columns([3, 1])
        with c_input:
            node_num = st.number_input(
                "Node number", min_value=1, max_value=len(index), step=1,
                value=1, label_visibility="collapsed",
            )
        with c_btn:
            if st.button("Select", use_container_width=True):
                st.session_state.selected_id = index[int(node_num)].id
                st.rerun()

    with col_panel:
        # ── Selected node card ───────────────────────────────────────────────
        if selected:
            color = PTYPE_COLOR.get(selected.ptype, "#333")
            icon  = PTYPE_ICON.get(selected.ptype, "")
            st.markdown(
                f'<div style="padding:14px 16px;background:#f0f9ff;border-radius:10px;'
                f'border-left:4px solid {color};margin-bottom:16px;">'
                f'<div style="font-size:11px;color:{color};font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.05em;">'
                f'{icon} {selected.ptype.value}</div>'
                f'<div style="font-size:17px;font-weight:700;color:#0f172a;margin-top:4px;">'
                f'{selected.name}</div>'
                f'<div style="font-size:13px;color:#475569;margin-top:8px;line-height:1.6;">'
                f'{selected.description}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Add child ────────────────────────────────────────────────────────
        allowed = get_allowed_children(selected.ptype) if selected else []
        if allowed:
            st.markdown("**Propose a child node**")
            child_type = st.selectbox(
                "Child type",
                options=allowed,
                format_func=lambda t: f"{PTYPE_ICON.get(t, '')}  {t.value}",
                label_visibility="collapsed",
            )
            if st.button("🤖 Propose with AI", type="primary", use_container_width=True):
                with st.spinner(f"Asking the LLM to propose a {child_type.value}…"):
                    try:
                        new_post = st.session_state.engine.propose_achiever(child_type, selected)
                        st.session_state.selected_id = new_post.id
                        st.toast(f'Created: "{new_post.name}"', icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        elif selected:
            ptype_color = PTYPE_COLOR.get(selected.ptype, "#94a3b8")
            st.markdown(
                f'<div style="padding:10px;background:#f1f5f9;border-radius:8px;'
                f'font-size:13px;color:#64748b;">No children allowed for '
                f'<b style="color:{ptype_color}">{selected.ptype.value}</b>.</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Save / reset ─────────────────────────────────────────────────────
        tree_json = json.dumps(tree_to_dict(root), indent=2)
        st.download_button(
            "⬇️ Download Tree (JSON)",
            data=tree_json,
            file_name="aideator_tree.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("🗑️ Reset Tree", use_container_width=True):
            st.session_state.root = None
            st.session_state.selected_id = None
            st.session_state.engine = None
            st.rerun()


# ── Experiment runner tab ─────────────────────────────────────────────────────
def _run_pipeline(
    mission_name: str,
    mission_desc: str,
    pipeline: list[tuple[PostType, int]],
) -> tuple[Post, list[Post]]:
    """Execute the pipeline layer-by-layer with live st.status updates."""
    engine = IdeaEngine()
    root = engine.create_mission(mission_name, mission_desc)
    current_nodes: list[Post] = [root]

    with st.status("Running pipeline…", expanded=True) as status:
        st.write(f"**Mission:** {mission_name}")
        total = len(pipeline)

        for layer_idx, (target_type, branching) in enumerate(pipeline, 1):
            expected = len(current_nodes) * branching
            st.write(
                f"**Layer {layer_idx}/{total} — {target_type.value.upper()}** "
                f"({branching} per parent · ~{expected} nodes expected)"
            )
            next_layer: list[Post] = []

            for parent in current_nodes:
                for _ in range(branching):
                    try:
                        post = robust_propose_achiever(engine, target_type, parent)
                    except Exception as e:
                        st.warning(f"Skipped: {e}")
                        continue
                    if post:
                        next_layer.append(post)
                        st.write(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;{PTYPE_ICON.get(target_type, '•')} "
                            f"`{post.name}`",
                            unsafe_allow_html=True,
                        )

            if not next_layer:
                status.update(label="Pipeline stopped — no nodes produced.", state="error")
                return root, []

            current_nodes = next_layer

        solutions = [p for p in current_nodes if p.ptype == PostType.SOLUTION]
        status.update(
            label=f"✅ Complete — {len(solutions)} solutions generated.",
            state="complete",
            expanded=False,
        )

    return root, solutions


def _render_results(root: Post, solutions: list[Post]) -> None:
    tab_tree, tab_solutions = st.tabs(["🌳 Full Tree", "✅ Solutions"])

    with tab_tree:
        tree_html, _ = _tree_html(root, scrollable=True)
        st.markdown(tree_html, unsafe_allow_html=True)

    with tab_solutions:
        if not solutions:
            st.warning("No solutions were generated.")
            return

        st.markdown(
            f'<div style="font-size:13px;color:#64748b;margin-bottom:16px;">'
            f'{len(solutions)} solutions generated</div>',
            unsafe_allow_html=True,
        )
        for i, sol in enumerate(solutions, 1):
            parent_label = ""
            if sol.purpose:
                parent_label = (
                    f"Via **{sol.purpose.ptype.value}** → _{sol.purpose.name}_"
                )
            with st.expander(f"{i}. {sol.name}", expanded=(i <= 3)):
                st.markdown(sol.description)
                if parent_label:
                    st.caption(parent_label)


def _render_runner() -> None:
    st.caption(
        "Automates the full abstraction-and-analogy pipeline breadth-first. "
        "Adjust branching to control tree width and total solutions."
    )

    col_form, col_output = st.columns([2, 3], gap="large")

    with col_form:
        with st.form("runner_form"):
            st.markdown("#### Mission")
            mission_name = st.text_input(
                "Name", value="Next-Gen Grid-Scale Energy Storage"
            )
            mission_desc = st.text_area(
                "Description",
                value=(
                    "Develop novel methods to store grid-scale renewable energy "
                    "that do not rely on rare-earth lithium-ion batteries, targeting "
                    "cost, scalability, and environmental sustainability."
                ),
                height=110,
            )

            st.markdown("#### Branching Factors")
            st.caption("Number of children generated per parent at each layer.")
            b_goal     = st.slider("🏆  Goals / Stakeholder",       1, 4, 2)
            b_abs      = st.slider("💡  Abstractions / Goal",        1, 4, 2)
            b_analogy  = st.slider("🔄  Analogies / Abstraction",    1, 6, 4)
            b_insp     = st.slider("⚡  Inspirations / Analogy",     1, 4, 2)
            b_solution = st.slider("✅  Solutions / Inspiration",    1, 4, 3)

            total_est = 1 * b_goal * b_abs * b_analogy * b_insp * b_solution
            st.caption(f"Theoretical max solutions: **{total_est}**")

            run_btn = st.form_submit_button(
                "🚀 Run Experiment", type="primary", use_container_width=True
            )

        if st.session_state.exp_solutions:
            st.success(f"{len(st.session_state.exp_solutions)} solutions generated.")
            if st.session_state.exp_root:
                st.download_button(
                    "⬇️ Download Results (JSON)",
                    data=json.dumps(tree_to_dict(st.session_state.exp_root), indent=2),
                    file_name="experiment_results.json",
                    mime="application/json",
                    use_container_width=True,
                )
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.exp_root = None
                st.session_state.exp_solutions = []
                st.rerun()

    with col_output:
        if run_btn:
            pipeline = [
                (PostType.STAKEHOLDER, 1),
                (PostType.GOAL,        b_goal),
                (PostType.ABSTRACTION, b_abs),
                (PostType.ANALOGY,     b_analogy),
                (PostType.INSPIRATION, b_insp),
                (PostType.SOLUTION,    b_solution),
            ]
            try:
                root, solutions = _run_pipeline(mission_name, mission_desc, pipeline)
                st.session_state.exp_root = root
                st.session_state.exp_solutions = solutions
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

        if st.session_state.exp_root:
            _render_results(st.session_state.exp_root, st.session_state.exp_solutions)
        else:
            st.info("Configure the mission and click **Run Experiment** to start.")


# ── Entry point ───────────────────────────────────────────────────────────────
_init_state()

st.markdown(
    "<h2 style='margin-bottom:0;'>🌳 Aideator</h2>"
    "<p style='color:#64748b;margin-top:4px;margin-bottom:24px;'>"
    "LLM-powered idea tree builder using structured deliberation.</p>",
    unsafe_allow_html=True,
)

tab_builder, tab_runner = st.tabs(["🌳  Interactive Builder", "🚀  Experiment Runner"])

with tab_builder:
    _render_builder()

with tab_runner:
    _render_runner()
