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

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from aideator.engine import IdeaEngine
from aideator.models import Post, PostType
from aideator.serialization import dict_to_tree, import_json, tree_to_dict
from aideator.transitions import get_allowed_children

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aideator",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ──────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

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
    "starting": "🔵",
    "running":  "🟡",
    "complete": "✅",
    "failed":   "❌",
}

# ── Session state ──────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults: dict = {
        "root":          None,   # builder: active tree root
        "selected_id":   None,   # builder: selected node id
        "engine":        None,   # builder: IdeaEngine instance
        "viewed_exp_id": None,   # runner:  experiment currently on display
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
    """Render the tree as styled HTML. Returns (html, {num: Post})."""
    index: dict[int, Post] = {}
    rows: list[str] = []

    for num, post, depth in _all_nodes(root):
        index[num] = post
        icon   = PTYPE_ICON.get(post.ptype, "•")
        color  = PTYPE_COLOR.get(post.ptype, "#333")
        is_sel = post.id == selected_id

        indent_px  = depth * 22
        row_bg     = "background:#dbeafe;border-left:3px solid #2563eb;" if is_sel else ""
        name_style = "font-weight:700;" if is_sel else "font-weight:400;"

        rows.append(
            f'<div style="display:flex;align-items:center;padding:3px 8px;'
            f'border-radius:5px;margin:2px 0;{row_bg}">'
            f'<span style="min-width:{indent_px}px;display:inline-block;"></span>'
            f'<span style="margin-right:5px;font-size:14px;">{icon}</span>'
            f'<span style="font-size:10px;font-weight:700;color:{color};'
            f'background:{color}18;padding:1px 6px;border-radius:3px;'
            f'margin-right:6px;white-space:nowrap;">{post.ptype.value}</span>'
            f'<span style="{name_style}font-size:13px;color:#1e293b;">'
            f'<span style="color:#94a3b8;margin-right:4px;">{num}.</span>'
            f'{post.name}</span>'
            f'</div>'
        )

    scroll = "max-height:440px;overflow-y:auto;" if scrollable else ""
    html = (
        f'<div style="font-family:\'SF Mono\',\'Fira Code\',monospace;line-height:1.7;'
        f'padding:10px;background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;{scroll}">'
        + "".join(rows) + "</div>"
    )
    return html, index


# ═══════════════════════════════════════════════════════════════════════════════
# BUILDER TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_builder() -> None:
    if st.session_state.root is None:
        col_new, col_load = st.columns(2, gap="large")

        with col_new:
            st.markdown("#### New Mission")
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
            st.markdown("#### Load from File")
            with st.form("load_form"):
                uploaded = st.file_uploader("Upload a saved JSON tree", type="json")
                if st.form_submit_button("📂 Load Tree", use_container_width=True) and uploaded:
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
        return

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
                "Node number", min_value=1, max_value=len(index),
                step=1, value=1, label_visibility="collapsed",
            )
        with c_btn:
            if st.button("Select", use_container_width=True):
                st.session_state.selected_id = index[int(node_num)].id
                st.rerun()

    with col_panel:
        if selected:
            color = PTYPE_COLOR.get(selected.ptype, "#333")
            icon  = PTYPE_ICON.get(selected.ptype, "")
            st.markdown(
                f'<div style="padding:14px 16px;background:#f0f9ff;border-radius:10px;'
                f'border-left:4px solid {color};margin-bottom:16px;">'
                f'<div style="font-size:11px;color:{color};font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.05em;">{icon} {selected.ptype.value}</div>'
                f'<div style="font-size:17px;font-weight:700;color:#0f172a;margin-top:4px;">'
                f'{selected.name}</div>'
                f'<div style="font-size:13px;color:#475569;margin-top:8px;line-height:1.6;">'
                f'{selected.description}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        allowed = get_allowed_children(selected.ptype) if selected else []
        if allowed:
            st.markdown("**Propose a child node**")
            child_type = st.selectbox(
                "Child type", options=allowed,
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
            color = PTYPE_COLOR.get(selected.ptype, "#94a3b8")
            st.markdown(
                f'<div style="padding:10px;background:#f1f5f9;border-radius:8px;'
                f'font-size:13px;color:#64748b;">No children allowed for '
                f'<b style="color:{color}">{selected.ptype.value}</b>.</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.download_button(
            "⬇️ Download Tree (JSON)",
            data=json.dumps(tree_to_dict(root), indent=2),
            file_name="aideator_tree.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("🗑️ Reset Tree", use_container_width=True):
            st.session_state.root        = None
            st.session_state.selected_id = None
            st.session_state.engine      = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER — persistence helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _exp_slug(name: str) -> str:
    slug = re.sub(r"[^\w\s]", "", name.lower()).strip()
    slug = re.sub(r"\s+", "_", slug)[:30]
    return slug or "experiment"


def _list_experiments() -> list[dict]:
    """Return all experiment metadata sorted newest-first."""
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
        try:
            config = json.loads(config_f.read_text())
            status = json.loads(status_f.read_text()) if status_f.exists() else {"state": "unknown"}
        except (json.JSONDecodeError, OSError):
            continue
        results.append({
            "id":              exp_dir.name,
            "path":            str(exp_dir),
            "mission_name":    config.get("mission_name", "Unknown"),
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
    if not f.exists():
        return {"state": "unknown"}
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return {"state": "unknown"}


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


def _read_results(exp_id: str) -> dict | None:
    f = EXPERIMENTS_DIR / exp_id / "results.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _launch_experiment(
    mission_name: str,
    mission_desc: str,
    pipeline: list[tuple[PostType, int]],
) -> str:
    """Write config, create experiment directory, launch worker subprocess."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    exp_id  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_exp_slug(mission_name)}"
    exp_dir = EXPERIMENTS_DIR / exp_id
    exp_dir.mkdir()

    config = {
        "mission_name": mission_name,
        "mission_desc": mission_desc,
        "pipeline":     [[pt.value, b] for pt, b in pipeline],
        "started_at":   datetime.now().isoformat(),
        "experiment_id": exp_id,
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "status.json").write_text(json.dumps({"state": "starting"}, indent=2))
    (exp_dir / "log.jsonl").write_text("")

    worker = Path(__file__).parent / "experiment_worker.py"
    subprocess.Popen(
        [sys.executable, str(worker), str(exp_dir)],
        cwd=str(Path(__file__).parent),
    )
    return exp_id


def _rerun_experiment(source_exp_id: str) -> str:
    """Start a new experiment using mission + pipeline from an existing run."""
    config_f = EXPERIMENTS_DIR / source_exp_id / "config.json"
    if not config_f.exists():
        raise FileNotFoundError(f"No config for experiment {source_exp_id!r}")
    config = json.loads(config_f.read_text())
    raw_pipeline = config.get("pipeline")
    if not raw_pipeline:
        raise ValueError("Saved experiment has no pipeline; cannot rerun.")
    pipeline = [(PostType(pt), int(b)) for pt, b in raw_pipeline]
    return _launch_experiment(
        str(config["mission_name"]).strip(),
        str(config["mission_desc"]).strip(),
        pipeline,
    )


def _resume_experiment_worker(exp_id: str) -> None:
    """Start (or restart) the background worker for an existing experiment directory.

    Use when status is starting/running but the process died, or after a failure.
    Re-runs the full pipeline in that folder (same as a fresh worker launch).
    """
    exp_dir = EXPERIMENTS_DIR / exp_id
    if not (exp_dir / "config.json").exists():
        raise FileNotFoundError(f"No config for experiment {exp_id!r}")
    worker = Path(__file__).parent / "experiment_worker.py"
    subprocess.Popen(
        [sys.executable, str(worker), str(exp_dir.resolve())],
        cwd=str(Path(__file__).parent.resolve()),
    )


def _delete_experiment_dir(exp_id: str) -> None:
    """Remove an experiment directory and clear selection if it was open."""
    p = EXPERIMENTS_DIR / exp_id
    if p.exists() and p.is_dir():
        shutil.rmtree(p)
    if st.session_state.get("viewed_exp_id") == exp_id:
        st.session_state.viewed_exp_id = None


@st.dialog("Delete experiment?")
def _delete_experiment_dialog(exp_id: str) -> None:
    st.markdown(f"Remove **`{exp_id}`** and all files under it? This cannot be undone.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Delete permanently", type="primary", use_container_width=True):
            _delete_experiment_dir(exp_id)
            st.session_state.pop("pending_delete_exp", None)
            st.rerun()
    with c2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.pop("pending_delete_exp", None)
            st.rerun()


def _experiment_can_resume(state: str) -> bool:
    return state in ("starting", "running", "failed")


def _maybe_show_delete_dialog() -> None:
    """Open delete confirmation when `pending_delete_exp` was set by a Delete button."""
    exp_id = st.session_state.get("pending_delete_exp")
    if exp_id:
        _delete_experiment_dialog(exp_id)


# ── Live experiment progress (periodic fragment rerun, no parent window reload) ─

def _render_experiment_live_progress(exp_id: str) -> None:
    """Poll status/log while worker runs; uses st.fragment instead of JS page reload."""

    @st.fragment(run_every=timedelta(seconds=4))
    def _live_poll() -> None:
        status = _read_status(exp_id)
        state = status.get("state", "unknown")
        if state not in ("starting", "running"):
            st.rerun()
            return

        layer = status.get("layer", 0)
        total_lay = status.get("total_layers", 1)
        nodes = status.get("nodes_generated", 0)
        cur_type = status.get("current_type") or "—"
        progress = layer / total_lay if total_lay else 0

        st.progress(progress, text=f"Layer {layer} / {total_lay}  ·  {cur_type.upper()}")
        st.caption(f"Nodes generated so far: **{nodes}**")

        log_entries = _read_log(exp_id)
        node_entries = [e for e in log_entries if e.get("event") == "node"]

        if node_entries:
            st.markdown("**Recently generated nodes**")
            rows = []
            for e in node_entries[-30:]:
                ptype_str = e.get("type", "")
                try:
                    icon = PTYPE_ICON.get(PostType(ptype_str), "•")
                    color = PTYPE_COLOR.get(PostType(ptype_str), "#333")
                except ValueError:
                    icon, color = "•", "#333"
                rows.append(
                    f'<div style="padding:3px 8px;font-size:13px;">'
                    f'{icon} <span style="font-size:10px;color:{color};font-weight:700;'
                    f'background:{color}18;padding:1px 5px;border-radius:3px;">'
                    f'{ptype_str}</span> {e.get("name", "")}</div>'
                )
            st.markdown(
                '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
                'max-height:280px;overflow-y:auto;padding:6px;">'
                + "".join(rows) + "</div>",
                unsafe_allow_html=True,
            )

    _live_poll()


# ── Experiment detail view ────────────────────────────────────────────────────

def _render_experiment_details(exp_id: str) -> None:
    config_f = EXPERIMENTS_DIR / exp_id / "config.json"
    if not config_f.exists():
        st.warning("Experiment data not found.")
        return

    config = json.loads(config_f.read_text())
    status = _read_status(exp_id)
    state  = status.get("state", "unknown")

    # ── Header ────────────────────────────────────────────────────────────────
    badge = STATE_ICON.get(state, "⚪")
    started = config.get("started_at", "")[:19].replace("T", " ")
    st.markdown(
        f'<div style="padding:12px 16px;background:#f8fafc;border-radius:10px;'
        f'border:1px solid #e2e8f0;margin-bottom:16px;">'
        f'<div style="font-size:18px;font-weight:700;color:#0f172a;">'
        f'{badge} {config["mission_name"]}</div>'
        f'<div style="font-size:12px;color:#94a3b8;margin-top:4px;">'
        f'Started: {started} &nbsp;·&nbsp; ID: <code>{exp_id}</code></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    d_resume, d_del = st.columns(2)
    with d_resume:
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
    with d_del:
        if st.button(
            "🗑️ Delete run",
            key=f"detail_del_{exp_id}",
            use_container_width=True,
        ):
            st.session_state.pending_delete_exp = exp_id
            st.rerun()

    # ── Running: live progress (fragment refresh — avoids full-page reload loops) ─
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

        results = _read_results(exp_id)
        if results:
            tab_tree, tab_solutions, tab_dl = st.tabs(["🌳 Tree", "✅ Solutions", "⬇️ Download"])

            with tab_tree:
                root = dict_to_tree(results["tree"])
                html, _ = _tree_html(root, scrollable=True)
                st.markdown(html, unsafe_allow_html=True)

            with tab_solutions:
                sols = results.get("solutions", [])
                if not sols:
                    st.warning("No solutions found.")
                else:
                    for i, sol in enumerate(sols, 1):
                        with st.expander(f"{i}. {sol['name']}", expanded=(i <= 3)):
                            st.markdown(sol["description"])

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

    else:
        st.info(f"Status: {state}")


# ── Experiment History (list + inline detail panel) ────────────────────────────

def _render_experiment_history_cards(key_prefix: str) -> None:
    """Experiment list with Open / Rerun / Resume / Delete."""
    experiments = _list_experiments()
    if not experiments:
        st.caption("No experiments yet. Go to **Experiment Runner** and launch one.")
        return

    for exp in experiments:
        icon = STATE_ICON.get(exp["state"], "⚪")
        started = exp.get("started_at", "")[:19].replace("T", " ")
        state = exp["state"]
        suffix = ""
        if state == "complete":
            suffix = f" · {exp['total_solutions']} solutions"
        elif state == "running" or state == "starting":
            suffix = f" · layer {exp['layer']}/{exp['total_layers']}"

        is_viewing = exp["id"] == st.session_state.get("viewed_exp_id")
        eid = exp["id"]
        with st.container(border=True):
            st.markdown(f"**{icon} {exp['mission_name']}**{suffix}  ")
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


def _render_experiment_history_page() -> None:
    """List on the left; **Open** shows the same experiment details on the right."""
    st.caption("Runs live under `experiments/`. **Open** loads progress, results, and downloads in the panel beside the list.")
    # Use Streamlit primitives (no fixed hex colors) so text stays visible in dark theme.
    st.subheader("Experiment History")
    st.markdown(
        "Latest runs first. **Open** shows that run on this page (right). "
        "**Rerun** clones config into a *new* run. **▶ Resume** starts the worker again "
        "in the same folder if a run is stuck, failed, or never finished. **🗑️** removes "
        "that run from disk."
    )

    col_list, col_detail = st.columns([2, 3], gap="large")

    with col_list:
        st.markdown("##### All runs")
        # Fixed pixel height enables Streamlit’s built-in vertical scroll (see st.container docs).
        # Keys become classes like st-key-streamlit-<hash>-hist_all_runs, so plain CSS selectors
        # for st-key-hist_all_runs never matched.
        with st.container(
            key="hist_all_runs",
            border=True,
            height=560,
            autoscroll=False,
        ):
            _render_experiment_history_cards(key_prefix="histpg_")

    with col_detail:
        st.markdown("##### Experiment")
        viewed_id = st.session_state.get("viewed_exp_id")
        if viewed_id and (EXPERIMENTS_DIR / viewed_id).exists():
            _render_experiment_details(viewed_id)
        else:
            st.info(
                "Click **Open** on a run to show live progress, results, and downloads here."
            )


# ── Runner tab ────────────────────────────────────────────────────────────────

def _render_runner() -> None:
    st.caption(
        "Experiments run as **background processes** and persist across page refreshes "
        "and tab switches. Results are saved to `experiments/` automatically."
    )

    col_left, col_right = st.columns([2, 3], gap="large")

    with col_left:
        # ── New experiment form ────────────────────────────────────────────────
        with st.expander("➕ New Experiment", expanded=True):
            with st.form("runner_form"):
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
                b_goal     = st.slider("🏆  Goals / Stakeholder",     1, 4, 2)
                b_abs      = st.slider("💡  Abstractions / Goal",      1, 4, 2)
                b_analogy  = st.slider("🔄  Analogies / Abstraction",  1, 6, 4)
                b_insp     = st.slider("⚡  Inspirations / Analogy",   1, 4, 2)
                b_solution = st.slider("✅  Solutions / Inspiration",  1, 4, 3)
                est = b_goal * b_abs * b_analogy * b_insp * b_solution
                st.caption(f"Theoretical max solutions: **{est}**")

                if st.form_submit_button("🚀 Launch Experiment", type="primary", use_container_width=True):
                    if not mission_name.strip() or not mission_desc.strip():
                        st.error("Mission name and description are required.")
                    else:
                        pipeline = [
                            (PostType.STAKEHOLDER, 1),
                            (PostType.GOAL,        b_goal),
                            (PostType.ABSTRACTION, b_abs),
                            (PostType.ANALOGY,     b_analogy),
                            (PostType.INSPIRATION, b_insp),
                            (PostType.SOLUTION,    b_solution),
                        ]
                        exp_id = _launch_experiment(
                            mission_name.strip(), mission_desc.strip(), pipeline
                        )
                        st.session_state.viewed_exp_id = exp_id
                        st.rerun()

    with col_right:
        viewed_id = st.session_state.get("viewed_exp_id")
        if viewed_id and (EXPERIMENTS_DIR / viewed_id).exists():
            _render_experiment_details(viewed_id)
        else:
            st.info(
                "Launch a new experiment or open a run from the **Experiment History** "
                "tab to view live progress or results."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

_init_state()

st.markdown(
    "<h2 style='margin-bottom:0;'>🌳 Aideator</h2>"
    "<p style='color:#64748b;margin-top:4px;margin-bottom:24px;'>"
    "LLM-powered idea tree builder using structured deliberation.</p>",
    unsafe_allow_html=True,
)

# Use radio (not st.tabs) so only the selected view runs — avoids running Builder +
# Runner + History on every rerun (fragments, LLM init side effects, etc.).
_tab = st.radio(
    "Main view",
    options=("builder", "runner", "history"),
    format_func=lambda x: (
        "🌳  Interactive Builder"
        if x == "builder"
        else "🚀  Experiment Runner"
        if x == "runner"
        else "📜  Experiment History"
    ),
    horizontal=True,
    label_visibility="collapsed",
    key="main_view_tab",
)

if _tab == "builder":
    _render_builder()
elif _tab == "runner":
    _render_runner()
else:
    _render_experiment_history_page()

_maybe_show_delete_dialog()
