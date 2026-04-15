"""experiment_worker.py
======================
Standalone background worker launched by web_app.py as a subprocess.

Usage:
    python experiment_worker.py <experiment_dir>

Reads config.json from the experiment directory and writes:
  - status.json   current state (starting / running / complete / failed)
  - log.jsonl     append-only event log (one JSON object per line)
  - results.json  final tree + solutions list (written only on completion)

Parallelism
-----------
LLM calls are parallelised *across parent nodes* within each layer using a
ThreadPoolExecutor.  Children of the same parent are still generated
sequentially so that each sibling prompt can include the already-generated
siblings for diversity.  Configure concurrency via env vars:

  EXPERIMENT_MAX_CONCURRENT      (default 4) — max parallel parent threads
  EXPERIMENT_INTRA_PARENT_DELAY  (default 0.0) — seconds between sibling calls
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from aideator.engine import IdeaEngine
from aideator.llm import LLMClient
from aideator.models import Post, PostType
from aideator.serialization import dict_to_tree, tree_to_dict
from aideator.transitions import validate_transition
from experiment_runner import (
    experiment_gemini_model,
    experiment_request_delay_seconds,
    robust_propose_achiever,
)


# ── Tree helpers ──────────────────────────────────────────────────────────────

def _collect_by_type(root: Post, ptype: PostType) -> list[Post]:
    """Walk the tree (BFS) and return all Post nodes of the given type."""
    results = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.ptype == ptype:
            results.append(node)
        stack.extend(node.achievers)
    return results


def _collect_by_types(root: Post, ptypes: list[PostType]) -> list[Post]:
    """Walk the tree (BFS) and return all Post nodes matching any type, de-duped by id."""
    want = set(ptypes)
    results: list[Post] = []
    seen: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.ptype in want and node.id not in seen:
            seen.add(node.id)
            results.append(node)
        stack.extend(node.achievers)
    return results


def _count_all(node: Post) -> int:
    """Count total nodes in the subtree rooted at node."""
    return 1 + sum(_count_all(c) for c in node.achievers)


def _resolve_parent_type_for_layer(
    pipeline: list[tuple[PostType, int]], layer_pi: int
) -> PostType:
    """Which parent node type receives children in pipeline layer ``layer_pi`` (0-based).

    The web UI omits stages with branching 0, so the previous *listed* stage may
    not legally parent the current target (e.g. stakeholder → solution). Walk
    backward along earlier pipeline targets, then allow mission as a host when
    valid (mission → solution).
    """
    if layer_pi == 0:
        return PostType.MISSION

    cur = pipeline[layer_pi][0]
    for j in range(layer_pi - 1, -1, -1):
        t = pipeline[j][0]
        if validate_transition(t, cur):
            return t
    if validate_transition(PostType.MISSION, cur):
        return PostType.MISSION
    raise ValueError(
        f"Pipeline layer {layer_pi} ({cur.value}): no earlier listed stage (nor "
        "mission) can legally parent this type — adjust branching or ordering."
    )


# ── File helpers (atomic writes to avoid partial-read corruption) ─────────────

def _write_status(exp_dir: Path, data: dict) -> None:
    data["updated_at"] = datetime.now().isoformat()
    tmp = exp_dir / "status.json.tmp"
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(exp_dir / "status.json")


def _append_log(exp_dir: Path, entry: dict, lock: threading.Lock | None = None) -> None:
    """Append a JSON entry to log.jsonl.  Pass a lock when called from threads."""
    entry.setdefault("time", datetime.now().isoformat())
    line = json.dumps(entry) + "\n"
    if lock:
        with lock:
            with open(exp_dir / "log.jsonl", "a") as f:
                f.write(line)
    else:
        with open(exp_dir / "log.jsonl", "a") as f:
            f.write(line)


# ── Main worker logic ─────────────────────────────────────────────────────────

def run(exp_dir_str: str) -> None:
    exp_dir = Path(exp_dir_str)

    config = json.loads((exp_dir / "config.json").read_text())
    mission_name = config["mission_name"]
    mission_desc = config["mission_desc"]
    pipeline = [(PostType(pt), b) for pt, b in config["pipeline"]]
    solution_parent_types_raw = config.get("solution_parent_types") or None
    solution_parent_types: list[PostType] | None = None
    if solution_parent_types_raw:
        try:
            solution_parent_types = [PostType(str(x)) for x in solution_parent_types_raw]
        except Exception:
            solution_parent_types = None

    _write_status(exp_dir, {
        "state": "running",
        "layer": 0,
        "total_layers": len(pipeline),
        "current_type": None,
        "nodes_generated": 1,
    })
    _append_log(exp_dir, {"event": "start", "mission": mission_name})

    model = experiment_gemini_model()
    engine = IdeaEngine(LLMClient(model_name=model))
    _append_log(exp_dir, {"event": "config", "llm_model": model})

    # ── Detect edited tree: resume from results.json if edited_at is present ──
    results_f = exp_dir / "results.json"
    resuming = (
        results_f.exists()
        and json.loads(results_f.read_text()).get("edited_at")
    )
    if resuming:
        existing_data = json.loads(results_f.read_text())
        root = dict_to_tree(existing_data["tree"])
        nodes_generated = _count_all(root)
        _append_log(exp_dir, {
            "event": "resume_from_edited_tree",
            "message": "Loaded edited tree from results.json",
            "existing_nodes": nodes_generated,
        })
    else:
        root = engine.create_mission(mission_name, mission_desc)
        nodes_generated = 1

    # Concurrency settings
    max_workers = max(1, int(os.getenv("EXPERIMENT_MAX_CONCURRENT", "4")))
    intra_delay = max(0.0, float(os.getenv("EXPERIMENT_INTRA_PARENT_DELAY", "0.0")))

    for layer_idx, (target_type, branching) in enumerate(pipeline, start=1):
        layer_pi = layer_idx - 1
        parent_type = _resolve_parent_type_for_layer(pipeline, layer_pi)
        # When resuming from an edited tree, discover parents from the tree
        # rather than tracking current_nodes across layers — handles gaps/additions.
        #
        # Special case: C2-style runs can generate solutions under any "challenge"
        # node (goal/barrier/cause) rather than only the previous stage.
        if target_type == PostType.SOLUTION and solution_parent_types:
            # Prefer challenge-parent generation (goal/barrier/cause) when configured.
            # Fallback to the standard parent type (e.g. mission/inspiration) so
            # mission-only pipelines still work when no challenge nodes exist.
            all_parents = _collect_by_types(root, solution_parent_types)
            if not all_parents:
                all_parents = _collect_by_type(root, parent_type)
        else:
            all_parents = _collect_by_type(root, parent_type)
        if not all_parents:
            _write_status(exp_dir, {
                "state": "failed",
                "reason": f"Layer {layer_idx}: no parent nodes of type {parent_type.value} found.",
                "nodes_generated": nodes_generated,
            })
            _append_log(exp_dir, {"event": "failed", "layer": layer_idx,
                                   "reason": f"no {parent_type.value} nodes"})
            return

        _write_status(exp_dir, {
            "state": "running",
            "layer": layer_idx,
            "total_layers": len(pipeline),
            "current_type": target_type.value,
            "nodes_generated": nodes_generated,
        })
        _append_log(exp_dir, {
            "event": "layer_start",
            "layer": layer_idx,
            "type": target_type.value,
            "parents": len(all_parents),
            "expected": len(all_parents) * branching,
        })

        # Shared mutable state accessed from worker threads
        log_lock = threading.Lock()
        counter_lock = threading.Lock()
        nodes_generated_local = [0]  # list so threads can mutate via closure

        def _generate_for_parent(parent, _layer=layer_idx, _type=target_type, _b=branching):
            """Grow ``parent`` until it has ``_b`` children of ``_type`` (sequential LLM calls).

            ``build_post`` dedupes same-type siblings with the same normalised name and
            returns the existing node without attaching a duplicate. The old loop
            treated that as success, so mission-only runs often stopped at one solution.
            Here we only count progress when a *new* child id appears under ``parent``.
            """
            target_total = _b
            max_attempts = max(80, (target_total + 1) * 25)
            attempts = 0
            while len([c for c in parent.achievers if c.ptype == _type]) < target_total:
                if attempts >= max_attempts:
                    _append_log(
                        exp_dir,
                        {
                            "event": "warn",
                            "message": (
                                f"Stopped before {target_total} {_type.value} children "
                                f"for '{parent.name}' after {max_attempts} attempts "
                                "(name dedupe repeats and/or API errors)."
                            ),
                        },
                        log_lock,
                    )
                    break
                attempts += 1
                before_ids = {id(c) for c in parent.achievers if c.ptype == _type}
                try:
                    robust_propose_achiever(engine, _type, parent)
                except Exception as exc:
                    _append_log(exp_dir, {"event": "error", "message": str(exc)}, log_lock)
                    continue
                new_children = [
                    c
                    for c in parent.achievers
                    if c.ptype == _type and id(c) not in before_ids
                ]
                if not new_children:
                    continue
                new_post = new_children[-1]
                with counter_lock:
                    nodes_generated_local[0] += 1
                _append_log(exp_dir, {
                    "event": "node",
                    "layer": _layer,
                    "type": new_post.ptype.value,
                    "name": new_post.name,
                }, log_lock)
                if intra_delay > 0:
                    time.sleep(intra_delay)
            return [c for c in parent.achievers if c.ptype == _type]

        next_layer: list = []

        if len(all_parents) == 1:
            # Single parent — no benefit from a thread pool, run inline
            next_layer = _generate_for_parent(all_parents[0])
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(all_parents))) as executor:
                futures = {
                    executor.submit(_generate_for_parent, parent): parent
                    for parent in all_parents
                }
                for future in as_completed(futures):
                    try:
                        posts = future.result()
                        next_layer.extend(posts)
                    except Exception as exc:
                        _append_log(exp_dir, {
                            "event": "error",
                            "message": f"Parent thread failed: {exc}",
                        }, log_lock)

        nodes_generated += nodes_generated_local[0]

        if not next_layer:
            _write_status(exp_dir, {
                "state": "failed",
                "reason": f"Layer {layer_idx} ({target_type.value}) produced no nodes.",
                "nodes_generated": nodes_generated,
            })
            _append_log(exp_dir, {"event": "failed", "layer": layer_idx})
            return

        _append_log(exp_dir, {
            "event": "layer_done",
            "layer": layer_idx,
            "generated": len(next_layer),
        })

    # ── Save results ──────────────────────────────────────────────────────────
    solutions = _collect_by_type(root, PostType.SOLUTION)
    results = {
        "tree": tree_to_dict(root),
        "solutions": [
            {"name": s.name, "description": s.description}
            for s in solutions
        ],
        "total_solutions": len(solutions),
        "completed_at": datetime.now().isoformat(),
    }
    (exp_dir / "results.json").write_text(json.dumps(results, indent=2))

    _write_status(exp_dir, {
        "state": "complete",
        "nodes_generated": nodes_generated,
        "total_solutions": len(solutions),
        "completed_at": datetime.now().isoformat(),
    })
    _append_log(exp_dir, {
        "event": "complete",
        "total_solutions": len(solutions),
        "nodes_generated": nodes_generated,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python experiment_worker.py <experiment_dir>")
        sys.exit(1)

    _exp_dir = sys.argv[1]
    try:
        run(_exp_dir)
    except Exception as _e:
        try:
            _write_status(Path(_exp_dir), {
                "state": "failed",
                "error": str(_e),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
        raise
