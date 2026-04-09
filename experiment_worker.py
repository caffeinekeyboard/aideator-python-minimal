"""experiment_worker.py
======================
Standalone background worker launched by web_app.py as a subprocess.

Usage:
    python experiment_worker.py <experiment_dir>

Reads config.json from the experiment directory and writes:
  - status.json   current state (starting / running / complete / failed)
  - log.jsonl     append-only event log (one JSON object per line)
  - results.json  final tree + solutions list (written only on completion)
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from aideator.engine import IdeaEngine
from aideator.models import PostType
from aideator.serialization import tree_to_dict
from experiment_runner import robust_propose_achiever


# ── File helpers (atomic writes to avoid partial-read corruption) ─────────────

def _write_status(exp_dir: Path, data: dict) -> None:
    data["updated_at"] = datetime.now().isoformat()
    tmp = exp_dir / "status.json.tmp"
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(exp_dir / "status.json")


def _append_log(exp_dir: Path, entry: dict) -> None:
    entry.setdefault("time", datetime.now().isoformat())
    with open(exp_dir / "log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main worker logic ─────────────────────────────────────────────────────────

def run(exp_dir_str: str) -> None:
    exp_dir = Path(exp_dir_str)

    config = json.loads((exp_dir / "config.json").read_text())
    mission_name = config["mission_name"]
    mission_desc = config["mission_desc"]
    pipeline = [(PostType(pt), b) for pt, b in config["pipeline"]]

    _write_status(exp_dir, {
        "state": "running",
        "layer": 0,
        "total_layers": len(pipeline),
        "current_type": None,
        "nodes_generated": 1,
    })
    _append_log(exp_dir, {"event": "start", "mission": mission_name})

    engine = IdeaEngine()
    root = engine.create_mission(mission_name, mission_desc)
    current_nodes = [root]
    nodes_generated = 1

    for layer_idx, (target_type, branching) in enumerate(pipeline, 1):
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
            "parents": len(current_nodes),
            "expected": len(current_nodes) * branching,
        })

        next_layer: list = []
        for parent in current_nodes:
            for _ in range(branching):
                try:
                    post = robust_propose_achiever(engine, target_type, parent)
                except Exception as exc:
                    _append_log(exp_dir, {"event": "error", "message": str(exc)})
                    continue
                if post:
                    next_layer.append(post)
                    nodes_generated += 1
                    _append_log(exp_dir, {
                        "event": "node",
                        "layer": layer_idx,
                        "type": post.ptype.value,
                        "name": post.name,
                    })

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
        current_nodes = next_layer

    # ── Save results ──────────────────────────────────────────────────────────
    solutions = [p for p in current_nodes if p.ptype == PostType.SOLUTION]
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
