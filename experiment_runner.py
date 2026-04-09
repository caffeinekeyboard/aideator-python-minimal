"""
experiment_runner.py
====================
Automated, large-scale ideation experiment using the aideator framework's
abstraction-and-analogy workflow.  Expands the idea tree layer-by-layer
(never recursive DFS) with asymmetric branching that widens at the
creative/lateral stages.

Valid transition path (from aideator/transitions.py):
  MISSION -> STAKEHOLDER -> GOAL -> ABSTRACTION -> ANALOGY -> INSPIRATION -> SOLUTION

The user-facing "PURPOSE" concept maps to STAKEHOLDER + GOAL in the engine's
transition graph.  We keep STAKEHOLDER thin (1 child) and use GOAL (branching=2)
as the effective "purpose" layer.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional

from dotenv import load_dotenv

from aideator.engine import IdeaEngine
from aideator.llm import LLMClient
from aideator.models import Post, PostType
from aideator.serialization import export_json

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Keywords that signal a retryable LLM API capacity / rate-limit error
_CAPACITY_KEYWORDS: list[str] = [
    "429",
    "503",
    "capacity",
    "quota",
    "rate",
    "overloaded",
    "unavailable",
]


def experiment_gemini_model() -> str:
    """Model for bulk experiments: GEMINI_MODEL_EXPERIMENT, else GEMINI_MODEL, else Flash."""
    load_dotenv()
    for key in ("GEMINI_MODEL_EXPERIMENT", "GEMINI_MODEL"):
        v = os.getenv(key)
        if v and v.strip():
            return v.strip()
    return "gemini-2.0-flash"


def experiment_retry_settings(
    max_retries: int | None = None,
    base_wait: float | None = None,
) -> tuple[int, float]:
    """Resolve retry count and initial backoff (seconds) from args or env."""
    load_dotenv()
    if max_retries is None:
        max_retries = max(1, int(os.getenv("EXPERIMENT_LLM_MAX_RETRIES", "10")))
    if base_wait is None:
        base_wait = max(0.5, float(os.getenv("EXPERIMENT_LLM_BACKOFF_BASE", "3.0")))
    return max_retries, base_wait


def experiment_request_delay_seconds() -> float:
    load_dotenv()
    return max(0.0, float(os.getenv("EXPERIMENT_REQUEST_DELAY_SECONDS", "0.75")))


# ---------------------------------------------------------------------------
# 1. Resilient LLM wrapper with exponential back-off
# ---------------------------------------------------------------------------
def robust_propose_achiever(
    engine: IdeaEngine,
    child_type: PostType,
    parent_post: Post,
    max_retries: int | None = None,
    base_wait: float | None = None,
) -> Optional[Post]:
    """Wrap ``engine.propose_achiever`` with exponential back-off and jitter.

    Three error categories are handled differently:

    - **Logic bugs** (ValueError, TypeError, AttributeError): these indicate a
      programming error in the pipeline configuration or prompt logic. They are
      re-raised immediately so they surface rather than being silently swallowed.
    - **Capacity / rate-limit errors**: retried up to *max_retries* times with
      exponential back-off and jitter.
    - **Other transient API errors**: logged and the branch is skipped (None
      returned) so one bad response doesn't abort the whole experiment.

    *max_retries* / *base_wait* default from env ``EXPERIMENT_LLM_MAX_RETRIES``
    (default 10) and ``EXPERIMENT_LLM_BACKOFF_BASE`` (default 3.0) when omitted.
    """
    max_retries, base_wait = experiment_retry_settings(max_retries, base_wait)

    for attempt in range(1, max_retries + 1):
        try:
            return engine.propose_achiever(child_type, parent_post)

        except (ValueError, TypeError, AttributeError) as exc:
            # Logic bugs must surface — do not silently skip
            logger.error(
                "Logic error proposing %s for '%s': %s — re-raising.",
                child_type.value,
                parent_post.name,
                exc,
            )
            raise

        except Exception as exc:
            error_msg = str(exc).lower()
            is_capacity_error = any(kw in error_msg for kw in _CAPACITY_KEYWORDS)

            if is_capacity_error and attempt < max_retries:
                wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(
                    "API capacity/rate error. Retrying in %.2fs "
                    "(attempt %d/%d): %s",
                    wait,
                    attempt,
                    max_retries,
                    exc,
                )
                time.sleep(wait)
            elif is_capacity_error:
                logger.error(
                    "Failed to propose %s after %d retries: %s",
                    child_type.value,
                    max_retries,
                    exc,
                )
                return None
            else:
                # Unexpected transient error — log and skip this branch
                logger.error(
                    "Unexpected error proposing %s for '%s': %s",
                    child_type.value,
                    parent_post.name,
                    exc,
                )
                return None

    return None


# ---------------------------------------------------------------------------
# 2 & 3. Layer-by-layer creative pipeline with asymmetric branching
# ---------------------------------------------------------------------------
# Each tuple: (PostType to generate, number of children per parent node)
WORKFLOW_PIPELINE: list[tuple[PostType, int]] = [
    (PostType.STAKEHOLDER, 1),   # Thin gateway — one stakeholder per mission
    (PostType.GOAL, 2),          # "Purpose" layer — 2 goals per stakeholder
    (PostType.ABSTRACTION, 2),   # 2 abstract principles per goal
    (PostType.ANALOGY, 4),       # 4 cross-domain analogies (widest layer)
    (PostType.INSPIRATION, 2),   # 2 concrete inspirations per analogy
    (PostType.SOLUTION, 3),      # 3 actionable solutions per inspiration
]


def run_creative_pipeline(
    mission_name: str,
    mission_desc: str,
) -> tuple[Post, list[Post]]:
    """Execute the full abstraction-and-analogy pipeline layer-by-layer.

    Returns:
        A tuple of (root mission Post, flat list of SOLUTION Posts).
    """
    model = experiment_gemini_model()
    engine = IdeaEngine(LLMClient(model_name=model))
    root: Post = engine.create_mission(mission_name, mission_desc)

    logger.info("Mission created: '%s'", mission_name)
    logger.info("Experiment LLM model: %s", model)
    logger.info(
        "Pipeline: %s",
        " -> ".join(pt.value.upper() for pt, _ in WORKFLOW_PIPELINE),
    )

    # Start with the root mission as the only node in the current layer
    current_nodes: list[Post] = [root]

    for layer_idx, (target_type, branching_factor) in enumerate(WORKFLOW_PIPELINE, 1):
        total_layers = len(WORKFLOW_PIPELINE)
        expected = len(current_nodes) * branching_factor
        logger.info(
            "--- Layer %d/%d: %s (branching=%d, parents=%d, expected=%d) ---",
            layer_idx,
            total_layers,
            target_type.value.upper(),
            branching_factor,
            len(current_nodes),
            expected,
        )

        next_layer: list[Post] = []

        for parent_idx, parent in enumerate(current_nodes, 1):
            for child_num in range(1, branching_factor + 1):
                logger.info(
                    "  [%d/%d] Parent '%s' -> %s child %d/%d",
                    parent_idx,
                    len(current_nodes),
                    parent.name,
                    target_type.value.upper(),
                    child_num,
                    branching_factor,
                )

                new_post = robust_propose_achiever(engine, target_type, parent)
                if new_post is not None:
                    next_layer.append(new_post)
                    logger.info("    -> Generated: '%s'", new_post.name)
                else:
                    logger.warning("    -> Skipped (error or retries exhausted)")
                delay = experiment_request_delay_seconds()
                if delay > 0:
                    time.sleep(delay)

        if not next_layer:
            logger.error(
                "Layer %d produced zero nodes — pipeline cannot continue.",
                layer_idx,
            )
            break

        current_nodes = next_layer

    # The final layer contains all SOLUTION nodes (unless the pipeline broke early)
    solutions = [p for p in current_nodes if p.ptype == PostType.SOLUTION]

    # ----- Export full tree to JSON -----
    output_path = "experiment_results.json"
    export_json(root, output_path)
    logger.info("Tree exported to %s", output_path)

    return root, solutions


# ---------------------------------------------------------------------------
# 4 & 5. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mission = "Next-Gen Grid-Scale Energy Storage"
    desc = (
        "Develop novel methods to store grid-scale renewable energy "
        "that do not rely on rare-earth lithium-ion batteries, targeting "
        "cost, scalability, and environmental sustainability."
    )

    tree, solutions = run_creative_pipeline(mission, desc)

    print(f"\nExperiment complete. Total solutions generated: {len(solutions)}")
    for i, sol in enumerate(solutions, 1):
        print(f"  {i}. {sol.name}")
