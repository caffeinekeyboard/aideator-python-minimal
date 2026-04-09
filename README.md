# Aideator

An LLM-powered idea tree builder that uses structured deliberation to explore problems and generate solutions. Aideator builds a tree of interconnected posts -- starting from a mission, branching through stakeholders, goals, barriers, and solutions -- with each node proposed by the Gemini API based on its ancestors' context.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Gemini API key:

```
GEMINI_API_KEY="your-api-key-here"
```

## Quick Start

```bash
python main.py
```

You'll be prompted to create a new mission or load an existing tree from a JSON file. Choose `n` for new, then provide a name and description for your deliberation.

## How It Works

Aideator builds an **idea tree** where each node is a "post" with a type, name, and description. The tree follows a structured deliberation pattern:

```
mission
  stakeholder
    goal
      barrier
        cause
          cause (recursive)
          solution
          abstraction -> analogy -> inspiration -> solution
        solution
        abstraction -> analogy -> inspiration -> solution
      solution
      abstraction -> analogy -> inspiration -> solution
    solution
      improvement
      barrier (feeds back into barrier analysis)
      question
        answer
```

When you add a child post, Aideator sends the full ancestor context to the LLM, which proposes a new node that fits the deliberation structure. This ensures each idea is grounded in the problem context.

## CLI Commands

| Command | Description |
|---------|-------------|
| `tree` | Display the full idea tree with numbered posts |
| `select <N>` | Select post number N (numbers shown in tree display) |
| `add <TYPE>` | Ask the LLM to propose a new child of TYPE for the selected post |
| `actions` | Show what child types can be added to the selected post |
| `info` | Show full details of the selected post |
| `context` | Show the ancestor chain from root to the selected post |
| `save <FILE>` | Export the tree to a JSON file |
| `load <FILE>` | Import a tree from a JSON file |
| `help` | Show the help message |
| `quit` | Exit |

## Post Types

| Type | Purpose | Parent Types |
|------|---------|-------------|
| **mission** | Root node: defines the deliberation scope | -- |
| **stakeholder** | A class of entities whose needs matter | mission |
| **goal** | A desired outcome for a stakeholder | stakeholder |
| **barrier** | Something that undercuts a goal or solution | goal, barrier, cause, solution |
| **cause** | A root cause behind a barrier | barrier, cause |
| **solution** | A proposed fix for a goal, barrier, or cause | goal, barrier, cause, inspiration, question |
| **abstraction** | A generalization (for analogical reasoning) | goal, barrier, cause |
| **analogy** | A structurally similar problem from another domain | abstraction |
| **inspiration** | An idea drawn from an analogy | analogy |
| **improvement** | An enhancement to an existing solution | solution |
| **question** | A question that would make a solution more robust | solution |
| **answer** | A response to a question | question |

## Example Session

```
=== Aideator - Idea Tree Builder ===

(n)ew mission or (l)oad from file? n
Mission name: Urban Transportation Reform
Mission description:  can our city reduce traffic congestion while improving public transit access for all residents?How

Mission created: "Urban Transportation Reform"

1. [mission] Urban Transportation Reform
Selected: [mission] Urban Transportation Reform
Allowed children: stakeholder

> add stakeholder
Proposing stakeholder for [mission] Urban Transportation Reform...

Created [stakeholder] "Daily Commuters"
  Workers who rely on transportation infrastructure to get to and from their jobs...

> select 2
Selected: [stakeholder] Daily Commuters

> add goal
Proposing goal for [stakeholder] Daily Commuters...

Created [goal] "Reduce Average Commute Time"
  ...

> save my_tree.json
Tree saved to my_tree.json
```

## Project Structure

```
aideator/
  models.py          Post data model and type definitions
  tree.py            Tree traversal operations
  transitions.py     Rules for which child types are allowed
  prompts.py         LLM prompt templates for each post type
  llm.py             Gemini API client and response parsing
  engine.py          Core orchestrator
  serialization.py   JSON import/export and tree display
  cli.py             Interactive command-line interface
main.py              Entry point
experiment_runner.py Automated large-scale ideation experiment
```

---

## Automated Experiment Runner

`experiment_runner.py` is a standalone script for running automated, large-scale ideation experiments. Instead of manually adding nodes one-by-one through the CLI, the experiment runner programmatically expands an entire idea tree using the framework's **abstraction-and-analogy** workflow — generating dozens of diverse solutions in a single unattended run.

### Why Not Just Use the CLI?

The interactive CLI (`main.py`) is designed for human-guided exploration: you pick a node, choose a child type, inspect the result, and decide where to go next. This is great for understanding a problem space, but it doesn't scale when your goal is to **maximize creative diversity**. The experiment runner automates the entire expansion process with a forced pipeline, asymmetric branching, and built-in fault tolerance so you can kick off a run and walk away.

### How It Works

#### The Forced Creative Pipeline

The runner does **not** use recursive depth-first search, which can get stuck exploring a single branch deeply before visiting others. Instead, it expands the tree **layer-by-layer** (breadth-first), processing every node at one level before moving to the next. This guarantees that the tree grows evenly and all branches receive equal creative attention.

The pipeline follows this strict sequence of post types, which is the valid transition path defined in `aideator/transitions.py`:

```
MISSION -> STAKEHOLDER -> GOAL -> ABSTRACTION -> ANALOGY -> INSPIRATION -> SOLUTION
```

Each layer produces a set of child nodes, and that entire set becomes the input for the next layer. The STAKEHOLDER layer acts as a thin gateway (one stakeholder per mission), while GOAL serves as the effective "purpose" layer where the problem space first branches out.

#### Asymmetric Branching

Not every layer of the tree should be equally wide. The creative, lateral-thinking stages (especially ANALOGY) benefit from generating more variations, while the structural stages can stay narrow. The runner uses the following branching factors:

| Layer | Post Type | Children per Parent | Purpose |
|-------|-----------|---------------------|---------|
| 1 | STAKEHOLDER | 1 | Thin gateway to enter the goal layer |
| 2 | GOAL | 2 | Define 2 distinct goals (the "purpose" layer) |
| 3 | ABSTRACTION | 2 | Generalize each goal into 2 abstract principles |
| 4 | ANALOGY | 4 | **Widest layer** — 4 cross-domain analogies per abstraction |
| 5 | INSPIRATION | 2 | Extract 2 concrete inspirations from each analogy |
| 6 | SOLUTION | 3 | Derive 3 actionable solutions from each inspiration |

This yields a theoretical maximum of **1 x 2 x 2 x 4 x 2 x 3 = 96 solutions** per run. The actual count may be lower if some branches are skipped due to API errors (see below).

#### The Tree Shape

Here is what the generated tree looks like at full expansion:

```
[MISSION] Next-Gen Grid-Scale Energy Storage
  [STAKEHOLDER] (1 node)
    [GOAL] (2 nodes)
      [ABSTRACTION] (4 nodes total, 2 per goal)
        [ANALOGY] (16 nodes total, 4 per abstraction)
          [INSPIRATION] (32 nodes total, 2 per analogy)
            [SOLUTION] (96 nodes total, 3 per inspiration)
```

### Resilience: The `robust_propose_achiever` Wrapper

Each node in the tree requires an LLM API call (to Gemini). At scale (96+ calls per run), you **will** hit rate limits, capacity errors, and occasional transient failures. The `robust_propose_achiever` function wraps every `engine.propose_achiever()` call with production-grade error handling:

**Exponential Backoff with Jitter:**
When a retryable error is detected, the function waits before retrying. The wait time doubles with each attempt (2s, 4s, 8s, 16s, 32s, 64s) and adds a random jitter between 0-1 seconds to prevent the "thundering herd" problem (where multiple retries all fire at the same instant).

**Retryable Error Detection:**
The wrapper inspects exception messages for keywords that indicate API capacity/rate issues: `429`, `503`, `capacity`, `quota`, `rate`, `overloaded`. These errors are transient and worth retrying.

**Graceful Branch Skipping:**
If a non-retryable error occurs (e.g., a prompt validation failure or an unexpected API response), the function logs the error and returns `None` instead of crashing the entire experiment. The pipeline skips that branch and continues with the remaining nodes. This means a single bad response doesn't waste the dozens of successful calls that came before it.

**Max Retries:**
Each call is attempted up to 6 times (configurable via `max_retries`). After exhausting all retries on a capacity error, the branch is skipped rather than raising an exception.

### Console Output

The runner uses Python's `logging` module with timestamped, leveled output so you can monitor progress in real time:

```
14:23:01 | INFO    | Mission created: 'Next-Gen Grid-Scale Energy Storage'
14:23:01 | INFO    | Pipeline: STAKEHOLDER -> GOAL -> ABSTRACTION -> ANALOGY -> INSPIRATION -> SOLUTION
14:23:01 | INFO    | --- Layer 1/6: STAKEHOLDER (branching=1, parents=1, expected=1) ---
14:23:01 | INFO    |   [1/1] Parent 'Next-Gen Grid-Scale Energy Storage' -> STAKEHOLDER child 1/1
14:23:03 | INFO    |     -> Generated: 'Utility Grid Operators'
14:23:03 | INFO    | --- Layer 2/6: GOAL (branching=2, parents=1, expected=2) ---
14:23:03 | INFO    |   [1/1] Parent 'Utility Grid Operators' -> GOAL child 1/2
14:23:05 | INFO    |     -> Generated: 'Reduce Peak Load Dependency'
...
14:25:17 | WARNING | API capacity/rate error. Retrying in 2.73s (attempt 1/6): 429 Too Many Requests
...
14:45:22 | INFO    | Tree exported to experiment_results.json

Experiment complete. Total solutions generated: 91
  1. Compressed Air Energy Storage in Abandoned Mines
  2. Gravity-Based Rail Storage on Mountain Slopes
  ...
```

Each log line shows the current layer, which parent is being expanded, and which child number is being generated — making it easy to estimate remaining time and identify where failures occur.

### Running the Experiment

```bash
python experiment_runner.py
```

Make sure your `.env` file contains a valid `GEMINI_API_KEY` before running. The script will:

1. Instantiate the `IdeaEngine` (which loads the Gemini API client)
2. Create the root mission node
3. Expand all 6 layers of the pipeline, making LLM calls with automatic retry
4. Export the full tree to `experiment_results.json`
5. Print a summary of all generated solutions to the console

The default mission is **"Next-Gen Grid-Scale Energy Storage"** — developing novel methods to store grid-scale renewable energy without relying on rare-earth lithium-ion batteries. You can change this by editing the `mission` and `desc` variables in the `if __name__ == "__main__"` block.

### Output File

The experiment produces `experiment_results.json`, a nested JSON file representing the entire idea tree. You can reload this tree later using `aideator.serialization.import_json`:

```python
from aideator.serialization import import_json, print_tree

root = import_json("experiment_results.json")
print(print_tree(root))
```

Or load it in the interactive CLI:

```bash
python main.py
# Choose (l)oad, then enter: experiment_results.json
```

### Customization

**Changing the mission:** Edit the `mission` and `desc` strings in the `__main__` block.

**Adjusting branching factors:** Modify the `WORKFLOW_PIPELINE` list at the module level. For example, to generate more analogies:

```python
WORKFLOW_PIPELINE = [
    (PostType.STAKEHOLDER, 1),
    (PostType.GOAL, 3),          # 3 goals instead of 2
    (PostType.ABSTRACTION, 2),
    (PostType.ANALOGY, 6),       # 6 analogies instead of 4
    (PostType.INSPIRATION, 2),
    (PostType.SOLUTION, 3),
]
```

**Adjusting retry behavior:** Pass a different `max_retries` value to `robust_propose_achiever`, or modify the `base_wait` variable inside the function to change the initial backoff duration.

**Using the pipeline programmatically:** Import and call `run_creative_pipeline` from your own scripts:

```python
from experiment_runner import run_creative_pipeline

root, solutions = run_creative_pipeline(
    "Urban Traffic Reduction",
    "Reduce urban traffic congestion by 40% without building new roads."
)

for sol in solutions:
    print(f"{sol.name}: {sol.description}")
```
