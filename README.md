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
Mission description: How can our city reduce traffic congestion while improving public transit access for all residents?

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
```
