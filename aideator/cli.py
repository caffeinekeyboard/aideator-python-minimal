from __future__ import annotations

import sys

from aideator.models import PostType
from aideator.engine import IdeaEngine
from aideator.tree import describe_context
from aideator.transitions import get_allowed_children, ACTION_NAMES
from aideator.serialization import export_json, import_json, print_tree


HELP_TEXT = """
Commands:
  tree              Display the full idea tree
  select <N>        Select post number N (from tree display)
  info              Show details of the selected post
  context           Show the context chain for the selected post
  add <TYPE>        Propose a new child of TYPE for the selected post
  actions           Show allowed child types for the selected post
  save <FILE>       Export tree to a JSON file
  load <FILE>       Import tree from a JSON file
  help              Show this help message
  quit              Exit the program
"""


def _read_input(prompt: str) -> str:
    """Read a line of input, handling EOF."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)


def main() -> None:
    engine: IdeaEngine | None = None
    root = None
    selected = None
    post_index: dict[int, object] = {}

    print("=== Aideator - Idea Tree Builder ===\n")

    # Startup: create mission or load existing tree
    while root is None:
        choice = _read_input("(n)ew mission or (l)oad from file? ")
        if choice.lower().startswith("l"):
            filepath = _read_input("File path: ")
            try:
                root = import_json(filepath)
                selected = root
                engine = IdeaEngine()
                print(f"\nLoaded tree from {filepath}")
            except Exception as e:
                print(f"Error loading file: {e}")
        elif choice.lower().startswith("n"):
            name = _read_input("Mission name: ")
            description = _read_input("Mission description: ")
            if not name or not description:
                print("Name and description are required.")
                continue
            try:
                engine = IdeaEngine()
            except Exception as e:
                print(f"Error initializing LLM client: {e}")
                sys.exit(1)
            root = engine.create_mission(name, description)
            selected = root
            print(f'\nMission created: "{name}"')
        else:
            print("Please enter 'n' or 'l'.")

    assert engine is not None

    # Main interaction loop
    while True:
        print()
        post_index.clear()
        tree_str = print_tree(root, index=post_index)
        print(tree_str)

        if selected:
            allowed = get_allowed_children(selected.ptype)
            actions_str = ", ".join(t.value for t in allowed) if allowed else "none"
            print(f"Selected: [{selected.ptype.value}] {selected.name}")
            print(f"Allowed children: {actions_str}")

        cmd = _read_input("\n> ")
        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "quit" or command == "exit":
            print("Goodbye.")
            break

        elif command == "help":
            print(HELP_TEXT)

        elif command == "tree":
            # Tree is already printed at top of loop
            pass

        elif command == "select":
            if not arg.isdigit():
                print("Usage: select <N>")
                continue
            num = int(arg)
            if num in post_index:
                selected = post_index[num]
                print(f"Selected: [{selected.ptype.value}] {selected.name}")
            else:
                print(f"No post with number {num}. Valid: 1-{len(post_index)}")

        elif command == "info":
            if selected is None:
                print("No post selected.")
                continue
            print(f"  Type:        {selected.ptype.value}")
            print(f"  Name:        {selected.name}")
            print(f"  Description: {selected.description}")
            if selected.purpose:
                print(f"  Parent:      [{selected.purpose.ptype.value}] {selected.purpose.name}")
            print(f"  Children:    {len(selected.achievers)}")

        elif command == "context":
            if selected is None:
                print("No post selected.")
                continue
            print(describe_context(selected))

        elif command == "actions":
            if selected is None:
                print("No post selected.")
                continue
            allowed = get_allowed_children(selected.ptype)
            if not allowed:
                print(f"No actions available for {selected.ptype.value}.")
            else:
                print(f"Available actions for [{selected.ptype.value}] {selected.name}:")
                for child_type in allowed:
                    action_name = ACTION_NAMES.get((selected.ptype, child_type), "add")
                    print(f"  {action_name:20s} -> add {child_type.value}")

        elif command == "add":
            if selected is None:
                print("No post selected.")
                continue
            if not arg:
                print("Usage: add <TYPE>")
                allowed = get_allowed_children(selected.ptype)
                if allowed:
                    print(f"Available types: {', '.join(t.value for t in allowed)}")
                continue

            try:
                ptype = PostType(arg.lower())
            except ValueError:
                print(f"Unknown type: {arg}")
                valid = ", ".join(t.value for t in PostType)
                print(f"Valid types: {valid}")
                continue

            print(f"Proposing {ptype.value} for [{selected.ptype.value}] {selected.name}...")
            try:
                new_post = engine.propose_achiever(ptype, selected)
                print(f'\nCreated [{new_post.ptype.value}] "{new_post.name}"')
                print(f"  {new_post.description}")
                selected = new_post
            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"LLM error: {e}")

        elif command == "save":
            if not arg:
                print("Usage: save <FILE>")
                continue
            try:
                export_json(root, arg)
                print(f"Tree saved to {arg}")
            except Exception as e:
                print(f"Error saving: {e}")

        elif command == "load":
            if not arg:
                print("Usage: load <FILE>")
                continue
            try:
                root = import_json(arg)
                selected = root
                print(f"Tree loaded from {arg}")
            except Exception as e:
                print(f"Error loading: {e}")

        else:
            print(f"Unknown command: {command}. Type 'help' for available commands.")
