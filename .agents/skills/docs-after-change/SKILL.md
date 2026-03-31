---
name: docs-after-change
description: Mandatory after code changes to keep docstrings, README, and architecture docs aligned with the new behavior. Use after adding, editing, deleting, or renaming code, scripts, config, outputs, or workflow behavior.
---

# docs-after-change

This skill is mandatory after any code change, including refactors. It is optimized to keep repository documentation accurate without doing an unnecessary full-repo audit.

## Goal

Keep documentation and developer UX up to date by validating and updating:

- Docstrings in changed runnable scripts and touched public modules where docstrings clarify behavior.
- README files that document entrypoints, setup, env vars, and outputs.
- Architecture docs when flow or structure changes.

## Inputs to consider

- Python files (`.py`)
- Configuration files (`.yml`, `.yaml`, `.toml`, `.json`, `.env.example`) that affect runtime behavior
- Docker, Kubernetes, or deployment manifests
- Migrations, schema changes, or database models
- Anything that changes CLI flags, environment variables, output folders, or pipeline stages

## Instructions

### 1. Identify what changed and what it impacts

Determine whether the change affects:

- CLI entrypoints
- Configuration
- Outputs
- Architecture
- Tests and workflows

### 2. Update docstrings for runnable scripts and public entrypoints

For any file intended to be executed as a script or documented as an entrypoint:

- Ensure a top-level module docstring exists and is accurate.
- Cover what the script does, its inputs, its outputs, and a `python -m ...` example from the project root.
- Ensure side effects do not run at import time.
- Ensure the documented CLI arguments match actual `argparse` behavior.

For touched public modules, add or update docstrings when they clarify non-obvious behavior or invariants.

Additionally, ensure every function and method you touched has a docstring:

- Trivial functions and methods: a one-line docstring is enough.
- Non-trivial or side-effecting functions and methods: describe inputs, outputs, side effects, and raised exceptions when they are not obvious.

### 3. Update README files when developer instructions changed

Update `README.md` if any of these changed:

- Installation steps
- Required environment variables or keys
- Example commands and flags
- Output directories, filenames, or structure
- Typical workflows
- Model or provider configuration examples

Checklist:

- Keep examples runnable.
- Do not add secrets.
- If you introduce a new required env var, ensure it appears in `.env.example` and is referenced in `README.md`.

### 4. Update architecture docs when structure or flow changed

Update `architecture.md` and related architecture docs if you changed:

- Module responsibilities or boundaries
- Data flow between stages
- Components or services that exist or no longer exist
- Persistence or output formats

Rules of thumb:

- If diagrams mention modules that no longer exist, fix them.
- If a new stage is added, document what it reads and writes.

### 5. Run a consistency pass

- Ensure terminology matches across docs.
- Ensure paths are consistent relative to project root.
- Ensure references to the source of truth are consistent.

### 6. Report what changed

In the final response, include:

- Which docs you reviewed
- Which docs you changed and why
- Which docs you intentionally did not change, and why

## Non-goals

- Do not run a full-repo doc audit here.
- Do not rewrite docs for style alone; focus on correctness and minimal updates.
