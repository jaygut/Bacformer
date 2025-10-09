# Repository Guidelines

## Project Structure & Modules
- `bacformer/`: Python package (modeling, preprocessing, TL). Entry modules live under `modeling/`, `pp/`, and `tl/`.
- `tests/`: Pytest suite organized by area (`modeling/`, `tl/`, `pp/`).
- `tutorials/`: Jupyter notebooks demonstrating core tasks.
- `scripts/`: Small utilities used during development.
- `files/`: Sample assets (e.g., `pao1.gbff`, images).
- `ai_docs/`, `.claude/`: Design and agent notes (keep synced with code).

## Build, Test, and Dev Commands
- Install (dev + tests): `pip install .[dev,test]` (Python >= 3.10).
- Lint/format: `ruff check .` and `ruff format .` (or `pre-commit run -a`).
- Tests: `pytest -q` (or with coverage: `pytest --cov=bacformer`).
- Build wheel: `hatch build` (uses Hatchling per `pyproject.toml`).
- Pre-commit: `pre-commit install` then commit; CI expectations mirror hooks.

## Coding Style & Naming
- Formatter/linter: Ruff (120 char lines, import sorting, numpy-style docstrings). No nb formatting in CI.
- Indentation: 4 spaces; UTF-8; Unix line endings.
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Public APIs live in `bacformer/__init__.py` and subpackage `__init__.py`; keep imports explicit.

## Testing Guidelines
- Framework: Pytest; tests under `tests/`; name files `test_*.py` and functions `test_*`.
- Coverage: target meaningful lines in `bacformer/`; avoid testing tutorials and data assets (see coverage omit in `pyproject.toml`).
- Use small, deterministic fixtures; prefer CPU paths when possible.

## Commit & Pull Request Guidelines
- Style: Conventional prefixes (feat, fix, refactor, docs, test, chore); keep subject imperative and concise.
- Commits should be focused; include brief rationale when non-obvious.
- PRs: clear description, linked issues (`Fixes #123`), test updates, before/after notes for behavior, and any performance/memory impact.
- Run `pre-commit run -a` and `pytest` locally before opening the PR.

## Security & Configuration Tips
- GPU is optional; CPU tests should pass. Flash-Attn and FAESM are optional extras.
- Pin heavy deps only when required; prefer feature flags/extras (`.[faesm]`).
