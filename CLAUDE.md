# Electricity Generation Planning — Project Instructions

A [Shiny for Python](https://shiny.posit.co/py/) app that sizes a
least-cost (or least-carbon) electricity generation portfolio — wind,
solar, battery, and gas — for four northern Colorado communities by
solving an 8,760-hour linear program with OR-Tools GLOP. `app.py` (repo
root) is the Shiny UI; the model and its helpers live in `src/`
(`LP.py` the linear program, `parameters.py` the default inputs and shared
cost/emission formulas, `db.py` an in-memory SQLite results store,
`utils.py` the Plotly chart), with the hourly load/generation profiles in
`src/profiles.csv`. The project is packaged with `uv` on Python 3.11,
containerized under `docker/`, and deployed to **Posit Connect Cloud**
(`app.py` at the root, `requirements.txt` + `manifest.json`).

## Skills (Slash Commands)

Custom skills automate the review workflows:

| Command | What it does |
|---------|-------------|
| `/code-quality [file-or-dir]` | Review code for readability, documentation, onboarding, and minimal form (simplification per the Minimalism rules) — report-only. (Named to avoid colliding with Claude Code's built-in `/code-review`.) |
| `/comment-docstring <file-or-dir>` | Review and fix docstrings, type hints, inline comments; sweep the README for stale prose (edits in place) |
| `/security-scan [file-or-dir]` | Scan for leaked secrets (hardcoded tokens/keys, tracked `.env`/credential files) and unsafe patterns (report-only) |
| `/simplify-audit [file-or-dir]` | Repo-wide bloat audit — reports a delete-list of dead code, unused deps, and over-built abstractions (report-only) |

Skills are defined in `.claude/skills/` and committed to the repo.

## Agents

Custom subagents bundle a workflow into a single delegated pass — to
combine multiple steps, or to run a heavy read-only pass in an isolated
context so the main session stays clean:

| Agent | What it does |
|-------|-------------|
| `code-reviewer` | Pre-commit pass: runs the `code-quality` and `security-scan` skills (report-only) then the `comment-docstring` skill (edits in place) over the changed files or a given path |
| `simplify-auditor` | Runs the `simplify-audit` skill in an isolated context and returns a report-only bloat delete-list; keeps the repo-wide grep/read churn out of the main session |

**Run the `code-reviewer` agent before committing non-trivial changes.**
Invoke agents by name, optionally with a file or directory. With no
argument, `code-reviewer` defaults to the changed files
(`git diff --name-only HEAD` plus untracked) and `simplify-auditor`
defaults to the whole repo:

```
> use the code-reviewer agent
> code-reviewer src/LP.py
> code-reviewer            # defaults to all changed files
> use the simplify-auditor agent
> simplify-auditor         # whole-repo bloat audit, isolated context
```

Each agent reads its skill's `SKILL.md` at runtime rather than copying
the checklist, so it stays in sync as the skills evolve. Agents are
defined in `.claude/agents/` and committed to the repo. New agent
files are discovered at CLI start, so restart the session after adding
one.

## Minimalism (write less)

An ordered checklist to run *before* writing code. Walk it top to
bottom and stop at the first step that solves the problem:

1. **Does this need to exist?** Prefer not writing it (YAGNI). Deleting
   beats refactoring beats adding.
2. **Use the standard library** before hand-rolling.
3. **Use a library already imported** (`pandas`, `ortools`, `plotly`,
   `shiny`, `shinywidgets`) — reach for its built-in before writing your
   own.
4. **Use an already-installed dependency** before adding a new one.
5. **Prefer the smallest correct form.** No helper, class, config knob, or
   generalization until there are 2–3 real call sites — no premature
   abstraction.
6. **Only then** write minimal custom code.

To find existing bloat, run `/simplify-audit` (report-only delete-list).

## Single source of truth for parameter values

Treat parameter values and constants as having exactly one home.
Duplicate-source-of-truth is a class of bug that drifts silently and is
expensive to debug.

**The rule:** every parameter value (a number, a dict, a config entry)
gets ONE home. Other modules read from that home; they don't redeclare
it, copy it, or compile it into a parallel mirror.

**Where parameter values live in this repo:**
- **All model parameter values live in `src/parameters.py`.** The headless
  default input set (capacity bounds, battery parameters, `restrict_gas`,
  outside-energy settings) is `parameters.get_base_inputs()`; `run_lp` calls
  it and merges caller overrides on top via `inputs.update(...)`. The
  emission factors and carbon-inclusive cost formulas are
  `parameters.cost_inputs(...)`, shared by both `get_base_inputs` and the
  Shiny UI (`app.build_inputs`) so the two paths cannot drift. `src/LP.py`
  is just the model (build → solve → assemble); it holds no default values.
- **Results schema** — the table/metric names shared between the model,
  the store, and the UI → `ResultsDB.table_names` in `src/db.py`
  (`['inputs', 'cap_mw', 'metrics', 'final_df']`). The app reads result
  keys by these names; don't hardcode a parallel list.
- **The profile data path** (`'src/profiles.csv'`) → read in
  `src/LP.run_lp`; don't paste the literal path elsewhere.
- **Deploy dependency set** → `pyproject.toml` `[project].dependencies` is
  canonical; `requirements.txt` and `manifest.json` are *generated
  artifacts* (see Repo conventions) that must be regenerated from it, not
  hand-edited.

**Pattern to preserve (was the marquee drift bug):** the CO2 emission
factors and resource cost formulas used to be computed **twice** — once in
the UI path and once in the model defaults — and had drifted (30-year vs
20-year asset life). They are now consolidated into
`parameters.cost_inputs(...)`, which both `app.build_inputs` and
`parameters.get_base_inputs` invoke with their own knob values. Keep it that way: if you touch a cost/emission formula, change
it in `parameters.cost_inputs` only. Do not re-inline the math into the app
or the model.

**Common anti-patterns to refuse / fix on sight:**
- The app or a test hardcoding a value `src/LP.py` already defines
  (a capacity bound, the gas restriction, a battery parameter).
- Two blocks in different files that are "supposed to" stay identical
  (the emission/cost math above). Collapse to one home.
- A function default that silently disagrees with the constant the
  production path uses.
- A plot or metric re-deriving a value the model already produced in
  `final_df` / `metrics` instead of reading it back.

## Repo conventions

- **`app.py` lives at the repo root** — Posit Connect Cloud expects the
  Shiny entrypoint there (app object `app`). It imports the model via the
  package path (`from src.LP import run_lp`, `from src import db, utils`);
  `run_lp` reads `src/profiles.csv` relative to the working directory
  (repo root), so keep the cwd at the root when running or testing.
- **The model is UI-agnostic and headless-testable.** `src/LP.py` has no
  Shiny imports — `run_lp` is a pure function returning a results dict.
  The app runs it off the event loop via `@reactive.extended_task`
  (`asyncio.to_thread`) so the ~15–60s solve doesn't block the UI. Keep
  that separation: put model/compute logic in `src/`, never in `app.py`.
- **Tooling:** `uv` for envs/commands, `ruff` for lint/format (config in
  `pyproject.toml`). `pyproject.toml` + `uv.lock` are the dependency
  source of truth; `requirements.txt` and `manifest.json` exist for the
  Posit Connect Cloud deploy — **regenerate them when runtime deps or app
  files change**, don't hand-edit:
  ```bash
  uv run python scripts/update_manifest.py
  ```
  That script is the single home for this: it re-exports `requirements.txt`
  and writes a trimmed `manifest.json`. rsconnect bundles every tracked
  file and ignores `.gitignore`, so the script's `EXCLUDES` list keeps dev
  tooling, docs, and the reference PDFs out of the bundle — leaving only
  `app.py`, `src/`, and `requirements.txt` plus a couple of small root
  files. Change what ships by editing `EXCLUDES`, not the docs.
- **Python 3.11 is pinned** (`.python-version`, `requires-python`) because
  that is the Python runtime Posit Connect Cloud provisions for git-based
  deploys (it did not honor a 3.12 request). Don't change it without
  confirming what Connect Cloud actually runs — a mismatch fails the deploy
  when a pinned dep (e.g. `numpy`) needs a newer Python than the runtime.
- **Tests:** `tests/test_lp.py` (fast unit tests + a slower full-solve
  integration set marked `slow`) and `tests/test_app.py` (Playwright
  end-to-end driving the Shiny app, marked `e2e`).
  - Fast gate: `uv run pytest -m "not slow and not e2e"`.
  - Full LP checks: `uv run pytest -m slow`.
  - Browser e2e: `uv run pytest -m e2e` (needs
    `uv run playwright install chromium` once).
  - **Run the e2e suite whenever `app.py` or `src/utils.py` (the plot)
    changes** — the unit suite doesn't exercise the rendered app.
- **Deploy:** Posit Connect Cloud builds from the GitHub repo using
  `app.py` + `requirements.txt` + `manifest.json`; Docker via
  `docker/Dockerfile` (uv-based, serves on 8501). Saved runs live in a
  per-session in-memory SQLite DB and the download zip is written to
  ephemeral working-directory storage, so runs don't persist across
  sessions — the same behavior locally, in Docker, and on Connect Cloud.
- **Commit messages:** describe the change only — do **not** add a
  `Co-Authored-By: Claude` trailer or a "Generated with Claude Code" line.
- **Style:** Google-style docstrings (summary, Args, Returns),
  `X | None` over `Optional[X]`, direct imports for type hints, no `_`
  prefix on function names except internal helpers.
