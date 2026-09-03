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

## Skills and agents

Five review skills live in `.claude/skills/` and two subagents in
`.claude/agents/`; `.claude/README.md` indexes them, gives each agent's default
target, and sets out which file owns which rule. Their names and descriptions
are injected at session start, so none is listed here.

**Run `code-reviewer commit` before committing non-trivial changes**, and the
full pass — `code-reviewer` with no argument — before opening a pull request.

## Review in two tiers

Both tiers are run by the `code-reviewer` agent, and the difference is what
each is defined over:

- **`commit`, per commit, over what is being committed.** The three
  report-and-fix skills, and nothing else. It does **not** run the suite —
  you run the tests the commit can reach.
- **The full pass, per branch, before its pull request opens.** Adds the
  suite over the whole change, the `test-review` mutation phase, and a
  failing test pinning any confirmed defect.

**Resolve or waive every Must Fix and Should Fix before opening the pull
request.** Never let a finding lapse by calling it "pre-existing" or "out of
scope" — surface it for an explicit decision.

`main` is protected and takes no direct pushes, so every change arrives
through a squash-merged pull request. The required `test` check runs
`uv run ruff check .`, `uv run ruff format --check .` and
`uv run pytest -m "not e2e"` — the last includes a full LP solve, which
dominates its runtime; the browser
tests run after the merge in `e2e.yml`.

**The gate runs eight of the repo's ten tests — the other two carry the `e2e`
marker — so a green check is weaker evidence here than "passing" suggests.** Say what you actually verified rather
than leaning on it.

## Prose is professional and factual

**Everything written here — comments, docstrings, READMEs, commit messages,
pull-request bodies, skills and agents — states what is true and how the
reader can check it.** A sentence that rates something without evidence
describes the author's opinion, not the code's behavior. When the code
changes, unsupported ratings do not update with it.

Common categories to avoid: unmeasured rankings, personified programs where
the verb stands in for a mechanism, unmeasured cost or effort claims,
aesthetic verdicts like "elegant" or "hacky", aphorisms, and filler run-ups.
See `comment-docstring` for rewrites, greps, and the categories that need
manual review.

**A claim about the model's output is a number or it is nothing.** Name the
objective, the inputs, and what moved. "The new constraint gives a better
portfolio" is the exact sentence this section exists to prevent — better on
which metric, by how much, against which run?

**Argument is not editorializing.** State each claim with its reason, in the
same sentence or the next one — for example, "two copies of the same value
drift apart over time." Give the reader something to check; keep the
reasoning and drop unsupported ratings.

Judge sentences in context — some individual words that look like offenders
are fine. See `comment-docstring` for details.

## Comments & docstrings are self-contained

**Every comment, docstring, and doc must stand on its own for a reader who
has the repo and nothing else**, and must describe the code as it is now.
References that only make sense outside the repo, or only to people involved
in the original conversation, break for future readers.

Common categories to avoid: references to commits, tickets, "as discussed",
earlier versions of the code, shortened domain terms that collapse to common
English words, and bare dates. See `comment-docstring` for examples and
greps.

Describe the thing directly — what it does, what the constraint is, why this
way rather than the obvious alternative. Test: **delete every ticket and
commit message; would this sentence still teach a new reader anything?**

**An LP constraint is where this matters most.** The expression says what is
forbidden; only a comment can say why that bound and not another one.

Point to durable references freely: a README section, another module, an
external spec. Ask whether the reference will still exist a year from now.

- **Pull-request and issue bodies, at a stricter bar.** Their reader has the
  diff and little else, so even a pointer into this repo fails when the diff
  omits the file it points at. Name the thing, not its number.
  `.github/PULL_REQUEST_TEMPLATE.md` carries this reminder at the point of
  writing.
- **Directory READMEs point, never restate.** Each says what belongs in its
  directory and links to whatever owns the detail. Duplicated descriptions go
  stale when the code moves.

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

**Pattern to preserve:** the CO2 emission factors and resource cost formulas
are computed once, in `parameters.cost_inputs(...)`, which both
`app.build_inputs` and `parameters.get_base_inputs` invoke with their own knob
values. Change a cost or emission formula there and nowhere else — two copies
carrying different asset lives give two different answers, which is how these
came to be consolidated.

**Common anti-patterns to refuse / fix on sight:**
- The app or a test hardcoding a value `src/parameters.py` already defines
  (a capacity bound, the gas restriction, a battery parameter). `src/LP.py`
  holds none of them; it merges caller overrides over `get_base_inputs()`.
- Two blocks in different files that are "supposed to" stay identical
  (the emission/cost math above). Collapse to one home.
- A function default that silently disagrees with the constant the
  production path uses.
- A plot or metric re-deriving a value the model already produced in
  `final_df` / `metrics` instead of reading it back.

## Repo conventions

- **`app.py` lives at the repo root** — Posit Connect Cloud expects the
  Shiny entrypoint there (app object `app`). It imports the model via the
  package path (`from src.LP import run_lp`, `from src import parameters`,
  `from src.db import RESULTS_ZIP, ResultsDB`, `from src.utils import
  get_resource_stack_plot`);
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
  - Full LP checks: `uv run pytest -m "slow and not e2e"`. Plain `-m slow`
    also collects the browser test, which carries both markers.
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
  `X | None` over `Optional[X]`, direct imports for type hints. A `_` prefix
  marks a Shiny `@reactive.effect` binding or another module-internal helper
  and nothing else — a function reached from outside its module never carries
  one.
