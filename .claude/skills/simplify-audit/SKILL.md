---
name: simplify-audit
description: Repo-wide bloat audit. Finds code that should not exist or is not in its minimal form — dead code, unused deps, single-use abstractions, premature generalization — and reports a delete-list. Report-only; makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: [file-or-directory]
---

# Simplify Audit

Find excess and report a **delete-list** — code that should not exist, or
that exists but is not in its minimal form. Report issues but do NOT make
edits. Removal is a separate, user-driven step.

This skill's lens is **minimalism**, which is different from the
`code-quality` skill's lens (readability + documentation):

| | `code-quality` | `simplify-audit` (this skill) |
|---|---|---|
| Asks | "Is this clear and documented?" | "Should this exist, and is it minimal?" |
| Default scope | changed files (diff) | whole repo |
| Output | readability findings | a delete-list (LOC removable) |

For **duplication** specifically, defer to `code-quality` items 17–20 (and
its **Single Source of Truth for Parameter Values** section, items 21–25,
a high-priority duplication class) — do not restate those checklists here.
Point the user at them when you spot repeated code or duplicated
config-value sources.

## What counts as in scope

- **In scope:** live `app.py`, `src/`, and `tests/` *only* for unused
  test-helper bloat.
- **Out of scope:** any *gitignored* path (`.venv/`, `csv/`,
  `results.zip`, `__pycache__/`, `.pytest_cache/`) and generated deploy
  artifacts (`requirements.txt`, `uv.lock`, `manifest.json`). Run
  `git ls-files` if unsure whether a path is tracked.

## Audit Checklist (the minimalism lens)

1. **Existence / YAGNI** — functions, classes, branches, or config fields
   never referenced in live code. **Grep-confirm zero references outside
   the definition** before reporting (see Steps). Deleting beats
   refactoring beats adding.
2. **Reinvention** — hand-rolled logic that duplicates a built-in from the
   stdlib or an already-imported library (`pandas`, `ortools`, `plotly`,
   `shiny`, `shinywidgets`). Cite the built-in that replaces it (e.g. a
   manual row loop that a vectorized pandas op does in one line).
3. **Dependency justification** — cross-check each runtime dependency in
   `pyproject.toml` (`[project].dependencies`) against live `import`
   usage. Flag any dep with zero or near-zero live imports as a candidate
   for removal. The import name may differ from the package name, and a
   package pulled in only *transitively* (e.g. a numeric lib used by
   `pandas`/`ortools` but never imported directly) does not belong in the
   direct dependency list. Note that `requirements.txt` is a generated uv
   export mirroring deploy needs (Posit Connect Cloud) — if it drifts from
   `pyproject.toml`, flag that it needs regenerating rather than
   hand-editing.
4. **Single-use abstraction** — wrapper functions, one-method classes, or
   indirection layers used exactly once. Recommend inlining at the single
   call site.
5. **Premature generalization** — parameters, branches, config knobs, or
   "flexibility" that handle cases which never occur in practice (e.g. a
   `use_outside_energy=False` path or a `restrict_gas is None` branch in
   `LP.py` that the app never triggers — verify against the call sites
   before flagging).
6. **Repo-wide dead exports** — public symbols (functions, classes,
   constants) with no references anywhere in live code. This is broader
   than `code-quality`, which only sees the diff. Remember the app imports
   from `src/` (`from src.LP import run_lp`, `from src import db, utils`)
   and the tests import from `src/` too — grep `app.py` and `tests/`
   before declaring a `src/` symbol dead. Watch the `__main__` blocks in
   `src/LP.py` / `src/db.py`: helpers used only there (e.g.
   `make_fake_results`) are also imported by tests — confirm before
   listing.
7. **Size signals** — files > ~800 lines or functions > ~80 lines,
   reported as a bloat smell. Cite them; do NOT prescribe the split here
   (defer the "how" to `code-quality`'s function-length guidance). `run_lp`
   in `src/LP.py` is the main one — note it, but it's inherently a long
   build→solve→assemble procedure.
8. **Dead scaffolding** — commented-out code blocks and stale TODO stubs
   that were never finished.

## Output Format

Start with a one-block **summary**:

- Total live LOC (by area: `app.py`, `src/`, `tests/`).
- Estimated removable LOC.
- A **top 10 cleanups by LOC removed** table: `rank | file:line | action |
  est. LOC | one-line rationale`.

Then the full **delete-list**, grouped by action:

### Delete
Confirmed dead — grep-proven zero references. Each: `file:line`, est. LOC,
one-line rationale.

### Simplify
Exists but over-built — inline the single-use wrapper, use the library
built-in, drop the unused knob, collapse a duplicated config source. Each:
`file:line`, est. LOC, what to do.

### Verify
Looks removable but needs a human check before deleting (e.g., referenced
only via dynamic dispatch, a public entry point, a test, or an external
caller). Each: `file:line`, what to verify.

## Steps

1. **Determine scope.** If a path argument is given, audit that path.
   Otherwise audit the whole repo. Never audit gitignored/untracked files
   (run `git ls-files` if unsure whether a path is tracked).
2. **Mechanical passes first** — reuse existing tooling, do not reinvent it:
   - `uv run ruff check --select F401,F811,SIM .` — unused imports (F401),
     redefinitions (F811), and simplifiable code (SIM). Cite the rule code.
   - **Dependency cross-check:** for each runtime dep in `pyproject.toml`
     (`[project].dependencies`), grep its import name across live code. A
     dep with no live `import` is a removal candidate (see `numpy` above).
3. **Grep-confirm dead symbols.** For every candidate, grep the symbol name
   across the repo — including `app.py` and `tests/` — and confirm it has
   **no references outside its own definition** before listing it under
   **Delete**. If there is any ambiguity (dynamic dispatch, entry point,
   re-export, test use), downgrade it to **Verify**.
4. **Read the suspicious files** to confirm context before listing — do not
   report from grep counts alone.
5. **Emit the report** (summary + delete-list). Make **no edits** — this
   skill is report-only.
