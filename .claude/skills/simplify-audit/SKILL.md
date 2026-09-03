---
name: simplify-audit
description: Repo-wide bloat audit. Finds code that should not exist or is not in its minimal form — dead code, unused deps, single-use abstractions, premature generalization — and reports a delete-list. Report-only; makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: "[file-or-directory]"
---
# Simplify Audit

Find excess and report a **delete-list** — code that should not exist, or
that exists but is not in its minimal form. Report issues but do NOT make
edits. Removal is a separate, user-driven step.

This skill checks **minimalism**, while `code-quality-review` checks
readability and documentation:

| | `code-quality-review` | `simplify-audit` (this skill) |
|---|---|---|
| Asks | "Is this clear and documented?" | "Should this exist, and is it minimal?" |
| Default scope | changed files (diff) | whole repo |
| Output | readability findings | a delete-list (LOC removable) |

For **duplication** specifically, defer to `code-quality-review`'s
**Code Duplication & Helper Functions** section — do not restate that
checklist here. Point the user at it when you spot repeated patterns.
Likewise, repeated *parameter values* belong to that skill's **Single
Source of Truth for Parameter Values** section.

## Arguments

- **file-or-directory** (optional): Path to audit. If omitted, audit the
  whole repo.

## What counts as in scope

- **In scope:** live `app.py` and `src/`. `tests/` is in scope only for
  unused test-helper bloat.
- **Out of scope beyond the usual gitignored paths:** the generated deploy
  artifacts `requirements.txt`, `uv.lock` and `manifest.json`. They are
  exports, so a difference between them and `pyproject.toml` means they need
  regenerating — never hand-editing.
- **Out of scope:** any gitignored or untracked path. Run `git ls-files`
  if you are unsure whether a path is tracked — do not infer scope from
  directory names.
- **Runtime dependencies deserve special attention.** An unused entry in
  `[project] dependencies` is installed by the Posit Connect deploy. Check
  each against live `import` usage, and remember that a package pulled in only
  transitively — a numeric library `pandas` or `ortools` uses but nothing here
  imports — does not belong in the direct dependency list.
- **`app.py` and `tests/` are the call sites for nearly all of `src/`.** Grep
  both before declaring a `src/` symbol dead. The live spellings are
  `from src.LP import run_lp`, `from src import parameters`, `from src.db
  import RESULTS_ZIP, ResultsDB` and `from src.utils import
  get_resource_stack_plot` — grep the symbol, not a remembered import line.
- **Watch the `__main__` blocks in `src/LP.py` and `src/db.py`.** A helper
  that looks used only there — `make_fake_results` is the example — is also
  imported by the tests. Confirm before listing it.
- **An unused branch in `run_lp` may be a real finding, but verify it against
  the call sites first.** A `use_outside_energy=False` path or a
  `restrict_gas is None` branch the app never triggers is dead in practice;
  the same branch reached from a test is not.
- **`run_lp` being long is a size signal, not bloat.** It is inherently a
  build → solve → assemble procedure. Note it once and move on rather than
  reporting it every run.

## Audit checklist (minimalism)

- **Existence / YAGNI** — functions, classes, branches, or config fields
  never referenced in live code. **Grep-confirm zero references outside
  the definition** before reporting (see Steps). Prefer deletion, then
  refactoring, then addition.
- **Reinvention** — hand-rolled logic that duplicates a built-in from the
  stdlib or from a library this project already imports. Read the
  project's actual imports rather than assuming a fixed library set, and
  cite the specific built-in that replaces the hand-rolled code.
- **Dependency justification** — cross-check each runtime dependency in
  `pyproject.toml` against live `import` usage. Flag any dep with zero or
  near-zero live imports as a removal candidate.
- **Single-use abstraction** — wrapper functions, one-method classes, or
  indirection layers used exactly once. Recommend inlining at the single
  call site.
- **Premature generalization** — parameters, branches, config knobs, or
  "flexibility" that handle cases which never occur in practice. Flag the
  unused case and the code that exists only to serve it.
- **Repo-wide dead exports** — public symbols (functions, classes,
  constants) with no references anywhere in live code. This is broader
  than `code-quality-review`, which only sees the diff.
- **Size signals** — files over ~800 lines, and functions over the length
  threshold in `code-quality-review`'s **Readability** checklist. Cite them;
  do NOT prescribe the split here, and do not carry a second copy of that
  threshold — two numbers for one rule drift apart.
- **Dead scaffolding** — commented-out code blocks and stale TODO stubs
  that were never finished.

## Output format

Start with a one-block **summary**:

- Total live LOC.
- Estimated removable LOC.
- A **top 10 cleanups by LOC removed** table: `rank | file:line | action |
  est. LOC | one-line rationale`.

Then the full **delete-list**, grouped by action:

### Delete
Confirmed dead — grep-proven zero references. Each: `file:line`, est. LOC,
one-line rationale.

### Simplify
Exists but over-built — inline the single-use wrapper, use the library
built-in, drop the unused knob. Each: `file:line`, est. LOC, what to do.

### Verify
Looks removable but needs a human check before deleting (e.g. referenced
only via dynamic dispatch, a public entry point, or an external caller).
Each: `file:line`, what to verify.

## Steps

1. **Determine scope.** If a path argument is given, audit that path.
   Otherwise audit the whole repo (the in-scope set above).
2. **Mechanical passes first** — reuse existing tooling, do not reinvent it:
   - `uv run ruff check --select F401,F811,SIM .` — unused imports (F401),
     redefinitions (F811), and simplifiable code (SIM). These are mechanical
     bloat; cite the rule code in each finding.
   - **Dependency cross-check:** for each runtime dep in `pyproject.toml`
     (`[project].dependencies`), grep its import name across live code. A
     dep with no live `import` is a removal candidate. Two traps: the
     **import name often differs from the package name** (e.g.
     `python-dotenv` → `dotenv`), and some dependencies are **never
     imported directly at all** — an engine or backend that another
     library loads under the hood is required despite having no `import`
     anywhere. Verify these cases before recommending removal.
3. **Grep-confirm dead symbols.** For every candidate from the checklist,
   grep the symbol name across the repo and confirm it has **no references
   outside its own definition** before listing it under **Delete**. If
   there is any ambiguity (dynamic dispatch, entry point, re-export),
   downgrade it to **Verify**.
4. **Grep `app.py`, `tests/` and the `__main__` blocks before deleting.**
   Those are the call sites that are easy to miss here. For anything reached
   from them, or only via a Shiny reactive binding, downgrade to **Verify**
   and say where you looked.
5. **Read the suspicious files** to confirm context before listing — do not
   report from grep counts alone.
6. **Emit the report** (summary + delete-list). Make **no edits** — this
   skill is report-only.
