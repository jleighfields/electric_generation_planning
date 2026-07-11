---
name: code-quality
description: Review code for readability, documentation quality, onboarding ease, and minimal form (simplification per the project Minimalism rules). Report-only — presents findings, makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: [file-or-directory]
---

# Code Quality Review

Review code for readability, documentation quality, ease of
on-boarding, and whether the code is in its minimal form
(simplification per the project Minimalism rules). Report issues but do
NOT make edits — present findings for the user to approve.

(Named `code-quality` rather than `code-review` to avoid colliding with
Claude Code's built-in `/code-review` command.)

## Arguments

- **file-or-directory** (optional): Path to review. If omitted, review
  all staged and unstaged changed files (`git diff --name-only HEAD`).

## Review Checklist

### Readability

1. **Function length** — flag functions > 50 lines. Can they be split?
   (`run_lp` in `src/LP.py` is long by nature — model build + solve +
   result assembly; judge its *sections*, not the whole.)
2. **Variable names** — are they descriptive? Flag single-letter vars
   (except `h` for the hour index and `i` for a loop index; short-lived
   `df` is acceptable)
3. **Nesting depth** — flag > 3 levels of nesting. Can early returns
   or guard clauses simplify?
4. **Magic numbers** — flag hardcoded values without explanation. The
   model is full of domain constants (emission factors, heat rate, VOM,
   asset lifetimes); each needs a comment saying what it is and where it
   came from.
5. **Dead code** — commented-out code, unused imports, unreachable branches
6. **Silent failures** — missing file/dir checks that skip without logging
   a warning; a bare `except` that swallows an error (the `ResultsDB`
   methods catch and log — confirm they still log, not silently pass)
7. **Arbitrary decisions** — thresholds, caps, multipliers, or logic
   branches with no comment explaining *why* that value or approach was
   chosen (e.g., the `0.05` total-import cap, the `0.2` hourly-import cap,
   or the tiny charge/discharge objective penalties in `LP.py`)

### Documentation

8. **Missing docstrings** — every public function and class needs one
9. **Outdated docstrings** — does the docstring match the current code?
   (`run_lp`'s docstring lists every input key — keep it in sync.)
10. **Missing type hints** — all function signatures need types
11. **Confusing comments** — comments that describe *what* instead of *why*
12. **Missing comments** — complex logic without explanation

### Style (per CLAUDE.md)

13. **Google-style docstrings** with summary, Args, Returns
14. **`X | None`** syntax (not `Optional[X]`)
15. **Direct imports** for type hints (no forward references)
16. **No `_` prefix** on function names (except internal helpers)

### Code Duplication & Helper Functions

Flag repeated patterns and recommend concrete extractions. This is
a **Should Fix** at 2 copies and a **Must Fix** at 3+.

17. **Near-identical functions / blocks** — two or more that share >50%
    of their logic with only minor differences. The `capacity_boxes` /
    `cost_boxes` / `generation_boxes` renderers in `app.py` are the
    canonical local example: each guards on `results.get()`, then returns
    `ui.TagList(ui.hr(), ui.h4(...), ui.layout_columns(*value_boxes))`.
    Extract the shared shell into a helper
    (`metric_section(title, boxes) -> ui.TagList`) and make each renderer
    a thin caller.
18. **Copy-pasted logic with small variations** — loops, conditions, or
    data-processing blocks clearly copied and tweaked (e.g., the two
    `net_load` branches in `LP.py` that differ only by the
    `outside_energy` term). Extract into a function parameterized on the
    varying part.
19. **Hardcoded values repeated across files** — the same magic number or
    string appearing in 2+ files without a shared constant. The emission
    factors and cost formulas now live once in `parameters.cost_inputs`
    (called by both `app.build_inputs` and `parameters.get_base_inputs`) —
    flag any diff that re-inlines that math back into the app or model.
    See the **Single Source of Truth** section below.
20. **When NOT to extract** — do not flag single-use patterns shorter
    than 3 lines, or test setup code (test clarity > DRY).

When flagging duplication, always include:
- Which functions/blocks are duplicated
- How many copies exist
- A concrete suggested helper signature (name, parameters, return type)

For the broader minimalism lens (code that shouldn't exist, reinvention
of built-ins, premature abstraction), see **Simplification** below
(items 30–34).

### Single Source of Truth for Parameter Values

A high-priority class of review findings. Parameter values (numbers,
dicts, config entries) get ONE home; other modules read from it.
Duplicate-source-of-truth bugs drift silently and are expensive to
debug. Flag every instance, with severity calibrated to blast radius:

- **Must Fix** when the duplication is in the production model path
  (`src/`, `app.py`) or when the copies have already drifted (different
  values).
- **Should Fix** when the duplication is in plots or tests but the values
  currently agree (drift is a matter of time).

See CLAUDE.md §"Single source of truth for parameter values" for where
canonical values live in this repo.

21. **Hardcoded scalar where a `parameters.get_base_inputs()` value
    exists** — e.g., the app or a test re-declaring a capacity bound, the
    `restrict_gas` percent, or a battery parameter instead of sourcing it
    from the model defaults in `src/parameters.py`. Two sources for the
    same value drift on the next edit.
22. **Re-inlined emission / cost math** — the formulas live once in
    `parameters.cost_inputs`. **Must Fix** any diff that copies that math
    back into `app.build_inputs` or the model instead of calling the
    shared function.
23. **Mismatched defaults for the same logical value** — a function
    default that resolves differently from the constant the production
    path uses. Caller omits the kwarg → silent split between tests and the
    app. Fix: align defaults, or require the parameter.
24. **A plot or metric re-computing a value the model already produced** —
    e.g., re-deriving a total from `final_df` instead of reading the
    matching entry in `metrics`. Read the model's output; don't fork the
    math.
25. **A hardcoded results-schema name** — the table/metric keys
    (`'inputs'`, `'cap_mw'`, `'metrics'`, `'final_df'`) live once in
    `ResultsDB.table_names` (`src/db.py`). Flag a parallel hardcoded list.

When flagging a config-source violation, include:
- The two (or more) source locations
- Whether they currently agree (same value, just two copies) or already
  drift (different values)
- Which one is canonical (`src/` model code usually wins)
- The signature change required to consolidate

### On-boarding Ease

26. **Would a new team member understand this?** — flag sections that
    need context comments
27. **Are error messages helpful?** — do assertions/log lines explain what
    went wrong (e.g., a non-optimal solver status)?
28. **Are log messages informative?** — do they help debug a failed or
    surprising solve?
29. **Is the README up to date?** — does it reflect the current code?

### Simplification (per Minimalism rules)

Apply the **Minimalism (write less)** hierarchy from `CLAUDE.md` to the
changed code: walk it top to bottom and flag where the diff skipped a
cheaper step. This lens is about the *form* of the new code (is it
minimal?), as distinct from **Duplication** (items 17–20), the **Single
Source of Truth** section (items 21–25), and the `simplify-audit` skill
(is there dead/excess code across the *whole repo*?). Report only.

30. **Existence / YAGNI** — does the new code need to exist? Flag
    speculative helpers, unused parameters, config knobs, or branches the
    diff adds "just in case" with no current caller.
31. **Reinvention** — hand-rolled logic that duplicates a built-in from the
    stdlib or an already-imported library (`pandas`, `ortools`, `plotly`,
    `shiny`). Cite the built-in that replaces it (e.g. a manual row loop
    that a vectorized pandas op or `df.assign(...)` does in one line).
32. **Single-use abstraction** — a wrapper, helper, or class the diff
    introduces for exactly one call site. Recommend inlining. This is the
    inverse of items 17–20: extract at 2–3 copies, inline at one.
33. **Premature generalization** — parameters, `**kwargs`, or branches that
    handle cases which do not occur in the codebase yet.
34. **Smallest correct form** — multi-line constructs that collapse to a
    comprehension, vectorized op, or single call; needless intermediate
    variables.

For each simplification finding, cite the Minimalism step (1–6) it maps
to so the user sees which rule applies.

## Output Format

Group findings by severity. **Number findings sequentially across
all three buckets** (1, 2, 3, ...) starting at 1 in Must Fix and
continuing through Should Fix and Consider. Sequential numbering
gives every finding a unique short ID the user can reference in
conversation ("apply 1, 4, 7", "skip #11").

### Must Fix
- Critical issues (wrong docstrings, misleading comments, missing types)

### Should Fix
- Important readability issues (long functions, missing comments)

### Consider
- Style suggestions, minor improvements

For each finding, include:
- A leading sequential number (continuing from prior bucket)
- File and line number
- What the issue is
- Suggested fix (brief)

Example layout:

```
## Must Fix

### 1. `src/LP.py:42` — function raises but lacks Raises: section
...

## Should Fix

### 2. `app.py:200-260` — three metric renderers duplicate the same shell
...

## Consider

### 3. `src/utils.py:30` — variable name `x` could be `hour`
...
```

## Steps

1. Identify files to review:
   - If a file or directory argument is provided, review those files
   - **If no argument is provided**, you MUST run this command to find
     all changed files and review every one of them:
     ```bash
     git diff --name-only HEAD
     ```
     If this returns no files, also check for untracked files:
     ```bash
     git status --short
     ```
     Review ALL files returned — do not skip any.
2. Run static checks on the changed files first — these surface issues
   mechanically before you start reading:
   - `uv run ruff check <changed-files>` — unused imports, undefined
     names, style violations
   - `uv run ruff format --check <changed-files>` — formatter drift
   - `uv run pytest -m "not slow and not e2e"` — the fast unit gate
   - **If any changed file touches the app surface** (`app.py` or
     `src/utils.py`, the plot), also run the Playwright end-to-end suite:
     `uv run pytest -m e2e`. The unit suite does NOT exercise the rendered
     app, so render bugs an `import`/unit pass can't see only surface by
     driving a real browser. Report a failure as **Must Fix**. Needs
     Playwright browsers (`uv run playwright install chromium`); if they
     can't launch, say so and recommend the user run it rather than
     silently skipping.

   Treat any ruff finding as **at least Should Fix**; F821 (undefined
   name) and most B-class rules are **Must Fix** since they're real
   bugs. Cite the rule code (e.g. `F401`, `B008`) in each finding so
   the user knows what `--fix` would do. Skip checklist items 5
   (dead code/unused imports) and 15 (forward references) — ruff
   covers them more reliably than human review. **Do not report `S`
   (flake8-bandit) findings here** — those are security concerns owned
   by the `security-scan` skill (the `code-reviewer` agent runs it as a
   separate phase); reporting them here would double-count.
3. Read each file fully — do not skip any changed files
4. Apply the checklist to every changed file, paying special attention
   to code duplication (items 17-20) and simplification (items 30-34).
   For each duplication finding, include a concrete helper signature so
   the fix is actionable; for each simplification finding, cite the
   Minimalism step (1-6) it maps to.
5. Present findings grouped by severity, ruff findings cited inline
6. Do NOT make edits — let the user decide what to fix
7. After presenting findings, suggest running ``/comment-docstring``
   on the changed files to fill any docstring / type-hint / inline-
   comment gaps. Do not invoke it automatically — that's a separate
   editing step the user should opt into.

## After the review: present, then gate

This skill is report-only: **present every finding (numbered) for the
user to review — never auto-fix.** But the findings are a tracked
checklist, not advisory prose:

- Treat each **Must Fix** and **Should Fix** as an open item referenced
  by its number. Do **not** commit or merge the reviewed code until every
  such item is either fixed or **explicitly waived by the user**.
- "Pre-existing" / "out of scope" is never a silent pass — name the
  finding and get the user's waiver; do not drop it by omission.
- **Consider** items are optional and do not block.
