---
name: comment-docstring
description: Review Python files for missing docstrings, type hints, and inline comments. Generate Google-style docstrings, suggest helpful comments for onboarding, and sweep the README for stale references the changes invalidate.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Edit, Bash
argument-hint: <file-or-directory>
---

# Comment & Docstring Review

Scan Python files for missing or incomplete documentation. Sweep the
README for stale prose the changes invalidate. Fix issues in place and
report what was changed.

## Arguments

- **file-or-directory** (required): Path to a `.py` file or directory
  to scan. If a directory, scan all `.py` files recursively.

## What to Check

For each function and class:

1. **Docstring exists** — every public function needs one
2. **Google style** — summary line, Args section, Returns section
3. **Type hints** — all parameters and return types annotated
4. **Summary is accurate** — matches what the function actually does
5. **Parameters documented** — every parameter listed with type and description
6. **Returns documented** — return type and meaning described

For the file body:

7. **Inline comments** — add comments that explain *why*, not *what*
8. **Section headers** — this repo uses `#####`-bar / `# --- Section ---`
   headers to break up long procedures (see `src/LP.py` and the server
   function in `app.py`); match the style already present in the file
9. **Magic numbers** — explain any hardcoded value (emission factors,
   asset lifetimes, heat rate, VOM, import caps, objective penalties)
10. **Complex logic** — add comments for non-obvious algorithms (the SOC
    battery balance, the load-serving constraint, the objective terms)

## Style Rules

Follow the project's CLAUDE.md conventions:

- Google-style docstrings with summary, Args, and Returns
- Use `X | None` syntax (not `Optional[X]`)
- Import modules directly for type hints (no forward references)
- Prioritize simplicity and readability
- Add comments that help someone on-boarding to the project
- Do not prepend function names with `_` (except internal helpers)

## README sweep

Every pass under this skill should also sweep the repo's prose docs
for content that the diff just invalidated — stale formulas, removed
function/column names, retired flags, outdated commands. README drift is
a common silent-failure mode: the code moves, the doc says it didn't,
and the next developer follows the README into the wrong mental model.

### Which docs to check

| Touched file under… | Sweep these docs |
|---|---|
| `src/`, `app.py` | root `README.md` |
| Anything that renames a results-schema key, a model input, an env/deploy step, or changes the run/test/Docker commands | root `README.md` |
| Anything that changes repo-level architecture, conventions, or shared SSoT locations | `CLAUDE.md` (root) and `README.md` (root) |

### What to look for

Run a targeted grep over the doc set with the names of every removed /
renamed / refactored symbol from the diff. Common shapes of stale prose
to flag:

- **Removed constants or functions** — e.g., the README or CLAUDE.md
  citing a helper or input key that was renamed or deleted. Update the
  prose to point at the surviving mechanism (or drop the bullet).
- **Renamed columns / result keys** — a `final_df` column or a
  `metrics` / `cap_mw` key referenced by hand in the README's
  feature list.
- **Stale commands** — run/test/Docker/deploy instructions that no
  longer match `pyproject.toml`, the pytest markers, or the Dockerfile
  (e.g. the `uv run shiny run app.py` / `uv run pytest -m ...` lines).
- **Stale version facts** — the pinned Python version, or the
  "generated from uv" note on `requirements.txt` / `manifest.json`.

### What NOT to do

- **Don't add dates or "removed in" history** to the README. Prose
  describes the current state; `git log` carries the history. Write
  "the app serves via `uv run shiny run app.py`", not "switched from
  Streamlit on 2026-07-10".
- Don't write a "Migration notes" / "Changelog" section to track a
  refactor.

## Example Output

```python
def get_resource_stack_plot(
    final_df: pd.DataFrame,
    plot_range_start_default: str = "2030-07-01",
    plot_range_end_default: str = "2030-07-14",
) -> go.Figure:
    """Build the stacked hourly generation-vs-load chart.

    Stacks the hourly resource dispatch (hydro, solar, wind, battery
    discharge, gas, and emergency energy when used) as filled areas and
    overlays the load and load-plus-charge lines, with a range selector
    and slider defaulting to the given window.

    Args:
        final_df: Hourly solved values from run_lp, indexed by timestamp,
            with the resource, load, and load_and_charge columns.
        plot_range_start_default: Initial x-axis window start (ISO string).
        plot_range_end_default: Initial x-axis window end (ISO string).

    Returns:
        A Plotly figure of the hourly resource stack and load lines.
    """
```

## Steps

1. Read the target file(s)
2. For each function/class, check docstring completeness
3. For each function signature, check type hints
4. Scan for places where inline comments would help
5. Apply the "README sweep" section above — pick the doc set from the
   matching-files table, grep for every removed / renamed symbol from the
   diff, and fix any prose the change invalidated.
6. Make edits directly (don't just report — fix)
7. Run `uv run pytest -m "not slow and not e2e"` to verify nothing broke
8. Report a summary of changes made (including any README / CLAUDE.md
   updates)
