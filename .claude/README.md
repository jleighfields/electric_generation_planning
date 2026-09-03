# Skills and agents

Committed Claude Code configuration. Names and descriptions are injected at
session start, so a newly added skill or agent stays invisible until the
session restarts.

## Skills — `skills/`

Each is a slash command, invocable directly, and each is one phase of a review.

| Skill | What it asks | Edits? |
|---|---|---|
| `code-quality-review` | Is this correct, clear, documented, minimal? | no |
| `security-scan` | Does this leak a credential or open a hole? | no |
| `comment-docstring` | Are the docstrings, type hints and nearby prose right? | **yes** |
| `test-review` | Can these tests fail when the code they name is wrong? | no (mutates a throwaway worktree) |
| `simplify-audit` | Should this code exist, and is it minimal? | no |

`code-quality-review` is named to avoid colliding with Claude Code's built-in
`/code-review`.

## Agents — `agents/`

| Agent | Runs | Default target |
|---|---|---|
| `code-reviewer` | `code-quality-review` → `security-scan` → `comment-docstring`, and in the full pass `test-review` plus a phase pinning confirmed defects with a failing test | `commit`: the changed files. No argument: the branch diff against `origin/main` |
| `simplify-auditor` | `simplify-audit`, in its own context | the whole repo |

Both return a finished report rather than their search transcript — that is
what keeps repo-wide grep output out of the main session.

## Which file owns which rule

This is the split that stops these files re-growing into copies of each other:

- **What to look for, and what counts as a finding** → the **skill** running
  that phase, where it also applies to a direct invocation.
- **How a pass is ordered, composed, and escalated** → the **agent**.
- **What every session needs before starting work** → `CLAUDE.md`, which loads
  every session and pays for its length each time.
- **What a contributor needs to run or extend something** → `README.md`.

## Where the shared facts live

Point at these rather than transcribing them; a copy in a skill file drifts
silently and nothing reports it.

| Fact | Owner |
|---|---|
| Lint rule selection and per-file ignores | `pyproject.toml` `[tool.ruff.lint]` |
| Test markers and what each covers | `pyproject.toml` `[tool.pytest.ini_options]` |
| The commands the required check runs | `.github/workflows/test.yml` |
| The deploy bundle's exclude list | `scripts/update_manifest.py` |
| Model inputs and their defaults | `src/parameters.py` |
| Results table names | `ResultsDB.table_names` in `src/db.py` |
