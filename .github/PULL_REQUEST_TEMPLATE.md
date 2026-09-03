<!-- Write this so it stands on its own. Your reviewer has the diff and
     nothing else: no plan file, no skill checklist, no memory of the
     conversation that produced this. Say what a reference means instead of
     naming it — "the constraint that caps battery discharge", not "the fix in
     LP.py"; "the background solve the Run button starts", not "the task".
     Names and shorthand live inside the file that defines them, they shift
     when it is edited, and half the time that file is not in the diff.

     Same for the rest: no "as discussed", no ticket number standing in for
     the reason, no comparison to a version of the code the reviewer cannot
     see. CLAUDE.md's "Comments & docstrings are self-contained" is the same
     rule; this is where it applies to a PR. -->

## Summary
<!-- What does this PR do and why? Include the context needed to judge it. -->

## Changes
<!-- List the main changes in this PR -->
-

## Test plan
- [ ] Tests pass (`uv run pytest`) — includes the `slow` full LP solve
- [ ] Lint clean (`uv run ruff check .`)
- [ ] Formatted (`uv run ruff format --check .`)
- [ ] If `app.py` or `src/utils.py` (the plot) changed: `uv run pytest -m e2e` after `uv run playwright install chromium`

<!-- The `test` check runs the first three, so tick them from a local run and
     let CI be the second opinion rather than the only one.

     The browser tests are NOT in the required check — they need a browser
     download and a full background solve, so they run after the merge. A
     break in them lands first and is noticed second; if your change touches
     the UI, run them yourself. -->

## LP changes
<!-- Delete unless the model changed. A constraint or objective change moves
     numbers that no assertion pins, so say which direction they moved and
     why that is right. -->
- [ ] `test_lp.py` still passes, and a changed result is explained rather than
      absorbed into the expected value

## Dependencies
<!-- Delete unless pyproject.toml changed. requirements.txt is what the
     Connect deploy installs — regenerate it in the same PR, or the deploy and
     the repo disagree. -->
- [ ] `requirements.txt` regenerated
- [ ] `uv.lock` updated (CI runs `uv sync --locked` and fails on a stale lock)

## Manual testing
<!-- Describe any manual testing performed and results -->
