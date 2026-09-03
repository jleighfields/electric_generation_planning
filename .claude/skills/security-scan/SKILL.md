---
name: security-scan
description: Scan for leaked secrets and insecure patterns — hardcoded passwords/tokens/keys, tracked .env or credential files, secrets in logs, and unsafe defaults. Drives ruff S rules with a git + grep fallback; report-only, makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: "[file-or-directory]"
---
# Security Scan

Find secrets and unsafe patterns before they get committed, and report a
findings list. **Report-only** — never edit or delete a secret yourself.
Removing a secret and (critically) **rotating** it is a deliberate,
human-driven step.

Run tooling first, the same way `code-quality-review` runs ruff, then cover
the remaining cases with `git` checks and `grep`:

1. **`ruff` `S` rules (flake8-bandit)** — mechanical insecure-pattern
   detection (hardcoded passwords, `eval`/`exec`, `shell=True`, unsafe
   deserialization). Configured in `pyproject.toml`.
2. **`git ls-files` + `grep`** — tracked-credential-file checks and a regex
   fallback for token shapes the tools miss.

Security scan findings answer "does this leak a credential or open a hole?"
`code-quality-review` answers "is it clear?" and `simplify-audit` answers
"should it exist?"

## Arguments

- **file-or-directory** (optional): Path to scan for secret *content* /
  insecure patterns. If omitted, scan the changed files
  (`git diff --name-only HEAD` unioned with `git ls-files --others
  --exclude-standard` — union, not fallback).
  The tracked-file checks always run against the whole repo regardless of
  the argument.

## Scope

- **Content / pattern scan:** the target files (changed files by default,
  or the given path).
- **Tracked-file checks:** always whole-repo via `git ls-files` — a
  committed `.env` is a repo-wide fact, not a diff fact.
- **Never scan gitignored or untracked artifacts** for content. Confirm
  with `git ls-files` rather than guessing from directory names — but DO
  still confirm those paths are actually ignored.

## What to check

### Tracked credential files (whole repo, always)

Run `git ls-files` and flag any tracked file that should never be committed:

- `.env`, `.env.local`, `.env.*` **except** `.env.example` / `.env.template`
  / `.env.sample` (those are intended templates — see the caveat below).
- Private keys / certs: `*.pem`, `*.key`, `*.pfx`, `*.p12`, `*.keytab`,
  `id_rsa`, `id_dsa`.
- Credential dumps: `credentials.json`, `service-account*.json`,
  `*.kdbx`, `.netrc`, `.pgpass`, `.htpasswd`.
- `rsconnect-python/` config directories. The Posit Connect Cloud deploy is
  git-based and needs no key in the repo, but one written locally holds a
  working API key — and `.gitignore` does not cover it, so the coverage
  check below applies to it.

A tracked secret file is **Must Fix**: `git rm --cached` it, add it to
`.gitignore`, and **rotate** anything it exposed (it is already in history).

### `.gitignore` coverage

Confirm `.env` and the patterns above are gitignored, not merely absent. A
secret that is untracked today but not ignored can be committed by
`git add -A`.

### Hardcoded secrets — grep scan

`detect-secrets` is not set up in this repo (see the caveats below), so the
grep patterns below are the content scan rather than a supplement to one. Say
that in the report — a clean result here is weaker evidence than a clean
`detect-secrets` run, and claiming otherwise overstates what was checked.

Scan for assignments of a secret-looking name to a literal, and known
token shapes.
Pattern reference (ripgrep regex):

| What | Pattern |
|---|---|
| Secret-named literal | `(?i)(pass(word|wd)?\|secret\|token\|api[_-]?key\|client[_-]?secret\|access[_-]?key\|auth[_-]?token\|private[_-]?key)\s*[:=]\s*["'][^"']{6,}["']` |
| Private key block | `-----BEGIN (RSA \|EC \|OPENSSH \|DSA \|PGP )?PRIVATE KEY-----` |
| AWS access key id | `AKIA[0-9A-Z]{16}` |
| GitHub token | `gh[pousr]_[A-Za-z0-9]{36,}` or `github_pat_[A-Za-z0-9_]{60,}` |
| Slack token | `xox[baprs]-[A-Za-z0-9-]{10,}` |
| Bearer/JWT | `(?i)bearer\s+[A-Za-z0-9._\-]{20,}` / `eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}` |
| URL with embedded creds | `[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s:@]+@` |
| Connection-string password | `(?i)(password\|pwd)=[^;"'\s]{4,}` |

For each hit, **redact the value in your report** — show the variable name
and first few characters only, never the full secret.

### Secrets passed to logging or print

ruff `S` does not cover this. Flag log and print statements that
interpolate a secret-bearing value: `Authorization` headers, anything named
`*secret*` / `*token*` / `*password*`, or a whole `headers` / `auth` dict.
Even at DEBUG a logged token ends up in the deployment's log store, which
is usually retained longer and read by more people than the code.

### Insecure patterns — ruff `S` rules

`ruff check --select S` reports these patterns; cite the rule code in each
finding (as `code-quality-review` cites ruff codes):

- **Hardcoded password** — `S105` (string), `S106` (func arg), `S107`
  (default arg).
- **`eval` / `exec`** — `S307` (eval), `S102` (exec).
- **Unsafe deserialization** — `S301` (`pickle`), `S506` (`yaml.load`
  without `SafeLoader`).
- **Shell injection surface** — `S602`/`S604`/`S605` (`shell=True`),
  `S607` (partial executable path).
- **TLS verification disabled** — `S501` (`verify=False`).
- **Other** — `S104` (bind all interfaces), `S324` (weak hash). The
  `--host 0.0.0.0` in `docker/Dockerfile` is deliberate for container
  serving, not a leaked bind-all bug.

Treat any `S`-rule hit on the target files as **at least Should Fix**;
`S105-S107` (hardcoded secret) is **Must Fix**. Do not re-flag the
configured ignores below.

## Configured exceptions (don't re-flag these)

`pyproject.toml` encodes the legitimate exceptions — respect them. **Read
the actual per-file ignores rather than assuming**; they are the authority,
and this section describes only the common case.

- **`tests/**` ignores `S101` only.** pytest tests use `assert`, so
  `S101` is expected there. **The hardcoded-secret rules `S105-S107` stay
  ACTIVE in tests**, because a real credential pasted into a fixture is
  exactly as leaked as one in production code. Never suppress an
  `S105-S107` hit under `tests/` as a configured exception — it is a
  **Must Fix**.
- **`src/db.py` ignores `S608` and `S101`.** It interpolates only fixed
  table names from its own constants, and its `__main__` smoke test uses
  `assert`. An `S608` anywhere else is a real finding.
- **`scripts/**` ignores `S603`.** The deploy script shells out to
  `uv`/`rsconnect` with hardcoded argument lists, no user input.

## The `.env.example` caveat

- **`.env.example` is supposed to be committed.** Do NOT flag it as a
  tracked secret file. Instead, **open it and confirm every value is a
  placeholder** — a real value leaked into the template IS a finding, and a
  Must Fix.

## Repo-specific caveats

- **This is a pure-compute app with no live credential surface.** The LP
  model, the Shiny UI and an in-memory SQLite store are the whole of it; it
  reads no environment secrets and talks to no authenticated service. So the
  job of this scan is to catch a credential the moment one is *newly
  introduced*, not to audit an existing surface. Confirm the absence rather
  than assuming it: grep for `os.environ` and `getenv` and say what you found.
- **There is no `.env` and no `.env.example`.** A tracked `.env` appearing is
  therefore a real finding rather than the usual false positive, and there is
  no template to audit for a leaked real value.
- **`detect-secrets` is not set up here, and that is deliberate** — there are
  no secrets to baseline. No step below runs it, so say so in the report
  rather than reporting a clean scan it did not perform. If this project ever grows a credential surface, add
  `detect-secrets` and a committed `.secrets.baseline` and fold it in as the
  primary content scanner.
- **`src/db.py` builds SQL with f-strings, and `S608` there is a false
  positive.** It interpolates only fixed table names from the
  `self.table_names` constants — no user input reaches the query. The ignore
  is scoped to that file; an `S608` anywhere else is a real finding.
- **The SQLite store is `:memory:` and per-session**, so there is no on-disk
  database file to leak and none to find tracked.
- **`rsconnect-python/` config directories hold a working API key once
  written.** The Posit Connect Cloud deploy is git-based and needs no key in
  the repo, so one appearing is a finding.

Each caveat has to be anchored to something durable: a file that exists, a
documented convention. A caveat naming a deleted module authorizes an
exception that no longer applies.

## Output format

Group findings by severity, same buckets as `code-quality-review` so the
`code-reviewer` agent's report stays uniform. **Number findings
sequentially**, continuing from the review findings rather than restarting
at 1, so a closing list of open items is unambiguous about which section
each belongs to.

### Must Fix
- Confirmed live secret in code, a tracked `.env`/key file, a real value
  in `.env.example`, or an `S105-S107` hit. **Always include the
  remediation:** remove it, move it to an environment variable / `.env`,
  and **rotate the exposed credential** — state explicitly that
  working-tree removal is not enough if it was ever committed, since it
  persists in git history; scrub with `git filter-repo` / BFG if needed.

### Should Fix
- Other insecure-pattern `S`-rule hits (`verify=False`, `eval`/`exec`,
  unsafe deserialization, `shell=True`) and secret logging.

### Consider
- Possible-but-uncertain matches that might be fixtures, `.gitignore` gaps
  with nothing currently leaked.

For each finding: a sequential number, `file:line`, the rule code where
applicable, what matched (**redacted**), why it is a risk, and the
suggested fix.

## Steps

1. **Pick targets.** Path argument → that path. Otherwise
   `git diff --name-only HEAD` unioned with `git ls-files --others
   --exclude-standard`. Union, not fallback — see `code-quality-review`.
   The tracked-file and `.gitignore` checks run whole-repo regardless.
2. **Mechanical passes first** — reuse the configured tooling:
   - `uv run ruff check --select S <targets>` — insecure patterns. Cite
     each rule code.
3. **Tracked-file check.** `git ls-files`, filtered for the credential-file
   patterns above, applying the `.env.example` exception. Confirm
   `.gitignore` coverage.
4. **Fallback content scan.** Run the grep patterns over the target files
   for anything the tools did not surface, then check for secret logging.
   Open each hit to confirm it is a real secret versus a placeholder or
   fixture; redact before reporting.
5. **Emit the report** grouped by severity. Make **no edits** — this skill
   is report-only. If anything is Must Fix, lead with it. If there are no
   findings, say "No security findings" explicitly so the clean result is
   on record.
