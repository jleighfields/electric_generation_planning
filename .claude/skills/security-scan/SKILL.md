---
name: security-scan
description: Scan for leaked secrets and insecure patterns — hardcoded passwords/tokens/keys, tracked .env or credential files, secrets in logs, and unsafe defaults. Drives ruff S rules with a git + grep fallback; report-only, makes no edits.
disable-model-invocation: false
allowed-tools: Read, Glob, Grep, Bash
argument-hint: [file-or-directory]
---

# Security Scan

Find secrets and unsafe patterns before they get committed, and report a
findings list. **Report-only** — never edit or delete a secret yourself.
Removing a secret and (critically) **rotating** it is a deliberate,
human-driven step.

This skill leans on tooling first, the same way `code-quality` leans on
ruff, then fills gaps with `git` checks and `grep`:

1. **`ruff` `S` rules (flake8-bandit)** — mechanical insecure-pattern
   detection (hardcoded passwords, `eval`/`exec`, `shell=True`, unsafe
   deserialization, weak hashes). Per-file exceptions are configured in
   `pyproject.toml` (`tests/**` ignores `S101`).
2. **`git ls-files` + `grep`** — tracked-credential-file checks and a
   regex fallback for token shapes ruff misses.

> **Context for this repo:** it is a pure-compute app — the LP model, the
> Shiny UI, an in-memory SQLite store. It reads **no** environment
> secrets and talks to no authenticated service at runtime (grep confirms
> no `os.environ` / `getenv` credential reads). So there is no live
> credential surface today; the job of this scan is to catch a secret or
> credential file the moment one is *newly introduced*. (`detect-secrets`
> is not set up here — there are no secrets to baseline. If the project
> ever grows a credential surface, add `detect-secrets` + a committed
> `.secrets.baseline` and fold it in as the primary content scanner.)

Its lens (does this leak a credential or open a hole?) is distinct from
`code-quality` (is it clear?) and `simplify-audit` (should it exist?).

## Arguments

- **file-or-directory** (optional): Path to scan for secret *content* /
  insecure patterns. If omitted, scan the changed files
  (`git diff --name-only HEAD`, plus untracked from `git status --short`).
  The tracked-file checks always run against the whole repo regardless of
  the argument.

## Scope

- **Content / pattern scan:** the target files (changed files by default,
  or the given path).
- **Tracked-file checks:** always whole-repo via `git ls-files` — a
  committed `.env` is a repo-wide fact, not a diff fact.
- **Never scan** gitignored/untracked artifacts for content (`.venv/`,
  `csv/`, `results.zip`, `__pycache__/`) — but DO still confirm they are
  gitignored.

## What to check

### 1. Tracked credential files (whole repo, always)

Run `git ls-files` and flag any tracked file that should never be committed:

- `.env`, `.env.local`, `.env.*` **except** `.env.example` / `.env.template`
  / `.env.sample` (those are intended templates).
- Private keys / certs: `*.pem`, `*.key`, `*.pfx`, `*.p12`, `*.keytab`,
  `id_rsa`, `id_dsa`.
- Credential dumps: `credentials.json`, `service-account*.json`,
  `*.kdbx`, `.netrc`, `.pgpass`, `.htpasswd`,
  `rsconnect-python/` config directories (the Posit Connect Cloud deploy
  is git-based and needs no key in the repo, but an rsconnect API-key
  file must never be committed).

A tracked secret file is **Must Fix**: `git rm --cached` it, add it to
`.gitignore`, and **rotate** anything it exposed (it is already in history).

### 2. `.gitignore` coverage

Confirm `.env` (and the patterns above) are gitignored, not merely
absent. A secret that is untracked today but not ignored is one
`git add -A` away from being committed. This repo's `.gitignore` already
covers `.env`.

### 3. Hardcoded secrets (target files) — grep scan

Scan for assignments of a secret-looking name to a literal, and known
token shapes. Pattern reference (ripgrep regex):

| What | Pattern |
|---|---|
| Secret-named literal | `(?i)(pass(word|wd)?\|secret\|token\|api[_-]?key\|client[_-]?secret\|access[_-]?key\|auth[_-]?token\|private[_-]?key)\s*[:=]\s*["'][^"']{6,}["']` |
| Private key block | `-----BEGIN (RSA \|EC \|OPENSSH \|DSA \|PGP )?PRIVATE KEY-----` |
| AWS access key id | `AKIA[0-9A-Z]{16}` |
| GitHub token | `gh[pousr]_[A-Za-z0-9]{36,}` or `github_pat_[A-Za-z0-9_]{60,}` |
| Bearer/JWT | `(?i)bearer\s+[A-Za-z0-9._\-]{20,}` / `eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}` |
| URL with embedded creds | `[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s:@]+@` |
| Connection-string password | `(?i)(password\|pwd)=[^;"'\s]{4,}` |

This app uses no credentials today, so **any** hit here is suspicious —
open it and confirm whether it is a real secret or a false positive
(a docstring, a test fixture, a variable named `token` holding something
harmless). A confirmed hardcoded secret is a **Must Fix**.

For each hit, **redact the value in your report** (show the variable name
and first few chars only, never the full secret).

### 4. Secrets passed to logging or print

ruff `S` does not cover this. This repo logs via the `logging` module
(`src/LP.py`, `src/db.py`) — flag log/print statements that would
interpolate a secret-bearing value. There is nothing sensitive logged
today (the model logs capacities, metrics, and timings); the check exists
to catch a regression if a credential is ever introduced.

### 5. Insecure patterns — ruff `S` rules

These fire mechanically from `ruff check --select S`; cite the rule code in
each finding (like `code-quality` cites ruff codes):

- **Hardcoded password** — `S105` (string), `S106` (func arg), `S107`
  (default arg).
- **`eval` / `exec`** — `S307` (eval), `S102` (exec).
- **Unsafe deserialization** — `S301` (`pickle`), `S506` (`yaml.load`
  without `SafeLoader`).
- **Shell injection surface** — `S602`/`S604`/`S605` (`shell=True`),
  `S607` (partial executable path), `S603` (subprocess untrusted input).
- **TLS verification disabled** — `S501` (`verify=False`).
- **SQL injection** — `S608` (string-built query). Note `src/db.py` builds
  SQL with f-strings interpolating **`table_name` from the fixed
  `self.table_names` list** (code-controlled constants) — that is safe;
  a query interpolating user/app input would be a finding.
- **Bind-all / temp files** — `S104` (bind all interfaces), `S108`
  (insecure temp). The Docker/`shiny run` `--host 0.0.0.0` is intentional
  for containers, not a code `S104`; judge the context.
- **Other** — `S324` (weak hash), `S110` (try/except/pass — the
  `ResultsDB` methods catch-and-log rather than pass; confirm they log).

Treat any `S`-rule hit on the target files as **at least Should Fix**;
`S105-S107` (hardcoded secret) is **Must Fix**. Do not re-flag the
configured ignores (`tests/**` → `S101`).

## Configured exceptions (don't re-flag these)

`pyproject.toml` (`[tool.ruff.lint.per-file-ignores]`) encodes the
legitimate exceptions — respect it:

- **`tests/**`** ignores `S101`: pytest's whole model is `assert`, and
  test data is code-generated fixtures, not real creds.

## False-positive caveats specific to this repo

Apply these so the scan is accurate, not noisy:

- **The SQLite store is `:memory:` and per-session** — no on-disk DB file
  with data to leak.
- **`src/db.py` f-string SQL** interpolates only fixed table names from
  `self.table_names` (constants), so `S608` there is a false positive.
  Real user input goes through parameterized queries (`?` placeholders) —
  confirm that stays true.
- **`--host 0.0.0.0`** in the Dockerfile / `shiny run` is deliberate for
  container/deploy serving, not a leaked bind-all bug.

## Output Format

Group findings by severity, same buckets as `code-quality` so the
code-reviewer agent's report stays uniform:

### Must Fix
- Confirmed live secret in code, a tracked `.env`/key file, or an
  `S105-S107` hit. **Always include the remediation:** remove it, move it to
  an environment variable / `.env`, and **rotate the exposed credential**
  (state explicitly that working-tree removal is not enough if it was ever
  committed — it persists in git history; scrub with `git filter-repo` / BFG
  if needed).

### Should Fix
- Other insecure-pattern `S`-rule hits (`verify=False`, `eval`/`exec`,
  unsafe deserialization, `shell=True`, SQL injection) and secret logging.

### Consider
- Possible-but-uncertain matches, `.gitignore` gaps with nothing currently
  leaked.

For each finding: `file:line`, the rule code where applicable, what matched
(**redacted**), why it is a risk, and the suggested fix. If the scan is
clean, say "No security findings" explicitly so the clean result is on
record.

## Steps

1. **Pick targets.** Path argument → that path. Otherwise
   `git diff --name-only HEAD`; if empty, `git status --short` for
   untracked files. The tracked-file checks (1–2) run whole-repo regardless.
2. **Mechanical pass first:**
   - `uv run ruff check --select S <targets>` — insecure patterns (check 5).
     Cite each rule code.
3. **Tracked-file check.** `git ls-files`, filter for the secret-file
   patterns in check 1, applying the `.env.example` exception. Confirm
   `.gitignore` coverage (check 2).
4. **Content scan.** Run the check-3 grep patterns over the target files,
   then check 4 (secret logging). Open each hit to confirm it is a real
   secret vs a placeholder/fixture; redact before reporting.
5. **Emit the report** grouped by severity. Make **no edits** — this skill
   is report-only. If anything is Must Fix, lead with it.
