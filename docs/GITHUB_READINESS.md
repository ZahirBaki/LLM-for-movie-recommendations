# GitHub readiness

Audited 2026-10-09. Ready for a reviewed commit and ordinary push. No Git mutation was performed. No files were deleted. The index is empty of staged changes. A network-enabled production build passed; an initial restricted build could not fetch the existing Google Fonts.

## Repository configuration and state

- Existing repository, branch `main`, upstream `origin/main`, origin `https://github.com/ZahirBaki/LLM-for-movie-recommendations.git`. No embedded remote credential found.
- Eight reachable commits across local refs; `main` is one commit ahead and zero behind the cached origin/main. No fetch was performed, so actual remote changes/protection/permissions remain unverified.
- Git author name/email configured; `core.autocrlf=true`; no custom hooks path or active local hooks. Identity values are not reproduced here.
- At initial audit: 165 tracked paths, 138 untracked files, 0 staged files. The large deleted/untracked list represents the previous 145-file relocation. New audit documents subsequently add two untracked files; final proposed tracked-existing plus non-ignored untracked inventory contains 171 files. Do not interpret old source-path deletions as lost code: consult REORGANIZATION_MANIFEST.json.

## Files safe to commit

Commit backend/frontend source and configuration, frontend/package-lock.json, requirements files, public blank .env.example, README, documentation/manifests, all tests, developer/evaluation scripts, verified external fixture, historical non-human measurement outputs, data/processed/movie_mapping.csv and data/raw/movies.csv. Historical prototypes stay included until retirement is approved. No new dependency was added.

## Files that remain local

Root .env and any environment overrides, credential key files and .aws configuration; .venv and node_modules; Python caches, .next, build/coverage and TypeScript caches; logs and isolated synthetic browser fixture storage; genuine evaluation/results/human records; raw ml-32m dataset; generated TMDB catalog, embeddings, saved SVD factors/index. They are ignored. Share required runtime assets separately through an authorized distribution workflow; a source-only clone needs the local artifacts described in README/PROJECT_STRUCTURE.

## Credential audit

Scanned proposed tracked-existing/untracked-not-ignored file contents and all reachable historical blobs under 20 MiB (all reachable blobs are below that threshold) for OpenAI/GitHub/AWS key signatures, private-key headers and exact matches to locally configured credential values. Worktree and history had zero matches. Additional literal key/password/secret/token assignments found only two explicit test placeholders in tests/backend/test_chat.py. Ignored .env contents were used only for in-memory matching; no key value was printed or written to reports.

No environment/credential file is currently tracked. Ignore rules do not untrack existing files, and this signature/exact-value audit cannot prove the absence of every possible secret type. Do not force-add local configuration. Genuine human reviews remain ignored; public historical measurements contain algorithm diagnostics, not those private sessions.

## File sizes

GitHub warns above 50 MiB and blocks regular Git files above 100 MiB; browser uploads have a 25 MiB limit. [GitHub file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Largest proposed commit file: data/processed/movie_mapping.csv, 6,346,397 bytes (6.05 MiB). Next: evaluation/results/recommendation_quality.json, 1,633,462 bytes (1.56 MiB). No proposed file reaches any limit. Largest reachable historical blob: old data/svd_movie_factors.npy, 6,413,728 bytes (6.12 MiB), so history has no file-size push blocker. Older saved SVD factors/index remain in history despite being ignored in the current tree; no history rewrite is needed or authorized.

| Local asset | Bytes | MiB | Policy |
|---|---:|---:|---|
| `data/models/movie_embeddings.npy` | 95,843,456 | 91.40 | Keep ignored/local |
| `data/models/svd_movie_factors.npy` | 6,413,728 | 6.12 | Keep ignored/local |
| `data/processed/movie_mapping.csv` | 6,346,397 | 6.05 | Safe mapping intended for commit |
| `data/processed/tmdb_movies.csv` | 35,986,987 | 34.32 | Keep ignored/local |
| `data/raw/ml-32m/ratings.csv` | 877,076,222 | 836.45 | Keep ignored/local |
| `data/raw/ml-32m/tags.csv` | 72,353,890 | 69.00 | Keep ignored/local |

ratings.csv exceeds 100 MiB. tags.csv and embeddings exceed the 50 MiB warning threshold; embeddings are below 100 MiB. These are excluded, not planned for commit. Do not force-add them; a future asset-sharing decision may need Git LFS or separate storage.

## Ignore improvements and existing index entries

Existing exclusions already cover active models, MovieLens, private reviews, environment files, logs, Python/Next caches and installed dependencies. Added root .aws/, *.pem, *.key, *.log and npm/yarn/pnpm debug-log patterns. The root .env.example remains explicitly allowed; it contains blank API key fields. Frontend .gitignore remains unchanged.

After adding *.log, these old tracked paths match ignore rules but are already absent from the worktree (their local copies remain under ignored logs/archive). `git add -u` below records their removals from the index along with old relocated paths; it does not delete archive files. There is no currently present tracked file requiring git rm --cached.

Tracked historical root logs requiring index removal:

- `backend-external-regression.log`
- `backend-human-evaluation-regression.log`
- `backend-live-resolution-regression.log`
- `backend-multi-intent-regression.log`
- `backend-pending-regression.log`
- `backend-presentation-regression.log`
- `backend-quality-regression.log`
- `backend-regression.log`
- `backend-unit-regression.log`
- `experimental-backend-tests.log`
- `experimental-pending-test.log`
- `external-embedding-tests.log`
- `human-evaluation-tests.log`
- `live-movie-regression.log`
- `multi-intent-tests.log`
- `pending-live-regression.log`
- `pending-regression.log`
- `quality-tests.log`
- `session-regression.log`
- `word-followup-backend.log`
- `word-followup-focused.log`

For a future currently-present tracked local-only file, `git rm --cached -- <exact-file-path>` removes tracking without deleting its local copy. That is a template, not a command to run for absent relocation paths. Do not use a recursive blanket index removal.

## Exact commands for the owner

Run from the project root in PowerShell. Review the complete relocation together with the lint/audit changes. No command here deletes local files or rewrites history.

```powershell
git status --short
# Stage tracked modifications and absent old paths (including archived root logs):
git add -u
# Stage only intended project folders and public root files:
git add -- .gitignore .env.example README.md requirements.txt requirements-chat.txt backend frontend docs evaluation scripts tests data/processed/movie_mapping.csv data/raw/movies.csv
# Review index, renames, whitespace, and ignored tracked paths:
git diff --cached --stat
git diff --cached --name-status
git diff --cached --check
git ls-files --cached --ignored --exclude-standard
# Expected: no output from the previous command. Confirm no local/private paths:
git diff --cached --name-only -- .env .aws data/raw/ml-32m data/models data/processed/tmdb_movies.csv evaluation/results/human logs
# Expected: no output. Then inspect the staged patch locally:
git diff --cached
# Commit only after your review:
git commit -m "Reorganize Movie AI project and finish frontend cleanup"
# Refresh remote state before publishing; stop if diverged:
git fetch origin
git rev-list --left-right --count HEAD...origin/main
# Expected behind count (second number) is 0; otherwise reconcile first.
git push --dry-run origin main
git push origin main
```

The prior local commit will be included by a main push as well; review it before publishing. Do not force-push. Authentication, branch rules and possible newly arrived remote commits are not verified until the owner runs the network checks. The dry run can validate push eligibility but is not a guarantee against a later remote race.

## Regression validation and remaining issues

- All 371 backend tests passed without skips. Backend imports and cwd-independent startup verified.
- ESLint: zero errors, zero warnings; TypeScript passed.
- All five browser suites passed desktop/mobile: manual search, card/provider appearance, delayed images and scrolling, Original/Experimental selection, session memory, multi-intent word-count follow-ups, blind review drafts/exports.
- Production Next build passed, all five application routes generated. Existing Google Fonts need network access at build time; initial sandbox font fetch failed, network-enabled retry passed without code changes.
- Structure verifier passed all 29 protected hashes, nine runtime-text modules, unchanged API contract/13 routes, 62,398 catalog rows, 384 embedding columns and 16,034 SVD rows.
- No paid OpenAI calls, training, embeddings regeneration, private-record changes, source deletion or Git history rewrite.
- Existing FastAPI/Starlette test-client deprecation warning remains; it is unrelated to frontend ESLint and all checks pass. No dependency migration attempted.

No code/test/size/identified-secret blocker remains. Required runtime assets stay local, build fonts require connectivity, and remote freshness/authentication/policy checks remain the owner publication step. Retirement of historical files is optional and awaits explicit approval.
