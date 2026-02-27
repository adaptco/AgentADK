#!/usr/bin/env bash
# merge_receipt_pr268.sh
# Single-file, ready-to-run merge receipt generator for PR #268 merge-to-main.
# Output:
#   - merge-receipt.pr268.json        (full receipt, includes non-hashed timestamps)
#   - merge-receipt.pr268.hash.json   (hash-surface only, canonicalized)
#   - merge-receipt.pr268.sha256      (sha256 of canonical hash-surface)
#
# Determinism notes:
# - The receipt hash is computed ONLY over "hash_surface" (no timestamps, no local paths).
# - The full receipt contains created_at_utc, but it is excluded from hash surface.

set -euo pipefail

############################################
# Config (edit if base branch differs)
############################################
REPO_SLUG_DEFAULT="Q-Enterprises/core-orchestrator"
PR_NUMBER_DEFAULT="268"
BASE_BRANCH_DEFAULT="main"
HEAD_BRANCH_DEFAULT="codex/extend-openapi.yaml-with-handshake-endpoints-d2o961"

REPO_SLUG="${REPO_SLUG:-$REPO_SLUG_DEFAULT}"
PR_NUMBER="${PR_NUMBER:-$PR_NUMBER_DEFAULT}"
BASE_BRANCH="${BASE_BRANCH:-$BASE_BRANCH_DEFAULT}"
HEAD_BRANCH="${HEAD_BRANCH:-$HEAD_BRANCH_DEFAULT}"

# merge method: "merge" (merge commit), "squash", "rebase"
MERGE_METHOD="${MERGE_METHOD:-merge}"

############################################
# Helpers
############################################
die() { echo "ERR: $*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die "No sha256 tool found (need sha256sum or shasum)"
  fi
}

canonical_json() {
  # canonicalize stdin to stable JSON: sort keys + compact separators
  python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
sys.stdout.write(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
PY
}

json_escape() {
  # quote a string as JSON
  python - <<'PY'
import json,sys
s=sys.stdin.read()
sys.stdout.write(json.dumps(s))
PY
}

utc_now_iso() {
  # best-effort UTC timestamp (excluded from hash surface)
  date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
}

git_repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || die "Not inside a git repo"
}

git_is_clean() {
  test -z "$(git status --porcelain)"
}

############################################
# Preconditions
############################################
need git
need python

ROOT="$(git_repo_root)"
cd "$ROOT"

# optional: gh for PR-based merge + metadata
HAS_GH="0"
if command -v gh >/dev/null 2>&1; then HAS_GH="1"; fi

# ensure origin exists
git remote get-url origin >/dev/null 2>&1 || die "Missing git remote: origin"

# ensure we are in the intended repo (best-effort)
ORIGIN_URL="$(git remote get-url origin)"
# no hard-fail on mismatch because remotes can be SSH/HTTPS variants; record it

############################################
# Fetch + checkout base
############################################
git fetch origin --prune

git checkout "$BASE_BRANCH"
git pull --ff-only origin "$BASE_BRANCH"

git_is_clean || die "Working tree not clean. Commit/stash changes before merging."

############################################
# Capture pre-merge pointers
############################################
BASE_BEFORE_SHA="$(git rev-parse HEAD)"
HEAD_REMOTE_REF="origin/${HEAD_BRANCH}"

# ensure head exists remotely
git show-ref --verify --quiet "refs/remotes/${HEAD_REMOTE_REF}" \
  || die "Remote head branch not found: ${HEAD_REMOTE_REF} (did you fetch origin?)"

HEAD_SHA="$(git rev-parse "${HEAD_REMOTE_REF}")"

############################################
# Perform merge (merge commit supremacy)
############################################
# We intentionally use a merge commit (no-ff).
# If conflicts arise, script stops; resolve manually, then re-run from "Finalize receipt" section.
MERGE_COMMIT_SHA=""
MERGE_STATUS="unknown"

set +e
git merge "${HEAD_REMOTE_REF}" --no-ff -m "Merge PR #${PR_NUMBER}: handshake lifecycle endpoints" >/dev/null 2>&1
MERGE_RC=$?
set -e

if [ "$MERGE_RC" -ne 0 ]; then
  # detect conflict
  if git status --porcelain | grep -qE '^(UU|AA|DD|AU|UA) '; then
    MERGE_STATUS="conflict"
    die "Merge conflicts detected. Resolve conflicts, then run: git commit && re-run this script with SKIP_MERGE=1"
  else
    MERGE_STATUS="failed"
    die "Merge failed (rc=$MERGE_RC). Inspect git output and retry."
  fi
else
  MERGE_STATUS="merged"
  MERGE_COMMIT_SHA="$(git rev-parse HEAD)"
fi

############################################
# Push
############################################
read -p "Push merge commit to origin/${BASE_BRANCH}? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  git push origin "$BASE_BRANCH"
else
  echo "Push skipped. Run 'git push origin ${BASE_BRANCH}' manually."
fi

############################################
# Post-merge pointers
############################################
BASE_AFTER_SHA="$(git rev-parse HEAD)"

# Sanity: after == merge commit
if [ "$BASE_AFTER_SHA" != "$MERGE_COMMIT_SHA" ]; then
  die "Post-merge HEAD mismatch (expected merge commit)."
fi

############################################
# Toolchain binding
############################################
GIT_VERSION="$(git --version | sed -e 's/^git version //')"

GH_VERSION=""
GH_AUTH_USER=""
PR_URL=""
PR_BASE=""
PR_HEAD=""
PR_TITLE=""
PR_MERGEABLE=""
PR_MERGED_AT=""

if [ "$HAS_GH" = "1" ]; then
  GH_VERSION="$(gh --version 2>/dev/null | head -n1 | sed -e 's/^gh version //')"
  # best-effort identity
  GH_AUTH_USER="$(gh api user -q .login 2>/dev/null || true)"
  # best-effort PR metadata (if repo slug is accessible via gh)
  PR_URL="$(gh api "repos/${REPO_SLUG}/pulls/${PR_NUMBER}" -q .html_url 2>/dev/null || true)"
  PR_BASE="$(gh api "repos/${REPO_SLUG}/pulls/${PR_NUMBER}" -q .base.ref 2>/dev/null || true)"
  PR_HEAD="$(gh api "repos/${REPO_SLUG}/pulls/${PR_NUMBER}" -q .head.ref 2>/dev/null || true)"
  PR_TITLE="$(gh api "repos/${REPO_SLUG}/pulls/${PR_NUMBER}" -q .title 2>/dev/null || true)"
  PR_MERGEABLE="$(gh api "repos/${REPO_SLUG}/pulls/${PR_NUMBER}" -q .mergeable 2>/dev/null || true)"
  PR_MERGED_AT="$(gh api "repos/${REPO_SLUG}/pulls/${PR_NUMBER}" -q .merged_at 2>/dev/null || true)"
fi

############################################
# Commit metadata
############################################
MERGE_AUTHOR_NAME="$(git show -s --format='%an' "$MERGE_COMMIT_SHA")"
MERGE_AUTHOR_EMAIL="$(git show -s --format='%ae' "$MERGE_COMMIT_SHA")"
MERGE_AUTHOR_DATE="$(git show -s --format='%aI' "$MERGE_COMMIT_SHA")"
MERGE_COMMITTER_NAME="$(git show -s --format='%cn' "$MERGE_COMMIT_SHA")"
MERGE_COMMITTER_EMAIL="$(git show -s --format='%ce' "$MERGE_COMMIT_SHA")"
MERGE_COMMITTER_DATE="$(git show -s --format='%cI' "$MERGE_COMMIT_SHA")"
MERGE_SUBJECT="$(git show -s --format='%s' "$MERGE_COMMIT_SHA")"

############################################
# Build receipt objects
############################################
CREATED_AT_UTC="$(utc_now_iso)"

# hash_surface: ONLY deterministic fields
HASH_SURFACE_JSON="$(cat <<JSON
{
  "receipt_version":"1.0",
  "type":"git.merge_receipt",
  "repo":{
    "slug":"${REPO_SLUG}",
    "origin_url":$(printf "%s" "$ORIGIN_URL" | json_escape)
  },
  "pull_request":{
    "number":${PR_NUMBER},
    "url":$(printf "%s" "$PR_URL" | json_escape),
    "title":$(printf "%s" "$PR_TITLE" | json_escape),
    "base_ref":"${BASE_BRANCH}",
    "head_ref":"${HEAD_BRANCH}"
  },
  "pointers":{
    "base_before":"${BASE_BEFORE_SHA}",
    "head_sha":"${HEAD_SHA}",
    "merge_commit":"${MERGE_COMMIT_SHA}",
    "base_after":"${BASE_AFTER_SHA}"
  },
  "merge":{
    "method":"${MERGE_METHOD}",
    "no_ff":true,
    "status":"${MERGE_STATUS}"
  },
  "commit_metadata":{
    "subject":$(printf "%s" "$MERGE_SUBJECT" | json_escape),
    "author":{
      "name":$(printf "%s" "$MERGE_AUTHOR_NAME" | json_escape),
      "email":$(printf "%s" "$MERGE_AUTHOR_EMAIL" | json_escape),
      "date":$(printf "%s" "$MERGE_AUTHOR_DATE" | json_escape)
    },
    "committer":{
      "name":$(printf "%s" "$MERGE_COMMITTER_NAME" | json_escape),
      "email":$(printf "%s" "$MERGE_COMMITTER_EMAIL" | json_escape),
      "date":$(printf "%s" "$MERGE_COMMITTER_DATE" | json_escape)
    }
  },
  "toolchain":{
    "git_version":"${GIT_VERSION}",
    "gh_version":$(printf "%s" "$GH_VERSION" | json_escape)
  }
}
JSON
)"

# canonicalize hash surface and compute hash
HASH_SURFACE_CANON="$(printf "%s" "$HASH_SURFACE_JSON" | canonical_json)"
HASH_SURFACE_FILE="merge-receipt.pr${PR_NUMBER}.hash.json"
printf "%s" "$HASH_SURFACE_CANON" > "$HASH_SURFACE_FILE"

HASH_SHA256="$(sha256_file "$HASH_SURFACE_FILE")"
HASH_SHA_FILE="merge-receipt.pr${PR_NUMBER}.sha256"
printf "%s\n" "$HASH_SHA256" > "$HASH_SHA_FILE"

# full receipt (includes non-hashed metadata)
FULL_RECEIPT_FILE="merge-receipt.pr${PR_NUMBER}.json"
FULL_RECEIPT_JSON="$(cat <<JSON
{
  "created_at_utc":"${CREATED_AT_UTC}",
  "hash_surface_sha256":"${HASH_SHA256}",
  "hash_surface":${HASH_SURFACE_CANON},
  "non_hashed_metadata":{
    "notes":"created_at_utc is excluded from hash surface",
    "gh_context":{
      "auth_user":$(printf "%s" "$GH_AUTH_USER" | json_escape),
      "pr_base_ref":$(printf "%s" "$PR_BASE" | json_escape),
      "pr_head_ref":$(printf "%s" "$PR_HEAD" | json_escape),
      "pr_mergeable":$(printf "%s" "$PR_MERGEABLE" | json_escape),
      "pr_merged_at":$(printf "%s" "$PR_MERGED_AT" | json_escape)
    },
    "working_tree_clean":true
  }
}
JSON
)"
printf "%s" "$FULL_RECEIPT_JSON" | canonical_json > "$FULL_RECEIPT_FILE"

############################################
# Output summary
############################################
echo "OK"
echo "  Full receipt:      ${FULL_RECEIPT_FILE}"
echo "  Hash surface:      ${HASH_SURFACE_FILE}"
echo "  Hash (sha256):     ${HASH_SHA_FILE}"
echo "  Merge commit:      ${MERGE_COMMIT_SHA}"
echo "  Base before/after: ${BASE_BEFORE_SHA} -> ${BASE_AFTER_SHA}"
