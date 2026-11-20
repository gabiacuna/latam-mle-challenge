#!/usr/bin/env bash
# Apply branch protection to main and develop. Idempotent (PUT replaces the rule).
# Requires: gh (authenticated with admin rights on the repo).
# Usage: [REPO=owner/name] ./infra/branch-protection.sh
set -euo pipefail

command -v gh >/dev/null || { echo "gh is required (https://cli.github.com)" >&2; exit 1; }

REPO="${REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
CHECKS='["lint","test"]'

protect() {
  local branch="$1" approvals="$2"
  echo ">> Protecting ${REPO}@${branch} (approvals: ${approvals})"
  gh api -X PUT "repos/${REPO}/branches/${branch}/protection" --input - <<JSON
{
  "required_status_checks": { "strict": true, "contexts": ${CHECKS} },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": ${approvals},
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON
}

protect main 1
protect develop 0

# Challenge rule: never delete development branches.
gh api -X PATCH "repos/${REPO}" -F delete_branch_on_merge=false >/dev/null
echo "Done."
