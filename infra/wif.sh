#!/usr/bin/env bash
# Set up Workload Identity Federation so GitHub Actions can deploy to one GCP project
# without service-account keys. Idempotent. Run once per project (staging, production).
#
# Usage:
#   PROJECT_ID=<project> BRANCH=<develop|main> [GITHUB_REPO=owner/name] \
#   [GH_ENVIRONMENT=<staging|production>] ./infra/wif.sh
#
# BRANCH restricts which branch may authenticate (develop -> staging, main -> production).
# If GH_ENVIRONMENT is set, the four GitHub Environment variables are created with `gh`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${ROOT}/config/base.yaml"

command -v gcloud >/dev/null || { echo "gcloud is required" >&2; exit 1; }
command -v yq >/dev/null || { echo "yq is required (https://github.com/mikefarah/yq)" >&2; exit 1; }

: "${PROJECT_ID:?Set PROJECT_ID to the target GCP project}"
: "${BRANCH:?Set BRANCH (develop for staging, main for production)}"
GITHUB_REPO="${GITHUB_REPO:-$(git -C "${ROOT}" remote get-url origin | sed -E 's#.*github.com[:/]##; s#\.git$##')}"
REGION="${REGION:-$(yq -r .default_region "${CFG}")}"
RUNTIME_SA="$(yq -r .service_account_id "${CFG}")@${PROJECT_ID}.iam.gserviceaccount.com"

POOL="github"
PROVIDER="github-actions"
DEPLOYER_ID="github-deployer"
DEPLOYER_SA="${DEPLOYER_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format 'value(projectNumber)')"

echo ">> Enabling APIs"
gcloud services enable iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
  --project "${PROJECT_ID}"

echo ">> Workload Identity pool ${POOL}"
gcloud iam workload-identity-pools describe "${POOL}" --location global --project "${PROJECT_ID}" \
  >/dev/null 2>&1 \
  || gcloud iam workload-identity-pools create "${POOL}" --location global \
    --project "${PROJECT_ID}" --display-name "GitHub Actions"

CONDITION="assertion.repository == '${GITHUB_REPO}' && assertion.ref == 'refs/heads/${BRANCH}'"
PROVIDER_ARGS=(
  --location global --project "${PROJECT_ID}" --workload-identity-pool "${POOL}"
  --issuer-uri "https://token.actions.githubusercontent.com"
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref"
  --attribute-condition "${CONDITION}"
)
echo ">> OIDC provider ${PROVIDER} (only ${GITHUB_REPO}@${BRANCH})"
if gcloud iam workload-identity-pools providers describe "${PROVIDER}" --location global \
  --project "${PROJECT_ID}" --workload-identity-pool "${POOL}" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers update-oidc "${PROVIDER}" "${PROVIDER_ARGS[@]}"
else
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER}" "${PROVIDER_ARGS[@]}"
fi

echo ">> Deployer service account ${DEPLOYER_SA}"
gcloud iam service-accounts describe "${DEPLOYER_SA}" --project "${PROJECT_ID}" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${DEPLOYER_ID}" --project "${PROJECT_ID}" \
    --display-name "GitHub Actions deployer"

echo ">> IAM for the deployer"
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${DEPLOYER_SA}" --role "${role}" --condition=None >/dev/null
done
# Needed to deploy Cloud Run revisions that run as the runtime service account.
gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_SA}" --project "${PROJECT_ID}" \
  --member "serviceAccount:${DEPLOYER_SA}" --role roles/iam.serviceAccountUser >/dev/null
# Let identities from this repo (via the pool) impersonate the deployer.
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_SA}" --project "${PROJECT_ID}" \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" \
  --role roles/iam.workloadIdentityUser >/dev/null

WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"

if [[ -n "${GH_ENVIRONMENT:-}" ]]; then
  command -v gh >/dev/null || { echo "gh is required for GH_ENVIRONMENT" >&2; exit 1; }
  echo ">> GitHub Environment '${GH_ENVIRONMENT}' variables"
  gh api -X PUT "repos/${GITHUB_REPO}/environments/${GH_ENVIRONMENT}" >/dev/null
  gh variable set GCP_PROJECT_ID --env "${GH_ENVIRONMENT}" --repo "${GITHUB_REPO}" --body "${PROJECT_ID}"
  gh variable set GCP_REGION --env "${GH_ENVIRONMENT}" --repo "${GITHUB_REPO}" --body "${REGION}"
  gh variable set WIF_PROVIDER --env "${GH_ENVIRONMENT}" --repo "${GITHUB_REPO}" --body "${WIF_PROVIDER}"
  gh variable set WIF_SERVICE_ACCOUNT --env "${GH_ENVIRONMENT}" --repo "${GITHUB_REPO}" --body "${DEPLOYER_SA}"
fi

cat <<OUT

Done. GitHub Environment variables:
GCP_PROJECT_ID=${PROJECT_ID}
GCP_REGION=${REGION}
WIF_PROVIDER=${WIF_PROVIDER}
WIF_SERVICE_ACCOUNT=${DEPLOYER_SA}
OUT
