#!/usr/bin/env bash
# Provision the challenge's GCP data infrastructure with gcloud/bq. Idempotent.
# Usage: [PROJECT_ID=my-project] [REGION=us-central1] [NAME_PREFIX=...] ./infra/bootstrap.sh
# PROJECT_ID defaults to project_id in config/base.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${ROOT}/config/base.yaml"
SCHEMAS="${ROOT}/config/schemas"

command -v yq >/dev/null || { echo "yq is required (https://github.com/mikefarah/yq)" >&2; exit 1; }
command -v gcloud >/dev/null || { echo "gcloud is required" >&2; exit 1; }
command -v bq >/dev/null || { echo "bq is required (ships with gcloud)" >&2; exit 1; }

cfg() { yq -r "$1" "${CFG}"; }

PROJECT_ID="${PROJECT_ID:-$(cfg '.project_id // ""')}"
: "${PROJECT_ID:?Set project_id in config/base.yaml or export PROJECT_ID}"

REGION="${REGION:-$(cfg .default_region)}"
NAME_PREFIX="${NAME_PREFIX:-${PROJECT_ID}}"
SA_ID="$(cfg .service_account_id)"
SA_EMAIL="${SA_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET_INPUT="${NAME_PREFIX}-$(cfg .bucket_suffixes.input)"
BUCKET_ARTIFACTS="${NAME_PREFIX}-$(cfg .bucket_suffixes.artifacts)"
DATASET="$(cfg .bigquery.dataset)"
T_RAW="$(cfg .bigquery.tables.raw)"
T_SERVING="$(cfg .bigquery.tables.serving)"
T_PRED="$(cfg .bigquery.tables.predictions)"
OBJ_TRAIN="$(cfg .objects.training.path)"
OBJ_SERVE="$(cfg .objects.serving.path)"
LOCAL_TRAIN="${ROOT}/$(cfg .objects.training.local)"
LOCAL_SERVE="${ROOT}/$(cfg .objects.serving.local)"
REPO="$(cfg .artifact_registry_repo)"
LABELS="$(yq -r '.labels | to_entries | map(.key + "=" + .value) | join(",")' "${CFG}")"
DATASET_LABEL_ARGS=()
while IFS= read -r kv; do DATASET_LABEL_ARGS+=(--label "${kv}"); done \
  < <(yq -r '.labels | to_entries | .[] | .key + ":" + .value' "${CFG}")

echo ">> Enabling APIs"
gcloud services enable storage.googleapis.com bigquery.googleapis.com run.googleapis.com \
  artifactregistry.googleapis.com iam.googleapis.com cloudbuild.googleapis.com --project "${PROJECT_ID}"

echo ">> Service account ${SA_EMAIL}"
gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${SA_ID}" --project "${PROJECT_ID}" \
    --display-name "Challenge service account"

echo ">> Buckets"
for b in "${BUCKET_INPUT}" "${BUCKET_ARTIFACTS}"; do
  gcloud storage buckets describe "gs://${b}" --project "${PROJECT_ID}" >/dev/null 2>&1 \
    || gcloud storage buckets create "gs://${b}" --project "${PROJECT_ID}" --location "${REGION}" \
      --uniform-bucket-level-access --labels "${LABELS}"
done

echo ">> BigQuery dataset and tables"
bq --project_id "${PROJECT_ID}" show --dataset "${DATASET}" >/dev/null 2>&1 \
  || bq --project_id "${PROJECT_ID}" mk --dataset --location "${REGION}" \
    "${DATASET_LABEL_ARGS[@]}" "${PROJECT_ID}:${DATASET}"
for pair in "${T_RAW}:raw_flights" "${T_SERVING}:serving_input" "${T_PRED}:predictions"; do
  table="${pair%%:*}"; schema="${pair##*:}"
  bq --project_id "${PROJECT_ID}" show "${DATASET}.${table}" >/dev/null 2>&1 \
    || bq --project_id "${PROJECT_ID}" mk --table "${PROJECT_ID}:${DATASET}.${table}" \
      "${SCHEMAS}/${schema}.json"
done

echo ">> IAM"
for b in "${BUCKET_INPUT}" "${BUCKET_ARTIFACTS}"; do
  gcloud storage buckets add-iam-policy-binding "gs://${b}" \
    --member "serviceAccount:${SA_EMAIL}" --role roles/storage.objectAdmin >/dev/null
done
for role in roles/bigquery.jobUser roles/run.developer roles/run.invoker roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SA_EMAIL}" --role "${role}" --condition=None >/dev/null
done
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" --project "${PROJECT_ID}" \
  --member "serviceAccount:${SA_EMAIL}" --role roles/iam.serviceAccountUser >/dev/null
# Dataset-level dataEditor (read-modify-write of the dataset ACL, skipped if already present).
ACL="$(mktemp)"; trap 'rm -f "${ACL}"' EXIT
bq --project_id "${PROJECT_ID}" show --format=prettyjson "${PROJECT_ID}:${DATASET}" > "${ACL}"
if ! grep -q "\"userByEmail\": \"${SA_EMAIL}\"" "${ACL}"; then
  python3 - "${ACL}" "${SA_EMAIL}" <<'PY'
import json, sys
path, email = sys.argv[1:]
d = json.load(open(path))
d.setdefault("access", []).append({"role": "WRITER", "userByEmail": email})
json.dump(d, open(path, "w"))
PY
  bq --project_id "${PROJECT_ID}" update --source "${ACL}" "${PROJECT_ID}:${DATASET}" >/dev/null
fi

echo ">> Uploading CSVs"
gcloud storage cp "${LOCAL_TRAIN}" "gs://${BUCKET_INPUT}/${OBJ_TRAIN}"
gcloud storage cp "${LOCAL_SERVE}" "gs://${BUCKET_INPUT}/${OBJ_SERVE}"

echo ">> Loading tables (WRITE_TRUNCATE)"
# Tables already exist with their schema, so the CSV header is skipped and columns map by
# position: no schema file and no column_name_character_map are needed for 'Fecha-I', 'AÑO'.
bq --project_id "${PROJECT_ID}" --location "${REGION}" load --source_format=CSV \
  --skip_leading_rows=1 --replace \
  "${DATASET}.${T_RAW}" "gs://${BUCKET_INPUT}/${OBJ_TRAIN}"
bq --project_id "${PROJECT_ID}" --location "${REGION}" load --source_format=CSV \
  --skip_leading_rows=1 --replace \
  "${DATASET}.${T_SERVING}" "gs://${BUCKET_INPUT}/${OBJ_SERVE}"

echo ">> Cloud Build permissions for ${SA_EMAIL}"
for role in roles/storage.objectViewer roles/artifactregistry.writer roles/logging.logWriter roles/cloudbuild.builds.builder; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SA_EMAIL}" --role "${role}" --condition=None >/dev/null
done

echo ">> Artifact Registry"
gcloud artifacts repositories describe "${REPO}" --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${REPO}" --repository-format docker \
    --location "${REGION}" --project "${PROJECT_ID}" --labels "${LABELS}"

# Both jobs share one runtime env; they differ only in image and job name.
RUN_ENV="DEPLOYMENT_MODE=gcp,GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},GCP_BUCKET_INPUT=${BUCKET_INPUT},GCP_BUCKET_ARTIFACTS=${BUCKET_ARTIFACTS},BQ_DATASET=${DATASET},BQ_RAW_TABLE=${T_RAW},BQ_SERVING_TABLE=${T_SERVING},BQ_PREDICTIONS_TABLE=${T_PRED}"

for svc in training serving; do
  IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${svc}:latest"
  JOB="$(cfg ".cloud_run.${svc}_job")"

  echo ">> Building ${svc} image"
  BUILD_CFG="$(mktemp)"
  cat > "${BUILD_CFG}" <<YAML
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["build", "-f", "Dockerfile.${svc}", "-t", "${IMAGE}", "."]
images: ["${IMAGE}"]
options:
  logging: CLOUD_LOGGING_ONLY
YAML
  gcloud builds submit "${ROOT}" --config "${BUILD_CFG}" --project "${PROJECT_ID}" \
    --service-account "projects/${PROJECT_ID}/serviceAccounts/${SA_EMAIL}"
  rm -f "${BUILD_CFG}"

  echo ">> Cloud Run job ${JOB}"
  # 'deploy' creates or updates, so re-runs are idempotent.
  gcloud run jobs deploy "${JOB}" --image "${IMAGE}" --region "${REGION}" \
    --project "${PROJECT_ID}" --service-account "${SA_EMAIL}" \
    --cpu "$(cfg .cloud_run.cpu)" --memory "$(cfg .cloud_run.memory)" \
    --task-timeout "$(cfg .cloud_run.timeout)" --max-retries 0 --labels "${LABELS}" \
    --set-env-vars "${RUN_ENV}"
done

cat <<OUT

Done. Export for the pipelines:
export GCP_PROJECT_ID=${PROJECT_ID}
export GCP_REGION=${REGION}
export GCP_BUCKET_INPUT=${BUCKET_INPUT}
export GCP_BUCKET_ARTIFACTS=${BUCKET_ARTIFACTS}
export BQ_DATASET=${DATASET}
export BQ_RAW_TABLE=${T_RAW}
export BQ_SERVING_TABLE=${T_SERVING}
export BQ_PREDICTIONS_TABLE=${T_PRED}
OUT
