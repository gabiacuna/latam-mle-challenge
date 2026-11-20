## Part IV — Serving pipeline

`challenge/serving.py` chains five steps on a `DelayModel` with injected I/O: `step_load_model` → `step_load_data` → `step_preprocess` → `step_predict` → `step_write`. `main()` returns the number of predictions written and logs the row count and delayed share.

- **Data flow (gcp)**: the serving batch is read from `gs://<prefix>-input/serving/serving_input.csv` (`GCSCSVClient`, blob name `SERVING_BLOB` in `serving.py`, matching `objects.serving.path` in `config/base.yaml`). The model is restored from the latest artifact in `gs://<prefix>-artifacts` (`GCSArtifactStore`). Predictions go to `flights.predictions` in BigQuery.
- **Data flow (local)**: `data/serving_input.csv` → `./artifacts/predictions.csv`.
- **Write semantics**: every run reads the full batch and replaces the predictions table (`WRITE_TRUNCATE`). The table always mirrors the latest run, so re-runs do not duplicate rows. Change detection is out of scope.
- **Output**: columns `OPERA, TIPOVUELO, MES, prediction`, plus `model_run_id` and `predicted_at` (UTC), which match `config/schemas/predictions.json`. The three identifier columns and their types are kept stable for the Part V results endpoint filters. The metadata columns are written only when a run id is known, so the bare `write(predictions, identifiers)` call still works.
- **Model run id**: `ArtifactStore.load` records the resolved run id on the model (`DelayModel.run_id`), so serving reports which model produced each prediction.
- **Job**: `Dockerfile.serving` (same base as training, `CMD python -m challenge.serving`), Cloud Run job `challenge-serving` (2 vCPU, 2 GiB, 600 s, no retries) running as `challenge-sa` with the same env as training. Name set in `config/base.yaml` under `cloud_run.serving_job`.
- **IAM**: no new grants. `challenge-sa` already has `storage.objectAdmin` on the input and artifacts buckets and `bigquery.jobUser` at project level.
- **Errors** are not caught in `main()`; the `__main__` block logs them and exits 1, so the Cloud Run execution fails.
- **Run locally**: `make training-local && make serving-local`, then inspect `artifacts/predictions.csv`. The end-to-end test `tests/pipelines/test_serving_e2e.py` does the same in a temp directory.
- **Deploy**: `infra/bootstrap.sh` builds and deploys both `training` and `serving` images and jobs. Run it, then `gcloud run jobs execute challenge-serving --region <region> --wait`, and check with `bq query 'SELECT COUNT(*) FROM flights.predictions'`.
- **Test note**: `tests/pipelines/test_serving.py::test_main_gcp_branch_uses_gcs_csv_reader` reads the latest artifact from GCS, because `predict` loads the model when `step_load_model` is patched. It needs `GCP_BUCKET_ARTIFACTS` set, a trained artifact in that bucket, and credentials that can read it. With those set, `make pipelines-test` passes (9 tests).
- **Test fix**: the missing `FEATURES_COLS` import in `tests/pipelines/test_training.py` was fixed (Part III).

Detailed plan: [plan-task-4.md](plan-task-4.md).
