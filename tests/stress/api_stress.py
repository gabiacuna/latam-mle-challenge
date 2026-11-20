"""Stress tests for API using Locust."""

from locust import HttpUser, task


class StressUser(HttpUser):
    """Locust user for stress testing API endpoints."""

    @task
    def predict_results_default(self) -> None:
        """Fetch default page of prediction results."""
        self.client.get(
            "/pipeline/predict/results",
            params={"page": 1, "page_size": 10},
        )

    @task
    def predict_results_paged(self) -> None:
        """Fetch page 2 of prediction results."""
        self.client.get(
            "/pipeline/predict/results",
            params={"page": 2, "page_size": 5},
        )
