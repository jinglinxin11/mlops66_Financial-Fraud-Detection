"""Load testing for the Fraud Detection API using Locust.

Usage:
    # Start the API first (locally or via Docker)
    # Then run locust:
    locust -f locustfile.py --host=http://localhost:8000

    # Or run headless:
    locust -f locustfile.py --host=http://localhost:8000 --headless -u 10 -r 2 -t 60s

Options:
    -u, --users: Number of concurrent users
    -r, --spawn-rate: Users spawned per second
    -t, --run-time: Test duration (e.g., 60s, 5m)
"""

from locust import HttpUser, between, task


class FraudAPIUser(HttpUser):
    """Simulated user for load testing the Fraud Detection API."""

    # Wait between 1-3 seconds between tasks
    wait_time = between(1, 3)

    @task(10)
    def health_check(self):
        """Test the health endpoint (most frequent)."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                json_data = response.json()
                if json_data.get("status") == "running":
                    response.success()
                else:
                    response.failure("Unexpected response format")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def predict_small(self):
        """Test prediction endpoint with small limit."""
        with self.client.post("/predict_test?limit=5", catch_response=True) as response:
            if response.status_code == 200:
                json_data = response.json()
                if "predictions" in json_data and json_data.get("count") == 5:
                    response.success()
                else:
                    response.failure("Unexpected response format")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def predict_large(self):
        """Test prediction endpoint with larger limit (less frequent)."""
        with self.client.post("/predict_test?limit=20", catch_response=True) as response:
            if response.status_code == 200:
                json_data = response.json()
                if "predictions" in json_data:
                    response.success()
                else:
                    response.failure("Unexpected response format")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def get_docs(self):
        """Test the OpenAPI docs endpoint."""
        with self.client.get("/openapi.json", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
