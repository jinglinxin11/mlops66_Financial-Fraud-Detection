"""Tests for the FastAPI application."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


class MockConfig:
    """Mock configuration for testing."""

    TRAIN_TRANSACTION = "train_transaction.csv"
    TRAIN_IDENTITY = "train_identity.csv"
    TEST_TRANSACTION = "test_transaction.csv"
    TEST_IDENTITY = "test_identity.csv"
    PREPROCESSOR_PATH = "preprocessor.pkl"
    MODEL_PATH = "tabnet_fraud_model"


@pytest.fixture
def mock_preprocessor():
    """Create a mock preprocessor."""
    preprocessor = MagicMock()
    preprocessor.transform.return_value = {
        "X_test": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        "transaction_ids": np.array([1000001, 1000002, 1000003]),
    }
    return preprocessor


@pytest.fixture
def mock_model():
    """Create a mock TabNet model."""
    model = MagicMock()
    # Returns probabilities for [not_fraud, fraud] for each sample
    model.predict_proba.return_value = np.array(
        [[0.9, 0.1], [0.3, 0.7], [0.5, 0.5]]  # Not fraud  # Fraud  # Borderline
    )
    return model


@pytest.fixture
def mock_trainer(mock_model):
    """Create a mock trainer that returns the mock model."""
    trainer = MagicMock()
    trainer.load.return_value = mock_model
    return trainer


@pytest.fixture
def client(mock_preprocessor, mock_trainer):
    """Create a test client with mocked dependencies."""
    with (
        patch("api.main.Config", return_value=MockConfig()),
        patch("api.main.FraudPreprocessor", return_value=mock_preprocessor),
        patch("api.main.TabNetTrainer", return_value=mock_trainer),
    ):
        # Import app after patching to ensure mocks are in place
        from api.main import app

        yield TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_root_returns_running_status(self):
        """Test that root endpoint returns running status."""
        # This test doesn't need mocks - it's a simple health check
        from api.main import app

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert response.json() == {"status": "running"}

    def test_root_is_get_method(self):
        """Test that root endpoint only accepts GET requests."""
        from api.main import app

        client = TestClient(app)

        # POST should not be allowed (405 Method Not Allowed)
        response = client.post("/")
        assert response.status_code == 405


class TestPredictTestEndpoint:
    """Tests for the predict_test endpoint."""

    def test_predict_test_default_limit(self, client, mock_preprocessor, mock_trainer):
        """Test predict_test with default limit."""
        response = client.post("/predict_test")

        assert response.status_code == 200
        data = response.json()

        assert "count" in data
        assert "predictions" in data
        assert data["count"] == 5  # Default limit

        # Verify preprocessor was called
        mock_preprocessor.load.assert_called_once()
        mock_preprocessor.transform.assert_called_once()

        # Verify model was loaded
        mock_trainer.load.assert_called_once()

    def test_predict_test_custom_limit(self, client):
        """Test predict_test with custom limit."""
        response = client.post("/predict_test?limit=2")

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_predict_test_response_structure(self, client):
        """Test that predictions have correct structure."""
        response = client.post("/predict_test?limit=3")

        assert response.status_code == 200
        data = response.json()

        for prediction in data["predictions"]:
            assert "TransactionID" in prediction
            assert "fraud_probability" in prediction
            assert "is_fraud" in prediction

            # Type checks
            assert isinstance(prediction["TransactionID"], int)
            assert isinstance(prediction["fraud_probability"], float)
            assert isinstance(prediction["is_fraud"], bool)

            # Value range checks
            assert 0.0 <= prediction["fraud_probability"] <= 1.0

    def test_predict_test_fraud_classification(self, client):
        """Test that fraud classification is correct based on probability threshold."""
        response = client.post("/predict_test?limit=3")

        assert response.status_code == 200
        predictions = response.json()["predictions"]

        # Based on our mock: [0.1, 0.7, 0.5] fraud probabilities
        # is_fraud should be True when probability >= 0.5
        assert predictions[0]["is_fraud"] is False  # 0.1 < 0.5
        assert predictions[1]["is_fraud"] is True  # 0.7 >= 0.5
        assert predictions[2]["is_fraud"] is True  # 0.5 >= 0.5


class TestAPIMetadata:
    """Tests for API metadata and documentation."""

    def test_openapi_schema_available(self):
        """Test that OpenAPI schema is available."""
        from api.main import app

        client = TestClient(app)
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        assert schema["info"]["title"] == "Fraud Detection API"
        assert schema["info"]["version"] == "1.0"

    def test_docs_endpoint_available(self):
        """Test that Swagger UI docs are available."""
        from api.main import app

        client = TestClient(app)
        response = client.get("/docs")

        assert response.status_code == 200
