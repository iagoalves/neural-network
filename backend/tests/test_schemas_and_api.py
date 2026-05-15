from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import (
    Tarefa3ApplicationService,
    app,
    csv_patterns,
    csv_predictions,
    csv_samples,
    csv_training,
    health,
    learning_content,
    patterns,
    predict,
    samples,
    train,
)
from app.tarefa_3.schemas import ManualPredictionRequestSchema, TrainPerceptronRequestSchema


def build_x_matrix() -> list[list[int]]:
    return [
        [1, -1, -1, -1, 1],
        [-1, 1, -1, 1, -1],
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1],
    ]


class TestSchemaValidation:
    def test_train_request_schema_uses_current_defaults(self) -> None:
        request = TrainPerceptronRequestSchema()

        assert request.initialBias == 1
        assert request.mode == "error_correction"

    def test_manual_prediction_schema_accepts_valid_matrix(self) -> None:
        request = ManualPredictionRequestSchema(matrix=build_x_matrix(), target=1, identifier="manual")

        assert request.identifier == "manual"
        assert request.target == 1

    def test_manual_prediction_schema_rejects_invalid_shape(self) -> None:
        try:
            ManualPredictionRequestSchema(matrix=[[1, -1]], target=1, identifier="broken")
        except ValueError as exc:
            assert "5x5" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid shape")

    def test_manual_prediction_schema_rejects_invalid_values(self) -> None:
        matrix = build_x_matrix()
        matrix[0][0] = 0

        try:
            ManualPredictionRequestSchema(matrix=matrix, target=1, identifier="broken")
        except ValueError as exc:
            assert "apenas 1 e -1" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid values")


class TestApplicationService:
    def setup_method(self) -> None:
        self.service = Tarefa3ApplicationService()

    def test_build_training_points_returns_principal_samples(self) -> None:
        points = self.service._build_training_points()

        assert len(points) == 2
        assert points[0].identifier == "X_Principal"
        assert points[1].identifier == "T_Principal"

    def test_build_samples_returns_verification_set(self) -> None:
        samples = self.service._build_samples()

        assert len(samples) == 10
        assert samples[0].identifier == "X_Principal"
        assert samples[-1].identifier == "T4"

    def test_payload_from_samples_contains_model_samples_and_predictions(self) -> None:
        payload = self.service._payload_from_samples()

        assert payload["model"]["trainingMode"] == "error_correction"
        assert len(payload["samples"]) == 10
        assert len(payload["predictions"]) == 10
        assert payload["predictions"][0]["id"] == payload["samples"][0]["id"]

    def test_retrain_rebuilds_model_classifier_and_details(self) -> None:
        self.service.retrain(initial_bias=3)

        assert self.service.model.bias == 3
        assert self.service.classifier.model.bias == 3
        assert len(self.service.detail_service.weights) == 25

    def test_train_payload_returns_current_message_and_csv(self) -> None:
        payload = self.service.train_payload(TrainPerceptronRequestSchema(initialBias=4))

        assert payload["model"]["bias"] == 4
        assert "trainingCsv" in payload
        assert "correção de erro" in payload["message"]
        assert "modelo_final,pesos_treinados" in payload["trainingCsv"]

    def test_prediction_payload_returns_details(self) -> None:
        payload = self.service.prediction_payload(self.service.repository.pattern_x)

        assert payload["id"] == "X_Principal"
        assert len(payload["details"]) == 25
        assert payload["predictedLabel"] in {"X", "T"}

    def test_predict_manual_builds_transient_pattern(self) -> None:
        payload = self.service.predict_manual(
            ManualPredictionRequestSchema(matrix=build_x_matrix(), target=1, identifier="manual")
        )

        assert payload["id"] == "manual"
        assert payload["expectedLabel"] == "X"
        assert len(payload["details"]) == 25

    def test_csv_helpers_return_expected_sections(self) -> None:
        assert "id,classe,y,linha" in self.service.patterns_csv()
        assert "passo_correcao_erro" in self.service.training_csv()
        assert "id,esperado,predito" in self.service.predictions_csv()
        assert "X1" in self.service.samples_csv()


class TestRouteFunctions:
    def test_health_function(self) -> None:
        payload = health()

        assert payload["ok"] is True
        assert payload["runtime"] == "fastapi-pydantic-python-3.14"

    def test_learning_content_function(self) -> None:
        payload = learning_content()

        assert "theory" in payload
        assert "correção de erro" in payload["summary"]

    def test_patterns_and_samples_functions(self) -> None:
        patterns_payload = patterns()
        samples_payload = samples()

        assert len(patterns_payload["patterns"]) == 2
        assert len(samples_payload["samples"]) == 10
        assert len(samples_payload["predictions"]) == 10

    def test_train_function(self) -> None:
        payload = train(TrainPerceptronRequestSchema(initialBias=2, mode="error_correction"))

        assert payload["model"]["bias"] == 2
        assert payload["model"]["trainingMode"] == "error_correction"

    def test_predict_function(self) -> None:
        payload = predict(ManualPredictionRequestSchema(matrix=build_x_matrix(), target=1, identifier="manual"))

        assert payload["id"] == "manual"
        assert payload["expectedLabel"] == "X"
        assert len(payload["details"]) == 25

    def test_csv_route_functions(self) -> None:
        for response in (csv_patterns(), csv_training(), csv_samples(), csv_predictions()):
            assert response.media_type == "text/csv; charset=utf-8"


class TestAppConfiguration:
    def test_fastapi_metadata_and_routes(self) -> None:
        paths = {route.path for route in app.routes}

        assert app.title == "trabalho_3 - Perceptron X/T com correção de erro"
        assert "/api/health" in paths
        assert "/api/learning-content" in paths
        assert "/api/patterns" in paths
        assert "/api/samples" in paths
        assert "/api/train" in paths
        assert "/api/predict" in paths
        assert "/api/csv/patterns" in paths
        assert "/api/csv/training" in paths
        assert "/api/csv/samples" in paths
        assert "/api/csv/predictions" in paths
