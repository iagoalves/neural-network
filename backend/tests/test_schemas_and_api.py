from __future__ import annotations

import sys
import unittest
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


def build_matrix(fill: int = -1) -> list[list[int]]:
    return [[fill for _ in range(5)] for _ in range(5)]


class SchemaValidationTests(unittest.TestCase):
    def test_train_request_schema_uses_defaults(self) -> None:
        request = TrainPerceptronRequestSchema()

        self.assertEqual(request.initialBias, 1)
        self.assertEqual(request.mode, "hebb")

    def test_manual_prediction_schema_accepts_valid_matrix(self) -> None:
        request = ManualPredictionRequestSchema(matrix=build_matrix(), target=1, identifier="manual")

        self.assertEqual(request.identifier, "manual")
        self.assertEqual(request.target, 1)

    def test_manual_prediction_schema_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "5x5"):
            ManualPredictionRequestSchema(matrix=[[1, -1]], target=1, identifier="broken")

    def test_manual_prediction_schema_rejects_invalid_values(self) -> None:
        matrix = build_matrix()
        matrix[0][0] = 0

        with self.assertRaisesRegex(ValueError, "apenas 1 e -1"):
            ManualPredictionRequestSchema(matrix=matrix, target=1, identifier="broken")


class ApplicationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Tarefa3ApplicationService()

    def test_build_training_points_returns_principal_samples(self) -> None:
        points = self.service._build_training_points()

        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].identifier, "X_Principal")
        self.assertEqual(points[1].identifier, "T_Principal")

    def test_build_samples_returns_verification_set(self) -> None:
        samples_payload = self.service._build_samples()

        self.assertGreaterEqual(len(samples_payload), 10)

    def test_payload_from_samples_contains_model_samples_and_predictions(self) -> None:
        payload = self.service._payload_from_samples()

        self.assertIn("model", payload)
        self.assertIn("samples", payload)
        self.assertIn("predictions", payload)
        self.assertEqual(len(payload["samples"]), len(payload["predictions"]))

    def test_retrain_rebuilds_model_classifier_and_details(self) -> None:
        self.service.retrain(initial_bias=3)

        self.assertEqual(self.service.model.bias, 3)
        self.assertEqual(self.service.classifier.model.bias, 3)
        self.assertEqual(len(self.service.detail_service.weights), 25)

    def test_train_payload_retrains_and_returns_csv_and_message(self) -> None:
        payload = self.service.train_payload(TrainPerceptronRequestSchema(initialBias=4))

        self.assertEqual(payload["model"]["bias"], 4)
        self.assertIn("trainingCsv", payload)
        self.assertIn("Treino Hebb simples concluído", payload["message"])

    def test_prediction_payload_returns_details(self) -> None:
        pattern = self.service.repository.pattern_x

        payload = self.service.prediction_payload(pattern)

        self.assertEqual(payload["id"], "X_Principal")
        self.assertEqual(len(payload["details"]), 25)

    def test_samples_payload_matches_sample_count(self) -> None:
        payload = self.service.samples_payload()

        self.assertEqual(len(payload["samples"]), len(payload["predictions"]))

    def test_predict_manual_builds_transient_pattern(self) -> None:
        payload = self.service.predict_manual(ManualPredictionRequestSchema(matrix=build_matrix(), target=1, identifier="manual"))

        self.assertEqual(payload["id"], "manual")
        self.assertEqual(payload["expectedLabel"], "X")

    def test_csv_helpers_return_expected_sections(self) -> None:
        self.assertIn("id,classe,y,linha", self.service.patterns_csv())
        self.assertIn("modelo_final,pesos_treinados", self.service.training_csv())
        self.assertIn("id,esperado,predito", self.service.predictions_csv())
        self.assertIn("X1", self.service.samples_csv())


class RouteFunctionTests(unittest.TestCase):
    def test_health_function(self) -> None:
        self.assertEqual(health()["ok"], True)

    def test_learning_content_function(self) -> None:
        self.assertIn("theory", learning_content())

    def test_patterns_function(self) -> None:
        self.assertEqual(len(patterns()["patterns"]), 2)

    def test_samples_function(self) -> None:
        payload = samples()

        self.assertEqual(len(payload["samples"]), len(payload["predictions"]))

    def test_train_function(self) -> None:
        payload = train(TrainPerceptronRequestSchema(initialBias=2))

        self.assertEqual(payload["model"]["bias"], 2)

    def test_predict_function(self) -> None:
        payload = predict(ManualPredictionRequestSchema(matrix=build_matrix(), target=1, identifier="manual"))

        self.assertEqual(payload["id"], "manual")

    def test_csv_route_functions(self) -> None:
        self.assertIn("text/csv", csv_patterns().media_type)
        self.assertIn("text/csv", csv_training().media_type)
        self.assertIn("text/csv", csv_samples().media_type)
        self.assertIn("text/csv", csv_predictions().media_type)


class AppConfigurationTests(unittest.TestCase):
    def test_fastapi_app_metadata_and_routes(self) -> None:
        paths = {route.path for route in app.routes}

        self.assertEqual(app.title, "trabalho_3 - Perceptron X/T com Hebb simples")
        self.assertIn("/api/health", paths)
        self.assertIn("/api/learning-content", paths)
        self.assertIn("/api/patterns", paths)
        self.assertIn("/api/samples", paths)
        self.assertIn("/api/train", paths)
        self.assertIn("/api/predict", paths)
        self.assertIn("/api/csv/patterns", paths)
        self.assertIn("/api/csv/training", paths)
        self.assertIn("/api/csv/samples", paths)
        self.assertIn("/api/csv/predictions", paths)


if __name__ == "__main__":
    unittest.main()
