from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tarefa_3.domain.calculation import ElementCalculation
from app.tarefa_3.domain.model import TrainedPerceptronModel, TrainingPoint, TrainingStep
from app.tarefa_3.domain.pattern import MatrixPattern, PatternFactory
from app.tarefa_3.domain.prediction import PredictionResult


def build_matrix(fill: int = -1) -> list[list[int]]:
    return [[fill for _ in range(5)] for _ in range(5)]


def build_x_matrix() -> list[list[int]]:
    return [
        [1, -1, -1, -1, 1],
        [-1, 1, -1, 1, -1],
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1],
    ]


class MatrixPatternTests(unittest.TestCase):
    def test_post_init_normalizes_values_and_exposes_vector(self) -> None:
        pattern = MatrixPattern(matrix=build_x_matrix(), label="X", target=1, identifier="X_Principal")

        self.assertEqual(len(pattern.vector), 25)
        self.assertEqual(pattern.vector[0], 1)
        self.assertEqual(pattern.vector[-1], 1)

    def test_invalid_shape_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "5x5"):
            MatrixPattern(matrix=[[1, -1]], label="X", target=1, identifier="broken")

    def test_invalid_values_raise(self) -> None:
        matrix = build_matrix()
        matrix[0][0] = 0

        with self.assertRaisesRegex(ValueError, "apenas 1 e -1"):
            MatrixPattern(matrix=matrix, label="X", target=1, identifier="broken")

    def test_invalid_target_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "target"):
            MatrixPattern(matrix=build_matrix(), label="X", target=0, identifier="broken")  # type: ignore[arg-type]

    def test_to_dict_returns_serializable_payload(self) -> None:
        pattern = MatrixPattern(matrix=build_x_matrix(), label="X", target=1, identifier="X_Principal")

        payload = pattern.to_dict()

        self.assertEqual(payload["id"], "X_Principal")
        self.assertEqual(payload["label"], "X")
        self.assertEqual(payload["target"], 1)
        self.assertEqual(len(payload["vector"]), 25)


class PatternFactoryTests(unittest.TestCase):
    def test_create_builds_matrix_pattern(self) -> None:
        pattern = PatternFactory.create(build_x_matrix(), label="X", target=1, identifier="X_Principal")

        self.assertIsInstance(pattern, MatrixPattern)
        self.assertEqual(pattern.identifier, "X_Principal")


class TrainingModelTests(unittest.TestCase):
    def test_training_point_to_dict_includes_feature_columns(self) -> None:
        point = TrainingPoint(identifier="X_Principal", label="X", features=[1, -1, 1], target=1)

        payload = point.to_dict()

        self.assertEqual(payload["id"], "X_Principal")
        self.assertEqual(payload["x1"], 1)
        self.assertEqual(payload["x2"], -1)
        self.assertEqual(payload["x3"], 1)

    def test_training_step_to_dict_uses_api_field_names(self) -> None:
        step = TrainingStep(
            epoch=1,
            sample_id="X_Principal",
            target=1,
            features=[1, -1],
            weights_before=[0.0, 0.0],
            bias_before=1.0,
            u_before=1.0,
            y_before=1,
            updated=True,
            weights_after=[1.0, -1.0],
            bias_after=1.0,
        )

        payload = step.to_dict()

        self.assertEqual(payload["sampleId"], "X_Principal")
        self.assertEqual(payload["weightsBefore"], [0.0, 0.0])
        self.assertEqual(payload["weightsAfter"], [1.0, -1.0])

    def test_trained_model_to_dict_nests_points_and_steps(self) -> None:
        point = TrainingPoint(identifier="X_Principal", label="X", features=[1, -1], target=1)
        step = TrainingStep(
            epoch=1,
            sample_id="X_Principal",
            target=1,
            features=[1, -1],
            weights_before=[0.0, 0.0],
            bias_before=1.0,
            u_before=1.0,
            y_before=1,
            updated=True,
            weights_after=[1.0, -1.0],
            bias_after=1.0,
        )
        model = TrainedPerceptronModel(
            weights=[1.0, -1.0],
            bias=1.0,
            epochs=1,
            training_mode="hebb",
            training_points=[point],
            training_steps=[step],
        )

        payload = model.to_dict()

        self.assertEqual(payload["trainingMode"], "hebb")
        self.assertEqual(payload["trainingPoints"][0]["id"], "X_Principal")
        self.assertEqual(payload["trainingSteps"][0]["sampleId"], "X_Principal")


class PredictionAndCalculationTests(unittest.TestCase):
    def test_element_calculation_computes_product_and_dict(self) -> None:
        element = ElementCalculation(row=2, col=3, input_value=-1, weight=2.5)

        self.assertEqual(element.product, -2.5)
        self.assertEqual(element.to_dict()["product"], -2.5)

    def test_prediction_result_properties_and_dict(self) -> None:
        pattern = MatrixPattern(matrix=build_x_matrix(), label="X", target=1, identifier="X1")
        weights = [1.0] * 25
        contributions = [float(value) for value in pattern.vector]
        result = PredictionResult(
            pattern=pattern,
            weights=weights,
            bias=1.0,
            contributions=contributions,
            weighted_sum=sum(contributions),
            u=1.0 + sum(contributions),
            y_hat=1,
        )

        payload = result.to_dict()

        self.assertEqual(result.pred_label, "X")
        self.assertTrue(result.is_correct)
        self.assertEqual(len(result.weights_matrix), 5)
        self.assertEqual(len(result.contributions_matrix), 5)
        self.assertEqual(payload["predictedLabel"], "X")
        self.assertEqual(payload["isCorrect"], True)


if __name__ == "__main__":
    unittest.main()
