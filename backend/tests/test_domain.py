from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tarefa_3.domain.calculation import ElementCalculation
from app.tarefa_3.domain.model import TrainingLog, TrainingPoint, TrainingStep, TrainedPerceptronModel
from app.tarefa_3.domain.pattern import MatrixPattern, PatternFactory
from app.tarefa_3.domain.prediction import PredictionResult


def build_x_matrix() -> list[list[int]]:
    return [
        [1, -1, -1, -1, 1],
        [-1, 1, -1, 1, -1],
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1],
    ]


def build_training_step() -> TrainingStep:
    return TrainingStep(
        epoch=2,
        sample_id="T_Principal",
        target=-1,
        features=[1, -1],
        weights_before=[0.001, 0.001],
        bias_before=1.0,
        u_before=1.002,
        y_before=1,
        error=-2,
        updated=True,
        delta_weights=[-2.0, 2.0],
        weights_after=[-1.999, 2.001],
        bias_after=1.0,
        weighted_sum_after=-4.0,
        u_after=-3.0,
        y_after=-1,
    )


def build_prediction_result() -> PredictionResult:
    pattern = MatrixPattern(matrix=build_x_matrix(), label="X", target=1, identifier="X_Principal")
    contributions = [float(value) for value in pattern.vector]
    return PredictionResult(
        pattern=pattern,
        weights=[1.0] * 25,
        bias=1.0,
        contributions=contributions,
        weighted_sum=sum(contributions),
        u=1.0 + sum(contributions),
        y_hat=1,
    )


class TestMatrixPattern:
    def test_vector_and_dict_are_exposed(self) -> None:
        pattern = MatrixPattern(matrix=build_x_matrix(), label="X", target=1, identifier="X_Principal")

        assert len(pattern.vector) == 25
        assert pattern.vector[0] == 1
        assert pattern.vector[-1] == 1
        assert pattern.to_dict()["id"] == "X_Principal"
        assert pattern.to_dict()["label"] == "X"

    def test_invalid_shape_raises(self) -> None:
        try:
            MatrixPattern(matrix=[[1, -1]], label="X", target=1, identifier="broken")
        except ValueError as exc:
            assert "5x5" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid shape")

    def test_invalid_values_raise(self) -> None:
        matrix = build_x_matrix()
        matrix[0][0] = 0

        try:
            MatrixPattern(matrix=matrix, label="X", target=1, identifier="broken")
        except ValueError as exc:
            assert "apenas 1 e -1" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid values")

    def test_invalid_target_raises(self) -> None:
        try:
            MatrixPattern(matrix=build_x_matrix(), label="X", target=0, identifier="broken")  # type: ignore[arg-type]
        except ValueError as exc:
            assert "target" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid target")


class TestPatternFactory:
    def test_factory_creates_pattern(self) -> None:
        pattern = PatternFactory.create(build_x_matrix(), label="X", target=1, identifier="X_Principal")

        assert isinstance(pattern, MatrixPattern)
        assert pattern.identifier == "X_Principal"
        assert pattern.vector[12] == 1


class TestTrainingModel:
    def test_training_point_to_dict_includes_feature_columns(self) -> None:
        point = TrainingPoint(identifier="X_Principal", label="X", features=[1, -1, 1], target=1)
        payload = point.to_dict()

        assert payload["id"] == "X_Principal"
        assert payload["x1"] == 1
        assert payload["x2"] == -1
        assert payload["x3"] == 1

    def test_training_step_to_dict_uses_current_api_fields(self) -> None:
        payload = build_training_step().to_dict()

        assert payload["sampleId"] == "T_Principal"
        assert payload["error"] == -2
        assert payload["deltaWeights"] == [-2.0, 2.0]
        assert payload["weightedSumAfter"] == -4.0
        assert payload["yAfter"] == -1

    def test_trained_model_to_dict_nests_points_steps_and_logs(self) -> None:
        point = TrainingPoint(identifier="X_Principal", label="X", features=[1, -1], target=1)
        step = build_training_step()
        log = TrainingLog(level="info", message="Início da época 1.")
        model = TrainedPerceptronModel(
            weights=[0.1, -0.2],
            bias=1.0,
            epochs=3,
            training_mode="error_correction",
            training_points=[point],
            training_steps=[step],
            initial_weight=0.001,
            max_epochs=10,
            logs=[log],
        )

        payload = model.to_dict()

        assert payload["trainingMode"] == "error_correction"
        assert payload["trainingPoints"][0]["id"] == "X_Principal"
        assert payload["trainingSteps"][0]["sampleId"] == "T_Principal"
        assert payload["initialWeight"] == 0.001
        assert payload["logs"] == [{"level": "info", "message": "Início da época 1."}]


class TestPredictionAndCalculation:
    def test_element_calculation_exposes_product(self) -> None:
        element = ElementCalculation(row=2, col=3, input_value=-1, weight=2.5)

        assert element.product == -2.5
        assert element.to_dict() == {
            "row": 2,
            "col": 3,
            "inputValue": -1,
            "weight": 2.5,
            "product": -2.5,
        }

    def test_prediction_result_properties_and_dict(self) -> None:
        result = build_prediction_result()
        payload = result.to_dict()

        assert result.pred_label == "X"
        assert result.is_correct is True
        assert len(result.weights_matrix) == 5
        assert len(result.contributions_matrix) == 5
        assert payload["predictedLabel"] == "X"
        assert payload["isCorrect"] is True
        assert payload["weightsMatrix"][0] == [1.0] * 5
