from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.tarefa_2.domain.logic import LogicalFunction, TruthRow

TrainingMode = Literal["hebb"]


@dataclass(frozen=True)
class HebbTrainingStep:
    function_id: str
    row: int
    a: int
    b: int
    features: list[int]
    target: int
    weights_before: list[float]
    bias_before: float
    u_before: float
    y_before: int
    delta_weights: list[float]
    delta_bias: float
    weights_after: list[float]
    bias_after: float
    u_after: float
    y_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "functionId": self.function_id,
            "row": self.row,
            "a": self.a,
            "b": self.b,
            "features": self.features,
            "target": self.target,
            "weightsBefore": self.weights_before,
            "biasBefore": self.bias_before,
            "uBefore": self.u_before,
            "yBefore": self.y_before,
            "deltaWeights": self.delta_weights,
            "deltaBias": self.delta_bias,
            "weightsAfter": self.weights_after,
            "biasAfter": self.bias_after,
            "uAfter": self.u_after,
            "yAfter": self.y_after,
        }


@dataclass(frozen=True)
class HebbModel:
    weights: list[float]
    bias: float
    epochs: int
    training_mode: TrainingMode
    initial_weight: float
    initial_bias: float
    training_steps: list[HebbTrainingStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "epochs": self.epochs,
            "trainingMode": self.training_mode,
            "initialWeight": self.initial_weight,
            "initialBias": self.initial_bias,
            "trainingSteps": [step.to_dict() for step in self.training_steps],
        }


@dataclass(frozen=True)
class HebbPrediction:
    function: LogicalFunction
    row: TruthRow
    weights: list[float]
    bias: float
    contributions: list[float]
    weighted_sum: float
    u: float
    y_hat: int

    @property
    def predicted_output(self) -> int:
        return 1 if self.y_hat == 1 else 0

    @property
    def is_correct(self) -> bool:
        return self.y_hat == self.row.target

    def to_dict(self) -> dict[str, Any]:
        return {
            "functionId": self.function.identifier,
            "functionIndex": self.function.index,
            "row": self.row.to_dict(),
            "weights": self.weights,
            "bias": self.bias,
            "contributions": self.contributions,
            "weightedSum": self.weighted_sum,
            "u": self.u,
            "yHat": self.y_hat,
            "predictedOutput": self.predicted_output,
            "expectedOutput": self.row.output,
            "isCorrect": self.is_correct,
        }


@dataclass(frozen=True)
class FunctionTrainingResult:
    function: LogicalFunction
    model: HebbModel
    predictions: list[HebbPrediction]

    @property
    def correct_count(self) -> int:
        return sum(1 for prediction in self.predictions if prediction.is_correct)

    @property
    def accuracy(self) -> float:
        return self.correct_count / len(self.predictions) if self.predictions else 0.0

    @property
    def is_exact(self) -> bool:
        return self.correct_count == len(self.predictions)

    @property
    def classification(self) -> str:
        return "classificada_corretamente" if self.is_exact else "nao_classificada_perfeitamente"

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function.to_dict(),
            "model": self.model.to_dict(),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "correctCount": self.correct_count,
            "totalCount": len(self.predictions),
            "accuracy": self.accuracy,
            "isExact": self.is_exact,
            "classification": self.classification,
        }
