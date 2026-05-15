from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TrainingMode = Literal["error_correction"]


@dataclass(frozen=True)
class TrainingPoint:
    """Uma linha do treino supervisionado do perceptron.

    Cada matriz 5x5 é achatada para 25 entradas x1...x25.
    O alvo bipolar segue a regra:
    - X => y = 1
    - T => y = -1
    """

    identifier: str
    label: str
    features: list[int]
    target: int

    def to_dict(self) -> dict[str, Any]:
        row = {
            "id": self.identifier,
            "label": self.label,
            "target": self.target,
            "features": self.features,
        }
        row.update({f"x{index}": value for index, value in enumerate(self.features, start=1)})
        return row


@dataclass(frozen=True)
class TrainingStep:
    """Estado de uma amostra durante a regra de correção de erro."""

    epoch: int
    sample_id: str
    target: int
    features: list[int]
    weights_before: list[float]
    bias_before: float
    u_before: float
    y_before: int
    error: int
    updated: bool
    delta_weights: list[float]
    weights_after: list[float]
    bias_after: float
    weighted_sum_after: float
    u_after: float
    y_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "sampleId": self.sample_id,
            "target": self.target,
            "features": self.features,
            "weightsBefore": self.weights_before,
            "biasBefore": self.bias_before,
            "uBefore": self.u_before,
            "yBefore": self.y_before,
            "error": self.error,
            "updated": self.updated,
            "deltaWeights": self.delta_weights,
            "weightsAfter": self.weights_after,
            "biasAfter": self.bias_after,
            "weightedSumAfter": self.weighted_sum_after,
            "uAfter": self.u_after,
            "yAfter": self.y_after,
        }


@dataclass(frozen=True)
class TrainingLog:
    """Mensagem didática gerada durante o treinamento."""

    level: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "message": self.message}


@dataclass(frozen=True)
class TrainedPerceptronModel:
    """Modelo treinado com a regra de correção de erro do perceptron.

    - Os pesos começam em 0.001.
    - O bias é fixo nesta implementação.
    - A atualização só ocorre quando existe erro:
      erro = y - y_hat
      delta_wi = erro * xi
      wi <- wi + delta_wi
    """

    weights: list[float]
    bias: float
    epochs: int
    training_mode: TrainingMode
    training_points: list[TrainingPoint]
    training_steps: list[TrainingStep]
    initial_weight: float
    max_epochs: int
    logs: list[TrainingLog]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "epochs": self.epochs,
            "trainingMode": self.training_mode,
            "trainingPoints": [point.to_dict() for point in self.training_points],
            "trainingSteps": [step.to_dict() for step in self.training_steps],
            "initialWeight": self.initial_weight,
            "maxEpochs": self.max_epochs,
            "logs": [log.to_dict() for log in self.logs],
        }
