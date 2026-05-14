from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TrainingMode = Literal["hebb"]


@dataclass(frozen=True)
class TrainingPoint:
    """Uma linha do treino Hebb.

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
    """Estado de uma amostra durante a aplicação da Regra de Hebb."""

    epoch: int
    sample_id: str
    target: int
    features: list[int]
    weights_before: list[float]
    bias_before: float
    u_before: float
    y_before: int
    updated: bool
    weights_after: list[float]
    bias_after: float

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
            "updated": self.updated,
            "weightsAfter": self.weights_after,
            "biasAfter": self.bias_after,
        }


@dataclass(frozen=True)
class TrainedPerceptronModel:
    """Modelo gerado por Hebb simples.

    Não há taxa de aprendizagem configurável nem repetição de épocas.
    O treinamento aplica uma única passagem por X_Principal e T_Principal.
    O bias é mantido fixo e somado diretamente durante a ativação.
    """

    weights: list[float]
    bias: float
    epochs: int
    training_mode: TrainingMode
    training_points: list[TrainingPoint]
    training_steps: list[TrainingStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "epochs": self.epochs,
            "trainingMode": self.training_mode,
            "trainingPoints": [point.to_dict() for point in self.training_points],
            "trainingSteps": [step.to_dict() for step in self.training_steps],
        }
