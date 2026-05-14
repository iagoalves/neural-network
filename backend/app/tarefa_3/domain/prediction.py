from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.tarefa_3.domain.pattern import MatrixPattern


@dataclass(frozen=True)
class PredictionResult:
    """Resultado produzido pelo perceptron clássico treinado."""

    pattern: MatrixPattern
    weights: list[float]
    bias: float
    contributions: list[float]
    weighted_sum: float
    u: float
    y_hat: int

    @property
    def pred_label(self) -> str:
        return "X" if self.y_hat == 1 else "T"

    @property
    def is_correct(self) -> bool:
        return self.y_hat == self.pattern.target

    @property
    def weights_matrix(self) -> list[list[float]]:
        return [self.weights[index * 5:(index + 1) * 5] for index in range(5)]

    @property
    def contributions_matrix(self) -> list[list[float]]:
        return [self.contributions[index * 5:(index + 1) * 5] for index in range(5)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern.to_dict(),
            "id": self.pattern.identifier,
            "expectedLabel": self.pattern.label,
            "expectedTarget": self.pattern.target,
            "features": self.pattern.vector,
            "weights": self.weights,
            "weightsMatrix": self.weights_matrix,
            "bias": self.bias,
            "contributions": self.contributions,
            "contributionsMatrix": self.contributions_matrix,
            "weightedSum": self.weighted_sum,
            "u": self.u,
            "yHat": self.y_hat,
            "predictedLabel": self.pred_label,
            "isCorrect": self.is_correct,
        }
