from __future__ import annotations

from app.tarefa_3.domain.calculation import ElementCalculation
from app.tarefa_3.domain.pattern import MatrixPattern
from app.tarefa_3.services.classifier import PerceptronClassifier


class DetailedCalculationService:
    """Detalha as 25 multiplicações xi*wi usadas pelo perceptron."""

    def __init__(self, classifier: PerceptronClassifier) -> None:
        self.weights = classifier.model.weights

    def build_for_pattern(self, pattern: MatrixPattern) -> list[ElementCalculation]:
        calculations: list[ElementCalculation] = []

        for index, (input_value, weight) in enumerate(zip(pattern.vector, self.weights)):
            row_index = index // 5
            col_index = index % 5
            calculations.append(
                ElementCalculation(
                    row=row_index + 1,
                    col=col_index + 1,
                    input_value=int(input_value),
                    weight=float(weight),
                )
            )

        return calculations
