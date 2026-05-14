from __future__ import annotations

from app.tarefa_3.domain.model import TrainedPerceptronModel
from app.tarefa_3.domain.pattern import MatrixPattern
from app.tarefa_3.domain.prediction import PredictionResult


class VectorMathService:
    """Operações vetoriais usadas no exercício."""

    @staticmethod
    def dot(left: list[int], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Os vetores devem ter o mesmo tamanho.")
        return sum(a * b for a, b in zip(left, right))

    @staticmethod
    def multiply(left: list[int], right: list[float]) -> list[float]:
        if len(left) != len(right):
            raise ValueError("Os vetores devem ter o mesmo tamanho.")
        return [a * b for a, b in zip(left, right)]


class PerceptronClassifier:
    """Classificador X/T usando o perceptron clássico com 25 entradas.

    Fluxo correto do exercício:
    1. A matriz 5x5 vira um vetor x com 25 entradas: x1, x2, ..., x25.
    2. O CSV de treino contém X com y=1 e T com y=-1.
    3. O treino Hebb aprende w1, w2, ..., w25 e o bias a partir de amostras rotuladas.
    4. A classificação usa u = b + soma(xi*wi).
    5. A função de ativação retorna y=1 para X e y=-1 para T.
    """

    def __init__(self, model: TrainedPerceptronModel) -> None:
        self.model = model
        self.math = VectorMathService()

    @staticmethod
    def activate(u: float) -> int:
        return 1 if u >= 0 else -1

    def predict(self, pattern: MatrixPattern) -> PredictionResult:
        features = pattern.vector
        contributions = self.math.multiply(features, self.model.weights)
        weighted_sum = sum(contributions)
        u = self.model.bias + weighted_sum
        y_hat = self.activate(u)

        return PredictionResult(
            pattern=pattern,
            weights=self.model.weights,
            bias=self.model.bias,
            contributions=contributions,
            weighted_sum=weighted_sum,
            u=u,
            y_hat=y_hat,
        )
