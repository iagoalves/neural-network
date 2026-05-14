from __future__ import annotations

from app.tarefa_3.domain.model import TrainedPerceptronModel, TrainingPoint, TrainingStep


class HebbSimpleTrainer:
    """Treinador da Regra de Hebb simples.

    Regras desta versão:
    - usa somente X_Principal e T_Principal;
    - executa uma única passagem pelo conjunto de treino;
    - não usa taxa de aprendizagem configurável;
    - não verifica erro antes de ajustar;
    - aplica diretamente nos pesos:
      wi <- wi + y * xi
    - o bias é fixo nesta implementação e entra direto na ativação.
    """

    def __init__(self, initial_bias: float = 1) -> None:
        self.initial_bias = initial_bias

    @staticmethod
    def activate(u: float) -> int:
        return 1 if u >= 0 else -1

    @staticmethod
    def weighted_sum(features: list[int], weights: list[float], bias: float) -> float:
        return bias + sum(feature * weight for feature, weight in zip(features, weights))

    def train(self, training_points: list[TrainingPoint]) -> TrainedPerceptronModel:
        if not training_points:
            raise ValueError("É necessário pelo menos um ponto de treino.")

        feature_count = len(training_points[0].features)
        weights = [0.0 for _ in range(feature_count)]
        bias = self.initial_bias
        steps: list[TrainingStep] = []

        for point in training_points:
            if len(point.features) != feature_count:
                raise ValueError("Todos os pontos de treino devem ter o mesmo número de entradas.")

            weights_before = weights.copy()
            bias_before = bias
            u_before = self.weighted_sum(point.features, weights_before, bias_before)
            y_before = self.activate(u_before)

            weights = [
                weight + point.target * feature
                for weight, feature in zip(weights_before, point.features)
            ]
            bias = bias_before

            steps.append(
                TrainingStep(
                    epoch=1,
                    sample_id=point.identifier,
                    target=point.target,
                    features=point.features,
                    weights_before=weights_before,
                    bias_before=bias_before,
                    u_before=u_before,
                    y_before=y_before,
                    updated=True,
                    weights_after=weights.copy(),
                    bias_after=bias,
                )
            )

        return TrainedPerceptronModel(
            weights=weights,
            bias=bias,
            epochs=1,
            training_mode="hebb",
            training_points=training_points,
            training_steps=steps,
        )
