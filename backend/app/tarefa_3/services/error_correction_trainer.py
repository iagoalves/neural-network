from __future__ import annotations

from app.tarefa_3.domain.model import TrainingLog, TrainingPoint, TrainingStep, TrainedPerceptronModel


class PerceptronErrorCorrectionTrainer:
    """Treinador supervisionado com correção de erro.

    O algoritmo usa a arquitetura clássica do perceptron:
    u = b + soma(xi * wi)

    A correção ocorre somente quando a predição difere do alvo:
    erro = y - y_hat
    delta_wi = erro * xi
    wi <- wi + delta_wi

    Nesta tarefa, o bias é fixo para deixar claro que ele desloca a ativação,
    mas não participa da atualização dos pesos.
    """

    def __init__(self, initial_bias: float = 1, initial_weight: float = 0.001, max_epochs: int = 10) -> None:
        self.initial_bias = initial_bias
        self.initial_weight = initial_weight
        self.max_epochs = max_epochs

    @staticmethod
    def activate(u: float) -> int:
        return 1 if u >= 0 else -1

    @staticmethod
    def weighted_sum(features: list[int], weights: list[float], bias: float) -> float:
        return bias + sum(feature * weight for feature, weight in zip(features, weights))

    @staticmethod
    def pure_weighted_sum(features: list[int], weights: list[float]) -> float:
        return sum(feature * weight for feature, weight in zip(features, weights))

    def _validate_points(self, training_points: list[TrainingPoint]) -> int:
        if not training_points:
            raise ValueError("É necessário pelo menos um ponto de treino.")

        feature_count = len(training_points[0].features)
        if feature_count == 0:
            raise ValueError("Os pontos de treino devem possuir entradas.")

        for point in training_points:
            if len(point.features) != feature_count:
                raise ValueError("Todos os pontos de treino devem ter o mesmo número de entradas.")

        return feature_count

    def train(self, training_points: list[TrainingPoint]) -> TrainedPerceptronModel:
        feature_count = self._validate_points(training_points)
        weights = [self.initial_weight for _ in range(feature_count)]
        bias = self.initial_bias
        steps: list[TrainingStep] = []
        logs = [
            TrainingLog("info", f"Pesos inicializados com {self.initial_weight} em todas as 25 posições."),
            TrainingLog("info", f"Bias fixo configurado como {bias}."),
        ]

        epochs_run = 0
        for epoch in range(1, self.max_epochs + 1):
            epochs_run = epoch
            epoch_errors = 0
            logs.append(TrainingLog("info", f"Início da época {epoch}."))

            for point in training_points:
                weights_before = weights.copy()
                bias_before = bias
                u_before = self.weighted_sum(point.features, weights_before, bias_before)
                y_before = self.activate(u_before)
                error = point.target - y_before
                updated = error != 0
                delta_weights = [error * feature if updated else 0.0 for feature in point.features]

                if updated:
                    epoch_errors += 1
                    weights = [weight + delta for weight, delta in zip(weights_before, delta_weights)]
                    logs.append(
                        TrainingLog(
                            "warning",
                            f"{point.identifier}: esperado y={point.target}, obtido ŷ={y_before}; erro={error}. Pesos corrigidos.",
                        )
                    )
                else:
                    weights = weights_before.copy()
                    logs.append(
                        TrainingLog(
                            "success",
                            f"{point.identifier}: esperado y={point.target}, obtido ŷ={y_before}; sem correção.",
                        )
                    )

                weighted_sum_after = self.pure_weighted_sum(point.features, weights)
                u_after = bias + weighted_sum_after
                y_after = self.activate(u_after)

                steps.append(
                    TrainingStep(
                        epoch=epoch,
                        sample_id=point.identifier,
                        target=point.target,
                        features=point.features,
                        weights_before=weights_before,
                        bias_before=bias_before,
                        u_before=u_before,
                        y_before=y_before,
                        error=error,
                        updated=updated,
                        delta_weights=delta_weights,
                        weights_after=weights.copy(),
                        bias_after=bias,
                        weighted_sum_after=weighted_sum_after,
                        u_after=u_after,
                        y_after=y_after,
                    )
                )

            logs.append(TrainingLog("info", f"Fim da época {epoch}: {epoch_errors} erro(s)."))
            if epoch_errors == 0:
                logs.append(TrainingLog("success", f"Treino convergiu na época {epoch}."))
                break

        return TrainedPerceptronModel(
            weights=weights,
            bias=bias,
            epochs=epochs_run,
            training_mode="error_correction",
            training_points=training_points,
            training_steps=steps,
            initial_weight=self.initial_weight,
            max_epochs=self.max_epochs,
            logs=logs,
        )
