from __future__ import annotations

from app.tarefa_2.domain.logic import LogicalFunction, build_all_functions, build_logical_function
from app.tarefa_2.domain.model import FunctionTrainingResult, HebbModel, HebbPrediction, HebbTrainingStep


class HebbLogicTrainer:
    """Treina uma unidade linear para cada função lógica usando a Regra de Hebb.

    Representação adotada:
    - A=0/B=0 viram -1;
    - A=1/B=1 viram +1;
    - saída lógica 0 vira y=-1;
    - saída lógica 1 vira y=+1.

    Atualização:
    - ΔwA = y·A;
    - ΔwB = y·B;
    - Δb = y;
    - w <- w + Δw; b <- b + Δb.
    """

    def __init__(self, initial_bias: float = 0.0, initial_weight: float = 0.0) -> None:
        self.initial_bias = initial_bias
        self.initial_weight = initial_weight

    @staticmethod
    def activate(u: float) -> int:
        return 1 if u >= 0 else -1

    @staticmethod
    def weighted_sum(features: list[int], weights: list[float], bias: float) -> float:
        return bias + sum(feature * weight for feature, weight in zip(features, weights))

    def train_function(self, function: LogicalFunction) -> FunctionTrainingResult:
        weights = [self.initial_weight, self.initial_weight]
        bias = self.initial_bias
        steps: list[HebbTrainingStep] = []

        for row in function.truth_table:
            features = row.features
            weights_before = weights.copy()
            bias_before = bias
            u_before = self.weighted_sum(features, weights_before, bias_before)
            y_before = self.activate(u_before)

            delta_weights = [row.target * feature for feature in features]
            delta_bias = float(row.target)
            weights = [weight + delta for weight, delta in zip(weights_before, delta_weights)]
            bias = bias_before + delta_bias

            u_after = self.weighted_sum(features, weights, bias)
            y_after = self.activate(u_after)

            steps.append(
                HebbTrainingStep(
                    function_id=function.identifier,
                    row=row.row,
                    a=row.a,
                    b=row.b,
                    features=features,
                    target=row.target,
                    weights_before=weights_before,
                    bias_before=bias_before,
                    u_before=u_before,
                    y_before=y_before,
                    delta_weights=delta_weights,
                    delta_bias=delta_bias,
                    weights_after=weights.copy(),
                    bias_after=bias,
                    u_after=u_after,
                    y_after=y_after,
                )
            )

        model = HebbModel(
            weights=weights,
            bias=bias,
            epochs=1,
            training_mode="hebb",
            initial_weight=self.initial_weight,
            initial_bias=self.initial_bias,
            training_steps=steps,
        )
        predictions = [self.predict(function, model, row.row) for row in function.truth_table]
        return FunctionTrainingResult(function=function, model=model, predictions=predictions)

    def train_all(self) -> list[FunctionTrainingResult]:
        return [self.train_function(function) for function in build_all_functions()]

    def train_by_index(self, function_index: int) -> FunctionTrainingResult:
        return self.train_function(build_logical_function(function_index))

    def predict(self, function: LogicalFunction, model: HebbModel, row_number: int) -> HebbPrediction:
        row = next((item for item in function.truth_table if item.row == row_number), None)
        if row is None:
            raise ValueError("Linha da tabela verdade não encontrada.")

        contributions = [feature * weight for feature, weight in zip(row.features, model.weights)]
        weighted_sum = sum(contributions)
        u = model.bias + weighted_sum
        y_hat = self.activate(u)
        return HebbPrediction(
            function=function,
            row=row,
            weights=model.weights,
            bias=model.bias,
            contributions=contributions,
            weighted_sum=weighted_sum,
            u=u,
            y_hat=y_hat,
        )
