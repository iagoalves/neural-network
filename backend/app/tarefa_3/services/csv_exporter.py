from __future__ import annotations

import csv
import io

from app.tarefa_3.domain.model import TrainedPerceptronModel
from app.tarefa_3.domain.pattern import MatrixPattern
from app.tarefa_3.domain.prediction import PredictionResult


class CsvExporter:
    """Gera CSVs em memória para padrões, treino e predições."""

    def patterns_to_csv(self, patterns: list[MatrixPattern]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "classe", "y", "linha", "c1", "c2", "c3", "c4", "c5"])

        for pattern in patterns:
            for index, row in enumerate(pattern.matrix, start=1):
                writer.writerow([pattern.identifier, pattern.label, pattern.target, index, *row])

        return output.getvalue()

    def training_to_csv(self, model: TrainedPerceptronModel) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        feature_headers = [f"x{index}" for index in range(1, 26)]

        writer.writerow(["secao", "id", "classe", *feature_headers, "y"])

        for point in model.training_points:
            writer.writerow(["ponto_treino", point.identifier, point.label, *point.features, point.target])

        writer.writerow([])
        writer.writerow(["secao", "id", *[f"w{index}" for index in range(1, 26)], "bias_fixo", "passagens"])
        writer.writerow(["modelo_final", "pesos_treinados", *model.weights, model.bias, model.epochs])

        writer.writerow([])
        writer.writerow([
            "secao",
            "passo",
            "amostra",
            "target",
            "u_antes",
            "y_antes",
            "bias_fixo",
            "observacao",
        ])

        for index, step in enumerate(model.training_steps, start=1):
            writer.writerow([
                "passo_hebb",
                index,
                step.sample_id,
                step.target,
                step.u_before,
                step.y_before,
                step.bias_before,
                "bias não é atualizado; entra direto em u = b + Σ(xi·wi)",
            ])

        return output.getvalue()

    def predictions_to_csv(self, predictions: list[PredictionResult]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "esperado", "predito", "y_esperado", "y_predito", "soma_ponderada", "bias_fixo", "u", "correto"])

        for result in predictions:
            writer.writerow([
                result.pattern.identifier,
                result.pattern.label,
                result.pred_label,
                result.pattern.target,
                result.y_hat,
                result.weighted_sum,
                result.bias,
                result.u,
                result.is_correct,
            ])

        return output.getvalue()
