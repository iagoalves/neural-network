from __future__ import annotations

import csv
import io

from app.tarefa_2.domain.logic import LogicalFunction
from app.tarefa_2.domain.model import FunctionTrainingResult


class HebbLogicCsvExporter:
    def functions_to_csv(self, functions: list[LogicalFunction]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["funcao", "nome", "expressao", "padrao_saida", "linha", "A", "B", "A_bipolar", "B_bipolar", "saida", "y"])
        for function in functions:
            for row in function.truth_table:
                writer.writerow([
                    function.identifier,
                    function.name,
                    function.expression,
                    function.bit_pattern,
                    row.row,
                    row.a,
                    row.b,
                    row.a_bipolar,
                    row.b_bipolar,
                    row.output,
                    row.target,
                ])
        return output.getvalue()

    def training_to_csv(self, results: list[FunctionTrainingResult]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["funcao", "nome", "linha", "A", "B", "y", "wA_antes", "wB_antes", "b_antes", "u_antes", "delta_wA", "delta_wB", "delta_b", "wA_depois", "wB_depois", "b_depois", "u_depois"])
        for result in results:
            for step in result.model.training_steps:
                writer.writerow([
                    result.function.identifier,
                    result.function.name,
                    step.row,
                    step.a,
                    step.b,
                    step.target,
                    step.weights_before[0],
                    step.weights_before[1],
                    step.bias_before,
                    step.u_before,
                    step.delta_weights[0],
                    step.delta_weights[1],
                    step.delta_bias,
                    step.weights_after[0],
                    step.weights_after[1],
                    step.bias_after,
                    step.u_after,
                ])
        return output.getvalue()

    def predictions_to_csv(self, results: list[FunctionTrainingResult]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["funcao", "nome", "expressao", "classificacao", "acertos", "total", "acuracia", "linha", "A", "B", "saida_esperada", "saida_prevista", "y_esperado", "y_previsto", "wA", "wB", "bias", "u", "correto"])
        for result in results:
            for prediction in result.predictions:
                writer.writerow([
                    result.function.identifier,
                    result.function.name,
                    result.function.expression,
                    result.classification,
                    result.correct_count,
                    len(result.predictions),
                    result.accuracy,
                    prediction.row.row,
                    prediction.row.a,
                    prediction.row.b,
                    prediction.row.output,
                    prediction.predicted_output,
                    prediction.row.target,
                    prediction.y_hat,
                    prediction.weights[0],
                    prediction.weights[1],
                    prediction.bias,
                    prediction.u,
                    prediction.is_correct,
                ])
        return output.getvalue()
