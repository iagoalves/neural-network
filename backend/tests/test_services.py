from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tarefa_3.domain.model import TrainingPoint
from app.tarefa_3.domain.pattern import MatrixPattern
from app.tarefa_3.services.classifier import PerceptronClassifier, VectorMathService
from app.tarefa_3.services.csv_exporter import CsvExporter
from app.tarefa_3.services.detailed_calculation import DetailedCalculationService
from app.tarefa_3.services.simple_trainer import HebbSimpleTrainer


def build_x_matrix() -> list[list[int]]:
    return [
        [1, -1, -1, -1, 1],
        [-1, 1, -1, 1, -1],
        [-1, -1, 1, -1, -1],
        [-1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1],
    ]


def build_t_matrix() -> list[list[int]]:
    return [
        [1, 1, 1, 1, 1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
        [-1, -1, 1, -1, -1],
    ]


def build_training_points() -> list[TrainingPoint]:
    x_pattern = MatrixPattern(matrix=build_x_matrix(), label="X", target=1, identifier="X_Principal")
    t_pattern = MatrixPattern(matrix=build_t_matrix(), label="T", target=-1, identifier="T_Principal")
    return [
        TrainingPoint(identifier=x_pattern.identifier, label=x_pattern.label, features=x_pattern.vector, target=x_pattern.target),
        TrainingPoint(identifier=t_pattern.identifier, label=t_pattern.label, features=t_pattern.vector, target=t_pattern.target),
    ]


class VectorMathServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = VectorMathService()

    def test_dot_returns_weighted_sum(self) -> None:
        self.assertEqual(self.service.dot([1, -1], [2.0, 3.0]), -1.0)

    def test_dot_rejects_vectors_with_different_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "mesmo tamanho"):
            self.service.dot([1], [1.0, 2.0])

    def test_multiply_returns_element_wise_products(self) -> None:
        self.assertEqual(self.service.multiply([1, -1], [2.0, 3.0]), [2.0, -3.0])

    def test_multiply_rejects_vectors_with_different_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "mesmo tamanho"):
            self.service.multiply([1], [1.0, 2.0])


class HebbSimpleTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training_points = build_training_points()
        self.trainer = HebbSimpleTrainer(initial_bias=2)

    def test_activate_respects_zero_threshold(self) -> None:
        self.assertEqual(self.trainer.activate(0), 1)
        self.assertEqual(self.trainer.activate(-0.1), -1)

    def test_weighted_sum_adds_bias(self) -> None:
        total = self.trainer.weighted_sum([1, -1], [2.0, 3.0], bias=1.0)

        self.assertEqual(total, 0.0)

    def test_train_requires_at_least_one_point(self) -> None:
        with self.assertRaisesRegex(ValueError, "pelo menos um ponto"):
            self.trainer.train([])

    def test_train_requires_same_feature_count_for_all_points(self) -> None:
        broken = self.training_points + [
            TrainingPoint(identifier="bad", label="X", features=[1, -1], target=1)
        ]

        with self.assertRaisesRegex(ValueError, "mesmo número de entradas"):
            self.trainer.train(broken)

    def test_train_runs_single_hebb_pass_and_keeps_bias_fixed(self) -> None:
        model = self.trainer.train(self.training_points)

        expected_weights = [
            x_point - t_point
            for x_point, t_point in zip(self.training_points[0].features, self.training_points[1].features)
        ]

        self.assertEqual(model.weights, expected_weights)
        self.assertEqual(model.bias, 2)
        self.assertEqual(model.epochs, 1)
        self.assertEqual(model.training_mode, "hebb")
        self.assertEqual(len(model.training_steps), 2)
        self.assertEqual(model.training_steps[0].sample_id, "X_Principal")
        self.assertEqual(model.training_steps[1].sample_id, "T_Principal")


class ClassifierAndDetailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = HebbSimpleTrainer(initial_bias=1).train(build_training_points())
        self.classifier = PerceptronClassifier(self.model)

    def test_classifier_activate_respects_zero_threshold(self) -> None:
        self.assertEqual(self.classifier.activate(0), 1)
        self.assertEqual(self.classifier.activate(-1), -1)

    def test_predict_returns_consistent_payload(self) -> None:
        pattern = MatrixPattern(matrix=build_x_matrix(), label="X", target=1, identifier="X_Test")

        result = self.classifier.predict(pattern)

        self.assertEqual(result.pattern.identifier, "X_Test")
        self.assertEqual(len(result.contributions), 25)
        self.assertEqual(result.weighted_sum, sum(result.contributions))
        self.assertEqual(result.u, self.model.bias + result.weighted_sum)
        self.assertEqual(result.y_hat, 1)

    def test_detailed_calculation_service_builds_matrix_positions(self) -> None:
        pattern = MatrixPattern(matrix=build_x_matrix(), label="X", target=1, identifier="X_Test")
        detail_service = DetailedCalculationService(self.classifier)

        details = detail_service.build_for_pattern(pattern)

        self.assertEqual(len(details), 25)
        self.assertEqual(details[0].row, 1)
        self.assertEqual(details[0].col, 1)
        self.assertEqual(details[-1].row, 5)
        self.assertEqual(details[-1].col, 5)


class CsvExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training_points = build_training_points()
        self.model = HebbSimpleTrainer(initial_bias=1).train(self.training_points)
        self.classifier = PerceptronClassifier(self.model)
        self.exporter = CsvExporter()
        self.x_pattern = MatrixPattern(matrix=build_x_matrix(), label="X", target=1, identifier="X_Principal")
        self.t_pattern = MatrixPattern(matrix=build_t_matrix(), label="T", target=-1, identifier="T_Principal")

    def test_patterns_to_csv_outputs_header_and_rows(self) -> None:
        content = self.exporter.patterns_to_csv([self.x_pattern, self.t_pattern])

        self.assertIn("id,classe,y,linha,c1,c2,c3,c4,c5", content)
        self.assertIn("X_Principal,X,1,1,1,-1,-1,-1,1", content)
        self.assertIn("T_Principal,T,-1,5,-1,-1,1,-1,-1", content)

    def test_training_to_csv_outputs_points_model_and_steps(self) -> None:
        content = self.exporter.training_to_csv(self.model)

        self.assertIn("secao,id,classe,x1", content)
        self.assertIn("modelo_final,pesos_treinados", content)
        self.assertIn("passo_hebb,1,X_Principal,1", content)
        self.assertIn("bias não é atualizado", content)

    def test_predictions_to_csv_outputs_classification_rows(self) -> None:
        results = [self.classifier.predict(self.x_pattern), self.classifier.predict(self.t_pattern)]

        content = self.exporter.predictions_to_csv(results)

        self.assertIn("id,esperado,predito,y_esperado,y_predito,soma_ponderada,bias_fixo,u,correto", content)
        self.assertIn("X_Principal,X,X,1,1", content)
        self.assertIn("T_Principal,T,T,-1,-1", content)


if __name__ == "__main__":
    unittest.main()
