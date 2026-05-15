from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tarefa_3.domain.model import TrainingPoint
from app.tarefa_3.domain.pattern import MatrixPattern
from app.tarefa_3.repositories.pattern_repository import PatternRepository
from app.tarefa_3.services.classifier import PerceptronClassifier, VectorMathService
from app.tarefa_3.services.csv_exporter import CsvExporter
from app.tarefa_3.services.detailed_calculation import DetailedCalculationService
from app.tarefa_3.services.error_correction_trainer import PerceptronErrorCorrectionTrainer


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


class TestVectorMathService:
    def setup_method(self) -> None:
        self.service = VectorMathService()

    def test_dot_returns_weighted_sum(self) -> None:
        assert self.service.dot([1, -1], [2.0, 3.0]) == -1.0

    def test_dot_rejects_vectors_with_different_sizes(self) -> None:
        try:
            self.service.dot([1], [1.0, 2.0])
        except ValueError as exc:
            assert "mesmo tamanho" in str(exc)
        else:
            raise AssertionError("Expected ValueError for mismatched vectors")

    def test_multiply_returns_element_wise_products(self) -> None:
        assert self.service.multiply([1, -1], [2.0, 3.0]) == [2.0, -3.0]


class TestErrorCorrectionTrainer:
    def setup_method(self) -> None:
        self.training_points = build_training_points()
        self.trainer = PerceptronErrorCorrectionTrainer(initial_bias=1, initial_weight=0.001, max_epochs=10)

    def test_activate_respects_zero_threshold(self) -> None:
        assert self.trainer.activate(0) == 1
        assert self.trainer.activate(-0.1) == -1

    def test_weighted_sum_helpers(self) -> None:
        assert self.trainer.weighted_sum([1, -1], [2.0, 3.0], bias=1.0) == 0.0
        assert self.trainer.pure_weighted_sum([1, -1], [2.0, 3.0]) == -1.0

    def test_train_requires_at_least_one_point(self) -> None:
        try:
            self.trainer.train([])
        except ValueError as exc:
            assert "pelo menos um ponto" in str(exc)
        else:
            raise AssertionError("Expected ValueError for empty training set")

    def test_train_requires_same_feature_count_for_all_points(self) -> None:
        broken = self.training_points + [TrainingPoint(identifier="bad", label="X", features=[1, -1], target=1)]

        try:
            self.trainer.train(broken)
        except ValueError as exc:
            assert "mesmo número de entradas" in str(exc)
        else:
            raise AssertionError("Expected ValueError for inconsistent feature count")

    def test_train_runs_error_correction_until_convergence(self) -> None:
        model = self.trainer.train(self.training_points)

        assert model.bias == 1
        assert model.epochs == 3
        assert model.training_mode == "error_correction"
        assert len(model.training_steps) == 6
        assert len(model.logs) == 15
        assert model.training_steps[0].sample_id == "X_Principal"
        assert model.training_steps[1].sample_id == "T_Principal"
        assert model.training_steps[1].updated is True
        assert model.training_steps[1].error == -2
        assert model.training_steps[2].updated is True
        assert model.training_steps[-1].updated is False
        assert model.logs[-1].message == "Treino convergiu na época 3."


class TestClassifierAndDetails:
    def setup_method(self) -> None:
        repository = PatternRepository()
        points = [
            TrainingPoint(identifier=pattern.identifier, label=pattern.label, features=pattern.vector, target=pattern.target)
            for pattern in repository.get_training_patterns()
        ]
        self.model = PerceptronErrorCorrectionTrainer().train(points)
        self.classifier = PerceptronClassifier(self.model)
        self.x_pattern = repository.pattern_x
        self.t_pattern = repository.pattern_t

    def test_classifier_predict_returns_consistent_payload(self) -> None:
        result = self.classifier.predict(self.x_pattern)

        assert result.pattern.identifier == "X_Principal"
        assert len(result.contributions) == 25
        assert result.weighted_sum == sum(result.contributions)
        assert result.u == self.model.bias + result.weighted_sum
        assert result.y_hat == 1
        assert result.is_correct is True

    def test_classifier_classifies_t_pattern_as_t(self) -> None:
        result = self.classifier.predict(self.t_pattern)

        assert result.y_hat == -1
        assert result.pred_label == "T"
        assert result.is_correct is True

    def test_detailed_calculation_service_builds_5x5_positions(self) -> None:
        details = DetailedCalculationService(self.classifier).build_for_pattern(self.x_pattern)

        assert len(details) == 25
        assert details[0].row == 1
        assert details[0].col == 1
        assert details[-1].row == 5
        assert details[-1].col == 5


class TestCsvExporter:
    def setup_method(self) -> None:
        repository = PatternRepository()
        points = [
            TrainingPoint(identifier=pattern.identifier, label=pattern.label, features=pattern.vector, target=pattern.target)
            for pattern in repository.get_training_patterns()
        ]
        self.model = PerceptronErrorCorrectionTrainer().train(points)
        self.classifier = PerceptronClassifier(self.model)
        self.exporter = CsvExporter()
        self.x_pattern = repository.pattern_x
        self.t_pattern = repository.pattern_t

    def test_patterns_to_csv_outputs_header_and_rows(self) -> None:
        content = self.exporter.patterns_to_csv([self.x_pattern, self.t_pattern])

        assert "id,classe,y,linha,c1,c2,c3,c4,c5" in content
        assert "X_Principal,X,1,1,1,-1,-1,-1,1" in content
        assert "T_Principal,T,-1,5,-1,-1,1,-1,-1" in content

    def test_training_to_csv_outputs_current_sections(self) -> None:
        content = self.exporter.training_to_csv(self.model)

        assert "secao,id,classe,x1" in content
        assert "modelo_final,pesos_treinados" in content
        assert "passo_correcao_erro,1,T_Principal,-1" in content
        assert "log_treino,success,Treino convergiu na época 3." in content

    def test_predictions_to_csv_outputs_classification_rows(self) -> None:
        results = [self.classifier.predict(self.x_pattern), self.classifier.predict(self.t_pattern)]
        content = self.exporter.predictions_to_csv(results)

        assert "id,esperado,predito,y_esperado,y_predito,soma_ponderada,bias_fixo,u,correto" in content
        assert "X_Principal,X,X,1,1" in content
        assert "T_Principal,T,T,-1,-1" in content
