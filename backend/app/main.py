from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.tarefa_3.content import LEARNING_CONTENT
from app.tarefa_3.domain.model import TrainingPoint
from app.tarefa_3.domain.pattern import MatrixPattern
from app.tarefa_3.repositories.pattern_repository import PatternRepository
from app.tarefa_3.schemas import (
    HealthResponseSchema,
    LearningContentSchema,
    ManualPredictionRequestSchema,
    PatternsResponseSchema,
    PredictionResultSchema,
    SamplesResponseSchema,
    TrainPerceptronRequestSchema,
    TrainPerceptronResponseSchema,
)
from app.tarefa_3.services.classifier import PerceptronClassifier
from app.tarefa_3.services.csv_exporter import CsvExporter
from app.tarefa_3.services.detailed_calculation import DetailedCalculationService
from app.tarefa_3.services.error_correction_trainer import PerceptronErrorCorrectionTrainer


class Tarefa3ApplicationService:
    """Orquestra a Tarefa 3 sem misturar regra de negócio com HTTP."""

    def __init__(self) -> None:
        self.repository = PatternRepository()
        self.csv_exporter = CsvExporter()
        self.retrain(initial_bias=1)

    def _build_training_points(self) -> list[TrainingPoint]:
        return [
            TrainingPoint(
                identifier=pattern.identifier,
                label=pattern.label,
                features=pattern.vector,
                target=pattern.target,
            )
            for pattern in self.repository.get_training_patterns()
        ]

    def _build_samples(self) -> list[MatrixPattern]:
        """Amostras exibidas/verificadas: padrões principais + matrizes fixas de verificação salvas em CSV."""
        return self.repository.get_all_for_verification()

    def _payload_from_samples(self) -> dict:
        samples = self._build_samples()
        return {
            "model": self.model.to_dict(),
            "samples": [pattern.to_dict() for pattern in samples],
            "predictions": [self.prediction_payload(pattern) for pattern in samples],
        }

    def retrain(self, initial_bias: float) -> None:
        """Executa o treino por correção de erro com X_Principal e T_Principal."""
        training_points = self._build_training_points()
        trainer = PerceptronErrorCorrectionTrainer(initial_bias=initial_bias, initial_weight=0.001, max_epochs=10)
        self.model = trainer.train(training_points)
        self.classifier = PerceptronClassifier(self.model)
        self.detail_service = DetailedCalculationService(self.classifier)

    def train_payload(self, request: TrainPerceptronRequestSchema) -> dict:
        """Reexecuta o treino por correção de erro e devolve o CSV do passo a passo."""
        self.retrain(initial_bias=request.initialBias)
        payload = self._payload_from_samples()
        return {
            **payload,
            "trainingCsv": self.csv_exporter.training_to_csv(self.model),
            "message": "Treino por correção de erro concluído: pesos inicializados em 0.001, bias fixo e atualização somente quando há erro.",
        }

    def prediction_payload(self, pattern: MatrixPattern) -> dict:
        result = self.classifier.predict(pattern)
        details = [calculation.to_dict() for calculation in self.detail_service.build_for_pattern(pattern)]
        return {**result.to_dict(), "details": details}

    def samples_payload(self) -> dict:
        return self._payload_from_samples()

    def predict_manual(self, request: ManualPredictionRequestSchema) -> dict:
        pattern = MatrixPattern(
            matrix=request.matrix,
            label="X" if request.target == 1 else "T",
            target=request.target,
            identifier=request.identifier,
        )
        return self.prediction_payload(pattern)

    def patterns_csv(self) -> str:
        return self.csv_exporter.patterns_to_csv(self.repository.get_training_patterns())

    def samples_csv(self) -> str:
        return self.csv_exporter.patterns_to_csv(self.repository.get_all_for_verification())

    def training_csv(self) -> str:
        return self.csv_exporter.training_to_csv(self.model)

    def predictions_csv(self) -> str:
        results = [self.classifier.predict(pattern) for pattern in self._build_samples()]
        return self.csv_exporter.predictions_to_csv(results)


app = FastAPI(
    title="trabalho_3 - Perceptron X/T com correção de erro",
    version="7.0.0",
    description="API FastAPI + Pydantic para a Tarefa 3 com perceptron e correção de erro.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_service = Tarefa3ApplicationService()


@app.get("/api/health", response_model=HealthResponseSchema)
def health() -> dict:
    return {"ok": True, "runtime": "fastapi-pydantic-python-3.14"}


@app.get("/api/learning-content", response_model=LearningContentSchema)
def learning_content() -> dict:
    return LEARNING_CONTENT


@app.get("/api/patterns", response_model=PatternsResponseSchema)
def patterns() -> dict:
    return {"patterns": [pattern.to_dict() for pattern in app_service.repository.get_all()]}


@app.get("/api/samples", response_model=SamplesResponseSchema)
def samples() -> dict:
    return app_service.samples_payload()


@app.post("/api/train", response_model=TrainPerceptronResponseSchema)
def train(request: TrainPerceptronRequestSchema) -> dict:
    try:
        return app_service.train_payload(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/predict", response_model=PredictionResultSchema)
def predict(request: ManualPredictionRequestSchema) -> dict:
    try:
        return app_service.predict_manual(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/csv/patterns", response_class=PlainTextResponse)
def csv_patterns() -> PlainTextResponse:
    return PlainTextResponse(app_service.patterns_csv(), media_type="text/csv; charset=utf-8")


@app.get("/api/csv/training", response_class=PlainTextResponse)
def csv_training() -> PlainTextResponse:
    return PlainTextResponse(app_service.training_csv(), media_type="text/csv; charset=utf-8")


@app.get("/api/csv/samples", response_class=PlainTextResponse)
def csv_samples() -> PlainTextResponse:
    return PlainTextResponse(app_service.samples_csv(), media_type="text/csv; charset=utf-8")


@app.get("/api/csv/predictions", response_class=PlainTextResponse)
def csv_predictions() -> PlainTextResponse:
    return PlainTextResponse(app_service.predictions_csv(), media_type="text/csv; charset=utf-8")
# ─── Trabalho 2 · Regra de Hebb para funções lógicas ────────────────────────
from app.tarefa_2.content import LEARNING_CONTENT as TAREFA2_LEARNING_CONTENT
from app.tarefa_2.domain.logic import build_all_functions
from app.tarefa_2.schemas import (
    FunctionsResponseSchema as Tarefa2FunctionsResponseSchema,
    LearningContentSchema as Tarefa2LearningContentSchema,
    ManualHebbPredictionRequestSchema,
    ManualHebbPredictionResponseSchema,
    TrainHebbRequestSchema,
    TrainHebbResponseSchema,
)
from app.tarefa_2.services.csv_exporter import HebbLogicCsvExporter
from app.tarefa_2.services.hebb_trainer import HebbLogicTrainer


class Tarefa2ApplicationService:
    """Orquestra a Tarefa 2 sem misturar a regra de Hebb com HTTP."""

    def __init__(self) -> None:
        self.csv_exporter = HebbLogicCsvExporter()
        self.retrain(initial_bias=0, initial_weight=0)

    def _summary(self) -> dict:
        exact = [result.function.identifier for result in self.results if result.is_exact]
        imperfect = [result.function.identifier for result in self.results if not result.is_exact]
        return {
            "totalFunctions": len(self.results),
            "exactFunctions": len(exact),
            "imperfectFunctions": len(imperfect),
            "exactFunctionIds": exact,
            "imperfectFunctionIds": imperfect,
        }

    def retrain(self, initial_bias: float, initial_weight: float) -> None:
        self.trainer = HebbLogicTrainer(initial_bias=initial_bias, initial_weight=initial_weight)
        self.results = self.trainer.train_all()

    def train_payload(self, request: TrainHebbRequestSchema) -> dict:
        self.retrain(initial_bias=request.initialBias, initial_weight=request.initialWeight)
        return {
            "results": [result.to_dict() for result in self.results],
            "summary": self._summary(),
            "trainingCsv": self.csv_exporter.training_to_csv(self.results),
            "predictionsCsv": self.csv_exporter.predictions_to_csv(self.results),
            "message": "Treino hebbiano concluído para F0...F15. As funções XOR e XNOR tendem a ficar imperfeitas em uma unidade linear simples.",
        }

    def functions_payload(self) -> dict:
        return {"functions": [function.to_dict() for function in build_all_functions()]}

    def predict_manual(self, request: ManualHebbPredictionRequestSchema) -> dict:
        trainer = HebbLogicTrainer(initial_bias=request.initialBias, initial_weight=request.initialWeight)
        result = trainer.train_by_index(request.functionIndex)
        row_number = 1 + request.a * 2 + request.b
        prediction = next(prediction for prediction in result.predictions if prediction.row.row == row_number)
        return {"result": result.to_dict(), "prediction": prediction.to_dict()}

    def functions_csv(self) -> str:
        return self.csv_exporter.functions_to_csv(build_all_functions())

    def training_csv(self) -> str:
        return self.csv_exporter.training_to_csv(self.results)

    def predictions_csv(self) -> str:
        return self.csv_exporter.predictions_to_csv(self.results)


tarefa2_service = Tarefa2ApplicationService()


@app.get("/api/trabalho2/learning-content", response_model=Tarefa2LearningContentSchema)
def tarefa2_learning_content() -> dict:
    return TAREFA2_LEARNING_CONTENT


@app.get("/api/trabalho2/functions", response_model=Tarefa2FunctionsResponseSchema)
def tarefa2_functions() -> dict:
    return tarefa2_service.functions_payload()


@app.post("/api/trabalho2/train", response_model=TrainHebbResponseSchema)
def tarefa2_train(request: TrainHebbRequestSchema) -> dict:
    try:
        return tarefa2_service.train_payload(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/trabalho2/predict", response_model=ManualHebbPredictionResponseSchema)
def tarefa2_predict(request: ManualHebbPredictionRequestSchema) -> dict:
    try:
        return tarefa2_service.predict_manual(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/trabalho2/csv/functions", response_class=PlainTextResponse)
def tarefa2_csv_functions() -> PlainTextResponse:
    return PlainTextResponse(tarefa2_service.functions_csv(), media_type="text/csv; charset=utf-8")


@app.get("/api/trabalho2/csv/training", response_class=PlainTextResponse)
def tarefa2_csv_training() -> PlainTextResponse:
    return PlainTextResponse(tarefa2_service.training_csv(), media_type="text/csv; charset=utf-8")


@app.get("/api/trabalho2/csv/predictions", response_class=PlainTextResponse)
def tarefa2_csv_predictions() -> PlainTextResponse:
    return PlainTextResponse(tarefa2_service.predictions_csv(), media_type="text/csv; charset=utf-8")
