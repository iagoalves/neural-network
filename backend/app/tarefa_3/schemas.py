from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MatrixValue = Literal[1, -1]
TargetValue = Literal[1, -1]
PatternLabel = Literal["X", "T"]
TrainingMode = Literal["error_correction"]
Matrix5x5 = list[list[MatrixValue]]


class MatrixPatternSchema(BaseModel):
    id: str
    label: PatternLabel
    target: TargetValue
    matrix: Matrix5x5
    vector: list[MatrixValue]


class ElementCalculationSchema(BaseModel):
    row: int
    col: int
    inputValue: MatrixValue
    weight: float
    product: float


class TrainingPointSchema(BaseModel):
    id: str
    label: PatternLabel
    target: TargetValue
    features: list[MatrixValue]


class TrainingStepSchema(BaseModel):
    epoch: int
    sampleId: str
    target: TargetValue
    features: list[MatrixValue]
    weightsBefore: list[float]
    biasBefore: float
    uBefore: float
    yBefore: TargetValue
    error: int
    updated: bool
    deltaWeights: list[float]
    weightsAfter: list[float]
    biasAfter: float
    weightedSumAfter: float
    uAfter: float
    yAfter: TargetValue


class TrainedModelSchema(BaseModel):
    weights: list[float] = Field(description="Pesos treinados w1...w25.")
    bias: float
    epochs: int
    trainingMode: TrainingMode
    trainingPoints: list[TrainingPointSchema]
    trainingSteps: list[TrainingStepSchema]
    initialWeight: float
    maxEpochs: int
    logs: list[dict[str, str]]


class PredictionResultSchema(BaseModel):
    pattern: MatrixPatternSchema
    id: str
    expectedLabel: PatternLabel
    expectedTarget: TargetValue
    features: list[MatrixValue]
    weights: list[float]
    weightsMatrix: list[list[float]]
    bias: float
    contributions: list[float]
    contributionsMatrix: list[list[float]]
    weightedSum: float
    u: float
    yHat: TargetValue
    predictedLabel: PatternLabel
    isCorrect: bool
    details: list[ElementCalculationSchema] = Field(default_factory=list)


class SamplesResponseSchema(BaseModel):
    model: TrainedModelSchema
    samples: list[MatrixPatternSchema]
    predictions: list[PredictionResultSchema]


class TrainPerceptronRequestSchema(BaseModel):
    """Parâmetros controláveis do treino por correção de erro."""

    model_config = ConfigDict(extra="forbid")

    initialBias: int = Field(default=1, ge=-50, le=50)
    mode: TrainingMode = Field(default="error_correction")


class TrainPerceptronResponseSchema(SamplesResponseSchema):
    trainingCsv: str
    message: str


class PatternsResponseSchema(BaseModel):
    patterns: list[MatrixPatternSchema]


class LearningCardSchema(BaseModel):
    title: str
    text: str


class TheoryContentSchema(BaseModel):
    headline: str
    intro: str
    cards: list[LearningCardSchema]
    formula: str
    activation: str
    decisionLine: str


class ProgramContentSchema(BaseModel):
    headline: str
    summary: str


class LearningContentSchema(BaseModel):
    title: str
    summary: str
    theory: TheoryContentSchema
    program: ProgramContentSchema


class HealthResponseSchema(BaseModel):
    ok: bool
    runtime: str


class ManualPredictionRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matrix: list[list[int]]
    target: TargetValue = 1
    identifier: str = "Entrada_Manual"

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, value: list[list[int]]) -> list[list[int]]:
        if len(value) != 5 or any(len(row) != 5 for row in value):
            raise ValueError("A matriz deve possuir formato 5x5.")

        invalid_values = [
            item
            for row in value
            for item in row
            if item not in (-1, 1)
        ]

        if invalid_values:
            raise ValueError("A matriz deve conter apenas 1 e -1.")

        return value
