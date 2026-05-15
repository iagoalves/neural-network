from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TrainingMode = Literal["hebb"]


class TruthRowSchema(BaseModel):
    row: int
    a: int
    b: int
    aBipolar: int
    bBipolar: int
    output: int
    target: int


class LogicalFunctionSchema(BaseModel):
    index: int
    id: str
    name: str
    expression: str
    bitPattern: str
    truthTable: list[TruthRowSchema]
    targets: list[int]


class HebbTrainingStepSchema(BaseModel):
    functionId: str
    row: int
    a: int
    b: int
    features: list[int]
    target: int
    weightsBefore: list[float]
    biasBefore: float
    uBefore: float
    yBefore: int
    deltaWeights: list[float]
    deltaBias: float
    weightsAfter: list[float]
    biasAfter: float
    uAfter: float
    yAfter: int


class HebbModelSchema(BaseModel):
    weights: list[float]
    bias: float
    epochs: int
    trainingMode: TrainingMode
    initialWeight: float
    initialBias: float
    trainingSteps: list[HebbTrainingStepSchema]


class HebbPredictionSchema(BaseModel):
    functionId: str
    functionIndex: int
    row: TruthRowSchema
    weights: list[float]
    bias: float
    contributions: list[float]
    weightedSum: float
    u: float
    yHat: int
    predictedOutput: int
    expectedOutput: int
    isCorrect: bool


class FunctionTrainingResultSchema(BaseModel):
    function: LogicalFunctionSchema
    model: HebbModelSchema
    predictions: list[HebbPredictionSchema]
    correctCount: int
    totalCount: int
    accuracy: float
    isExact: bool
    classification: str


class HebbSummarySchema(BaseModel):
    totalFunctions: int
    exactFunctions: int
    imperfectFunctions: int
    exactFunctionIds: list[str]
    imperfectFunctionIds: list[str]


class TrainHebbRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initialBias: float = Field(default=0, ge=-10, le=10)
    initialWeight: float = Field(default=0, ge=-10, le=10)
    mode: TrainingMode = Field(default="hebb")


class TrainHebbResponseSchema(BaseModel):
    results: list[FunctionTrainingResultSchema]
    summary: HebbSummarySchema
    trainingCsv: str
    predictionsCsv: str
    message: str


class FunctionsResponseSchema(BaseModel):
    functions: list[LogicalFunctionSchema]


class ManualHebbPredictionRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functionIndex: int = Field(ge=0, le=15)
    a: Literal[0, 1]
    b: Literal[0, 1]
    initialBias: float = Field(default=0, ge=-10, le=10)
    initialWeight: float = Field(default=0, ge=-10, le=10)


class ManualHebbPredictionResponseSchema(BaseModel):
    result: FunctionTrainingResultSchema
    prediction: HebbPredictionSchema


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
