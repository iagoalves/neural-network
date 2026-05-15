export type TrainingMode = 'hebb';
export type BipolarValue = 1 | -1;
export type LogicalValue = 0 | 1;

export interface TruthRow {
  row: number;
  a: LogicalValue;
  b: LogicalValue;
  aBipolar: BipolarValue;
  bBipolar: BipolarValue;
  output: LogicalValue;
  target: BipolarValue;
}

export interface LogicalFunction {
  index: number;
  id: string;
  name: string;
  expression: string;
  bitPattern: string;
  truthTable: TruthRow[];
  targets: BipolarValue[];
}

export interface HebbTrainingStep {
  functionId: string;
  row: number;
  a: LogicalValue;
  b: LogicalValue;
  features: BipolarValue[];
  target: BipolarValue;
  weightsBefore: number[];
  biasBefore: number;
  uBefore: number;
  yBefore: BipolarValue;
  deltaWeights: number[];
  deltaBias: number;
  weightsAfter: number[];
  biasAfter: number;
  uAfter: number;
  yAfter: BipolarValue;
}

export interface HebbModel {
  weights: number[];
  bias: number;
  epochs: number;
  trainingMode: TrainingMode;
  initialWeight: number;
  initialBias: number;
  trainingSteps: HebbTrainingStep[];
}

export interface HebbPrediction {
  functionId: string;
  functionIndex: number;
  row: TruthRow;
  weights: number[];
  bias: number;
  contributions: number[];
  weightedSum: number;
  u: number;
  yHat: BipolarValue;
  predictedOutput: LogicalValue;
  expectedOutput: LogicalValue;
  isCorrect: boolean;
}

export interface FunctionTrainingResult {
  function: LogicalFunction;
  model: HebbModel;
  predictions: HebbPrediction[];
  correctCount: number;
  totalCount: number;
  accuracy: number;
  isExact: boolean;
  classification: 'classificada_corretamente' | 'nao_classificada_perfeitamente';
}

export interface HebbSummary {
  totalFunctions: number;
  exactFunctions: number;
  imperfectFunctions: number;
  exactFunctionIds: string[];
  imperfectFunctionIds: string[];
}

export interface TrainHebbRequest {
  initialBias: number;
  initialWeight: number;
  mode: TrainingMode;
}

export interface TrainHebbResponse {
  results: FunctionTrainingResult[];
  summary: HebbSummary;
  trainingCsv: string;
  predictionsCsv: string;
  message: string;
}

export interface ManualHebbPredictionRequest {
  functionIndex: number;
  a: LogicalValue;
  b: LogicalValue;
  initialBias: number;
  initialWeight: number;
}

export interface ManualHebbPredictionResponse {
  result: FunctionTrainingResult;
  prediction: HebbPrediction;
}

export interface LearningContent {
  title: string;
  summary: string;
  theory: {
    headline: string;
    intro: string;
    cards: Array<{ title: string; text: string }>;
    formula: string;
    activation: string;
    decisionLine: string;
  };
  program: {
    headline: string;
    summary: string;
  };
}
