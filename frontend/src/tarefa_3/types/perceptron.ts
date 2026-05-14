export type MatrixValue = 1 | -1;
export type Matrix5x5 = MatrixValue[][];
export type TargetValue = 1 | -1;
export type PatternLabel = 'X' | 'T';
export type TrainingMode = 'hebb';

export interface MatrixPattern {
  id: string;
  label: PatternLabel;
  target: TargetValue;
  matrix: Matrix5x5;
  vector: MatrixValue[];
}

export interface ElementCalculation {
  row: number;
  col: number;
  inputValue: MatrixValue;
  weight: number;
  product: number;
}

export interface TrainingPoint {
  id: string;
  label: PatternLabel;
  target: TargetValue;
  features: MatrixValue[];
}

export interface TrainingStep {
  epoch: number;
  sampleId: string;
  target: TargetValue;
  features: MatrixValue[];
  weightsBefore: number[];
  biasBefore: number;
  uBefore: number;
  yBefore: TargetValue;
  updated: boolean;
  weightsAfter: number[];
  biasAfter: number;
}

export interface TrainedModel {
  weights: number[];
  bias: number;
  epochs: number;
  trainingMode: TrainingMode;
  trainingPoints: TrainingPoint[];
  trainingSteps: TrainingStep[];
}

export interface PredictionResult {
  pattern: MatrixPattern;
  id: string;
  expectedLabel: PatternLabel;
  expectedTarget: TargetValue;
  features: MatrixValue[];
  weights: number[];
  weightsMatrix: number[][];
  bias: number;
  contributions: number[];
  contributionsMatrix: number[][];
  weightedSum: number;
  u: number;
  yHat: TargetValue;
  predictedLabel: PatternLabel;
  isCorrect: boolean;
  details: ElementCalculation[];
}

export interface SamplesResponse {
  model: TrainedModel;
  samples: MatrixPattern[];
  predictions: PredictionResult[];
}

export interface TrainPerceptronRequest {
  initialBias: number;
  mode: TrainingMode;
}

export interface TrainPerceptronResponse extends SamplesResponse {
  trainingCsv: string;
  message: string;
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
