import { useRef, useState } from 'react';
import type { PredictionResult } from '../types/perceptron';

export type TheorySelectionKind = 'training' | 'verification';

export interface TheorySelection {
  readonly kind: TheorySelectionKind;
  readonly id: string;
}

export function useSelectedPrediction(predictions: PredictionResult[]) {
  const [selection, setSelection] = useState<TheorySelection | null>(null);
  const targetRef = useRef<HTMLDivElement | null>(null);

  const selectedPrediction =
    selection?.kind === 'verification'
      ? predictions.find((prediction) => prediction.id === selection.id) ?? null
      : null;

  const selectedTraining = selection?.kind === 'training' ? selection : null;

  function scrollToTarget() {
    window.setTimeout(() => {
      targetRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 80);
  }

  function selectPrediction(prediction: PredictionResult) {
    setSelection({ kind: 'verification', id: prediction.id });
    scrollToTarget();
  }

  function selectTraining(sampleId: string) {
    setSelection({ kind: 'training', id: sampleId });
    scrollToTarget();
  }

  return {
    selection,
    selectedPrediction,
    selectedTraining,
    selectPrediction,
    selectTraining,
    targetRef,
  };
}
