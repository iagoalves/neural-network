import { MatrixGrid } from './MatrixGrid';
import type { PredictionResult } from '../types/perceptron';
import { formatNumber } from '../utils/formatNumber';

interface SampleMatrixCardProps {
  readonly prediction: PredictionResult;
  readonly selected: boolean;
  readonly onSelect: (prediction: PredictionResult) => void;
}

export function SampleMatrixCard({ prediction, selected, onSelect }: SampleMatrixCardProps) {
  return (
    <article className={selected ? 'sample-matrix-card sample-matrix-card--selected' : 'sample-matrix-card'}>
      <button className="sample-matrix-card__button" onClick={() => onSelect(prediction)} type="button">
        <MatrixGrid
          compact
          title={prediction.id.replace('_Principal', ' principal')}
          subtitle={`y esperado = ${prediction.expectedTarget}`}
          matrix={prediction.pattern.matrix}
        />

        <div className="sample-matrix-card__meta">
          <span>soma: <strong>{formatNumber(prediction.weightedSum)}</strong></span>
          <span>bias: <strong>{formatNumber(prediction.bias)}</strong></span>
          <span>u: <strong>{formatNumber(prediction.u)}</strong></span>
          <span>ŷ: <strong>{prediction.yHat}</strong></span>
          <span className="sample-matrix-card__hint">Clique para ver a classificação</span>
        </div>
      </button>
    </article>
  );
}
