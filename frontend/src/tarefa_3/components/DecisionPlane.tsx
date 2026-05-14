import type { CSSProperties } from 'react';
import type { PredictionResult } from '../types/perceptron';
import { formatNumber } from '../utils/formatNumber';

interface DecisionPlaneProps {
  readonly predictions: PredictionResult[];
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export function DecisionPlane({ predictions }: DecisionPlaneProps) {
  const values = predictions.map((prediction) => prediction.u);
  const minU = Math.min(-5, ...values) - 4;
  const maxU = Math.max(5, ...values) + 4;
  const span = maxU - minU || 1;
  const zeroPosition = clamp(((0 - minU) / span) * 100, 5, 95);
  const laneTops = [20, 31, 42, 58, 69, 80];
  const minHorizontalGap = 12;
  const sortedPoints = predictions
    .map((prediction) => ({
      prediction,
      left: clamp(((prediction.u - minU) / span) * 100, 4, 96),
    }))
    .sort((a, b) => a.left - b.left);
  const lastLeftByLane = Array.from({ length: laneTops.length }, () => Number.NEGATIVE_INFINITY);
  const positionedPoints = sortedPoints.map(({ prediction, left }, sortedIndex) => {
    let laneIndex = lastLeftByLane.findIndex((lastLeft) => left - lastLeft >= minHorizontalGap);

    if (laneIndex === -1) {
      laneIndex = sortedIndex % laneTops.length;
    }

    lastLeftByLane[laneIndex] = left;

    return {
      prediction,
      left,
      top: laneTops[laneIndex],
      labelDirection: laneIndex < laneTops.length / 2 ? 'down' : 'up',
    };
  });

  return (
    <article className="activation-chart-card">
      <header>
        <span className="tag">fronteira de ativação</span>
        <h3>Regiões X e T separadas por u = 0</h3>
        <p>
          A separação real ocorre no hiperplano b + Σ(xi·wi)=0. Para visualizar, cada matriz é projetada no eixo u.
          Valores à direita de zero geram y=1, e valores à esquerda geram y=-1.
        </p>
      </header>

      <div className="activation-chart" style={{ '--zero-position': `${zeroPosition}%` } as CSSProperties}>
        <div className="activation-region activation-region--t">Região T · y=-1</div>
        <div className="activation-region activation-region--x">Região X · y=1</div>
        <div className="activation-zero-line"><span>u = 0</span></div>
        <div className="activation-axis" />

        {positionedPoints.map(({ prediction, left, top, labelDirection }) => {
          const className = prediction.predictedLabel === 'X' ? 'activation-point activation-point--x' : 'activation-point activation-point--t';

          return (
            <span
              className={className}
              key={prediction.id}
              style={{ left: `${left}%`, top: `${top}%` }}
              title={`${prediction.id}: u=${formatNumber(prediction.u)}, ŷ=${prediction.yHat}`}
            >
              <span className="activation-point__dot" aria-hidden="true" />
              <span className={`activation-point__label activation-point__label--${labelDirection}`}>
                {prediction.id.replace('_Principal', '')} · u={formatNumber(prediction.u)}
              </span>
            </span>
          );
        })}
      </div>
    </article>
  );
}
