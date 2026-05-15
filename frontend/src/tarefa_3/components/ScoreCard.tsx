import { useMemo } from 'react';
import { MatrixGrid } from './MatrixGrid';
import type { ElementCalculation, PredictionResult } from '../types/perceptron';
import { formatNumber } from '../utils/formatNumber';

interface ScoreCardProps {
  readonly prediction: PredictionResult;
}

function sumExpression(details: ElementCalculation[]) {
  return details.map((item) => formatNumber(item.product)).join(' + ');
}

export function ScoreCard({ prediction }: ScoreCardProps) {
  const groupedRows = useMemo(
    () => Array.from({ length: 5 }, (_, rowIndex) => prediction.details.filter((item) => item.row === rowIndex + 1)),
    [prediction.details],
  );

  return (
    <article className="score-card score-card--winner">
      <header className="score-card__header-box">
        <span className="score-card__eyebrow">Perceptron clássico · 25 entradas</span>
        <strong className="score-card__title">u = b + Σ(xi·wi)</strong>
        <span className="score-card__result score-card__result--winner">
          {prediction.predictedLabel} · ŷ = {prediction.yHat}
        </span>
      </header>

      <div className="score-card__values score-card__values--wide">
        <span>soma ponderada: <strong>{formatNumber(prediction.weightedSum)}</strong></span>
        <span>bias: <strong>{formatNumber(prediction.bias)}</strong></span>
        <span>u: <strong>{formatNumber(prediction.u)}</strong></span>
      </div>

      <div className="score-card__details">
        <div className="score-card__explanation">
          <strong>Conta passo a passo</strong>
          <p>
            Cada célula da entrada é uma entrada xi. Cada posição do modelo treinado é um peso wi.
            O produto xi·wi é somado para formar a soma ponderada. Depois o bias é adicionado.
          </p>
        </div>

        <div className="score-card__matrix-comparison">
          <MatrixGrid compact title="Entrada x" subtitle="x1...x25" matrix={prediction.pattern.matrix} />
          <MatrixGrid compact title="Pesos w" subtitle="w1...w25" matrix={prediction.weightsMatrix} precision={8} />
          <MatrixGrid compact title="Produtos" subtitle="xi × wi" matrix={prediction.contributionsMatrix} precision={8} />
        </div>

        <div className="score-card__steps" aria-label="Cálculo detalhado de xi vezes wi">
          {groupedRows.map((rowItems, rowIndex) => (
            <section className="score-card__step-row" key={`row-${rowIndex + 1}`}>
              <strong>Linha {rowIndex + 1}</strong>
              <div className="score-card__calc-list">
                {rowItems.map((item) => {
                  const index = (item.row - 1) * 5 + item.col;
                  return (
                    <code key={`${item.row}-${item.col}`}>
                      M[{item.row},{item.col}] = x{index} × w{index} = {item.inputValue} × {formatNumber(item.weight)} = {formatNumber(item.product)}
                    </code>
                  );
                })}
              </div>
            </section>
          ))}
        </div>

        <div className="score-card__formula">
          <code>Σ(xi·wi) = {sumExpression(prediction.details)} = {formatNumber(prediction.weightedSum)}</code>
          <code>u = b + Σ(xi·wi) = {formatNumber(prediction.bias)} + ({formatNumber(prediction.weightedSum)}) = {formatNumber(prediction.u)}</code>
          <code>ŷ = 1 se u ≥ 0; caso contrário, ŷ = -1</code>
        </div>
      </div>
    </article>
  );
}
