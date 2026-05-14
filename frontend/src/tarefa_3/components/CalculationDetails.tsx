import type { PredictionResult } from '../types/perceptron';
import { formatNumber } from '../utils/formatNumber';

interface CalculationDetailsProps {
  readonly prediction: PredictionResult;
}

export function CalculationDetails({ prediction }: CalculationDetailsProps) {
  return (
    <article className="detail-card">
      <header className="detail-card__header">
        <span className="tag">cálculo detalhado</span>
        <h3>{prediction.id}</h3>
        <p>
          Cada linha mostra uma multiplicação direta do perceptron clássico: <strong>xi × wi</strong>. A soma dos produtos recebe o bias e passa pela função de ativação.
        </p>
      </header>

      <div className="detail-list">
        <strong>Produtos xi × wi</strong>
        {prediction.details.map((item) => {
          const index = (item.row - 1) * 5 + item.col;
          return (
            <code key={`m-${item.row}-${item.col}`}>
              M[{item.row},{item.col}] = x{index} × w{index} = {item.inputValue} × {formatNumber(item.weight)} = {formatNumber(item.product)}
            </code>
          );
        })}
        <strong className="result-line">Σ(xi × wi) = {formatNumber(prediction.weightedSum)}</strong>
      </div>

      <div className="activation-panel">
        <strong>Ativação</strong>
        <code>u = {formatNumber(prediction.bias)} + ({formatNumber(prediction.weightedSum)}) = {formatNumber(prediction.u)}</code>
        <span>ŷ = {prediction.yHat} → classe {prediction.predictedLabel}</span>
      </div>
    </article>
  );
}
