import { useMemo } from 'react';
import { MatrixGrid } from './MatrixGrid';
import type { PredictionResult, TrainingStep } from '../types/perceptron';
import { formatNumber } from '../utils/formatNumber';

interface TrainingStepDetailsProps {
  readonly sampleId: string;
  readonly steps: TrainingStep[];
  readonly prediction?: PredictionResult;
}

function toMatrix(values: number[]): number[][] {
  return Array.from({ length: 5 }, (_, rowIndex) => values.slice(rowIndex * 5, (rowIndex + 1) * 5));
}

function stepContribution(step: TrainingStep, index: number) {
  return step.target * step.features[index];
}

function weightedProductsAfter(step: TrainingStep) {
  return step.features.map((feature, index) => feature * step.weightsAfter[index]);
}

function sum(values: number[]) {
  return values.reduce((total, value) => total + value, 0);
}

function buildWeightedExpression(step: TrainingStep) {
  return step.features
    .map((feature, index) => `${feature}×${formatNumber(step.weightsAfter[index])}`)
    .join(' + ');
}

function groupedIndexes() {
  return Array.from({ length: 5 }, (_, rowIndex) =>
    Array.from({ length: 5 }, (_, colIndex) => rowIndex * 5 + colIndex),
  );
}

function VerificationStepDetails({ prediction }: { readonly prediction: PredictionResult }) {
  const grouped = groupedIndexes();

  return (
    <article className="training-step-card training-step-card--verification">
      <header className="training-step-card__header">
        <span className="tag">passo a passo da verificação</span>
        <h3>{prediction.id}</h3>
        <p>
          Esta matriz não altera os pesos. Ela usa o modelo já treinado pela Regra de Hebb simples e verifica a saída com
          <strong> u = b + Σ(xi·wi)</strong>.
        </p>
      </header>

      <div className="training-rule-strip">
        <code>pesos fixos após o treino</code>
        <code>u = b + Σ(xi·wi)</code>
        <code>ŷ = 1 se u ≥ 0; senão ŷ = -1</code>
      </div>

      <div className="verification-step-matrices">
        <MatrixGrid compact title="Entrada x" subtitle="matriz verificada" matrix={prediction.pattern.matrix} />
        <MatrixGrid compact title="Pesos w" subtitle="modelo treinado" matrix={prediction.weightsMatrix} />
        <MatrixGrid compact title="Produtos" subtitle="xi × wi" matrix={prediction.contributionsMatrix} />
      </div>

      <div className="training-step-list" aria-label="Detalhamento da verificação com os pesos treinados">
        {grouped.map((indexes, rowIndex) => (
          <section className="training-step-row" key={`verification-row-${rowIndex + 1}`}>
            <strong>Linha {rowIndex + 1}</strong>
            <div className="training-step-row__items">
              {indexes.map((index) => {
                const row = Math.floor(index / 5) + 1;
                const col = (index % 5) + 1;
                const detail = prediction.details[index];
                return (
                  <code key={`verification-${row}-${col}`}>
                    M[{row},{col}] = x{index + 1} × w{index + 1} = {detail.inputValue} × {formatNumber(detail.weight)} = {formatNumber(detail.product)}
                  </code>
                );
              })}
            </div>
          </section>
        ))}
      </div>

      <div className="training-step-formula">
        <code>Σ(xi·wi) = {formatNumber(prediction.weightedSum)}</code>
        <code>u = b + Σ(xi·wi) = {formatNumber(prediction.bias)} + ({formatNumber(prediction.weightedSum)}) = {formatNumber(prediction.u)}</code>
        <code>resultado: ŷ = {prediction.yHat} → classe {prediction.predictedLabel}</code>
      </div>
    </article>
  );
}

export function TrainingStepDetails({ sampleId, steps, prediction }: TrainingStepDetailsProps) {
  const sortedSteps = useMemo(
    () => [...steps].sort((first, second) => first.epoch - second.epoch),
    [steps],
  );

  if (sortedSteps.length === 0) {
    if (prediction) {
      return <VerificationStepDetails prediction={prediction} />;
    }

    return (
      <article className="training-step-card training-step-card--empty">
        <header className="training-step-card__header">
          <span className="tag">amostra não treinada</span>
          <h3>{sampleId}</h3>
          <p>
            Esta matriz não participa do treino Hebb simples. Nesta versão, apenas
            <strong> X_Principal</strong> e <strong> T_Principal</strong> atualizam os pesos. As variações aparecem para testar visualmente o modelo treinado.
          </p>
        </header>
      </article>
    );
  }

  return (
    <article className="training-step-card">
      <header className="training-step-card__header">
        <span className="tag">passo a passo do treino</span>
        <h3>{sampleId}</h3>
        <p>
          Cada entrada <strong>xi</strong> atualiza seu peso correspondente <strong>wi</strong> usando o rótulo esperado
          da amostra. O Hebb simples aplica a regra uma vez para cada padrão principal.
        </p>
      </header>

      <div className="training-rule-strip">
        <code>wi ← wi + y·xi</code>
        <code>b fixo</code>
        <code>y = 1 para X; y = -1 para T</code>
      </div>

      {sortedSteps.map((step, stepIndex) => {
        const deltas = step.features.map((_, index) => stepContribution(step, index));
        const groupedDeltas = Array.from({ length: 5 }, (_, rowIndex) =>
          deltas.map((delta, index) => ({ delta, index })).slice(rowIndex * 5, (rowIndex + 1) * 5),
        );
        const productsAfter = weightedProductsAfter(step);
        const weightedSumAfter = sum(productsAfter);
        const uAfter = step.biasAfter + weightedSumAfter;
        const yAfter = uAfter >= 0 ? 1 : -1;
        const weightedExpression = buildWeightedExpression(step);

        return (
          <details className="training-step-details" key={`${step.sampleId}-${step.epoch}-${stepIndex}`} open={stepIndex === 0}>
            <summary>
              <span>Passo {stepIndex + 1} · {step.sampleId}</span>
              <span className="summary-pill">y = {step.target}</span>
              <span className="summary-pill">ŷ antes = {step.yBefore}</span>
              <span className="summary-pill">u antes = {formatNumber(step.uBefore)}</span>
            </summary>

            <div className="training-step-details__content">
              <div className="training-step-matrices">
                <MatrixGrid compact title="Entrada x" subtitle="x1...x25" matrix={toMatrix(step.features)} />
                <MatrixGrid compact title="Pesos antes" subtitle="w antes" matrix={toMatrix(step.weightsBefore)} />
                <MatrixGrid compact title="Ajuste Δw" subtitle="y·x" matrix={toMatrix(deltas)} />
                <MatrixGrid compact title="Pesos depois" subtitle="w depois" matrix={toMatrix(step.weightsAfter)} />
              </div>

              <div className="training-step-list" aria-label="Detalhamento da atualização dos pesos">
                {groupedDeltas.map((rowItems, rowIndex) => (
                  <section className="training-step-row" key={`row-${rowIndex + 1}`}>
                    <strong>Linha {rowIndex + 1}</strong>
                    <div className="training-step-row__items">
                      {rowItems.map(({ index }) => {
                        const oneBasedIndex = index + 1;
                        return (
                          <code key={`w-${oneBasedIndex}`}>
                            w{oneBasedIndex}: {formatNumber(step.weightsBefore[index])} + {step.target} × {step.features[index]} = {formatNumber(step.weightsAfter[index])}
                          </code>
                        );
                      })}
                    </div>
                  </section>
                ))}
              </div>

              <div className="training-step-formula">
                <code>Δw = y·x = {step.target} × x</code>
                <code>Σ(xi·wi depois) = {weightedExpression} = {formatNumber(weightedSumAfter)}</code>
                <code>u depois = b + Σ(xi·wi) = {formatNumber(step.biasAfter)} + ({formatNumber(weightedSumAfter)}) = {formatNumber(uAfter)}</code>
                <code>saída depois do passo: ŷ = {yAfter} → classe {yAfter === 1 ? 'X' : 'T'}</code>
              </div>
            </div>
          </details>
        );
      })}
    </article>
  );
}
