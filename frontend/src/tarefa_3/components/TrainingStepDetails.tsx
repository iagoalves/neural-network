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
        <span className="tag">passo a passo da classificação</span>
        <h3>{prediction.id}</h3>
        <p>
          Esta matriz não corrige os pesos. Ela usa o modelo treinado e calcula
          <strong> u = b + Σ(xi·wi)</strong> para verificar se cai na região X ou T.
        </p>
      </header>

      <div className="training-rule-strip">
        <code>pesos treinados fixos</code>
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
          <span className="tag">amostra não usada no treino</span>
          <h3>{sampleId}</h3>
          <p>Apenas X_Principal e T_Principal participam do treinamento. As demais matrizes são usadas para verificação.</p>
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
          Os pesos começam com <strong>0.001</strong>. Para cada amostra, o perceptron calcula a saída e só corrige os pesos se houver erro.
        </p>
      </header>

      <div className="training-rule-strip">
        <code>erro = y - ŷ</code>
        <code>Δwi = erro·xi</code>
        <code>wi ← wi + Δwi</code>
      </div>

      {sortedSteps.map((step, stepIndex) => {
        const groupedDeltas = Array.from({ length: 5 }, (_, rowIndex) =>
          step.deltaWeights.map((delta, index) => ({ delta, index })).slice(rowIndex * 5, (rowIndex + 1) * 5),
        );

        return (
          <details className="training-step-details" key={`${step.sampleId}-${step.epoch}-${stepIndex}`} open={stepIndex === 0}>
            <summary>
              <span>Época {step.epoch} · {step.sampleId}</span>
              <span className="summary-pill">y = {step.target}</span>
              <span className="summary-pill">ŷ antes = {step.yBefore}</span>
              <span className="summary-pill">erro = {step.error}</span>
              <span className="summary-pill">u antes = {formatNumber(step.uBefore)}</span>
            </summary>

            <div className="training-step-details__content">
              <div className="training-step-matrices">
                <MatrixGrid compact title="Entrada x" subtitle="x1...x25" matrix={toMatrix(step.features)} />
                <MatrixGrid compact title="Pesos antes" subtitle="w antes" matrix={toMatrix(step.weightsBefore)} />
                <MatrixGrid compact title="Correção Δw" subtitle="erro·x" matrix={toMatrix(step.deltaWeights)} />
                <MatrixGrid compact title="Pesos depois" subtitle="w depois" matrix={toMatrix(step.weightsAfter)} />
              </div>

              <div className="training-step-list" aria-label="Detalhamento da correção dos pesos">
                {groupedDeltas.map((rowItems, rowIndex) => (
                  <section className="training-step-row" key={`row-${rowIndex + 1}`}>
                    <strong>Linha {rowIndex + 1}</strong>
                    <div className="training-step-row__items">
                      {rowItems.map(({ index }) => {
                        const oneBasedIndex = index + 1;
                        return (
                          <code key={`w-${oneBasedIndex}`}>
                            w{oneBasedIndex}: {formatNumber(step.weightsBefore[index])} + ({step.error} × {step.features[index]}) = {formatNumber(step.weightsAfter[index])}
                          </code>
                        );
                      })}
                    </div>
                  </section>
                ))}
              </div>

              <div className="training-step-formula">
                <code>erro = y - ŷ = {step.target} - ({step.yBefore}) = {step.error}</code>
                <code>Σ(xi·wi depois) = {formatNumber(step.weightedSumAfter)}</code>
                <code>u depois = b + Σ(xi·wi) = {formatNumber(step.biasAfter)} + ({formatNumber(step.weightedSumAfter)}) = {formatNumber(step.uAfter)}</code>
                <code>saída depois do passo: ŷ = {step.yAfter}</code>
              </div>
            </div>
          </details>
        );
      })}
    </article>
  );
}
