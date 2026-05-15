import type { FunctionTrainingResult } from '../types/hebb';
import { formatNumber } from '../../geral/utils/formatNumber';
import { TruthTable } from './TruthTable';

interface HebbTrainingDetailsProps {
  readonly result: FunctionTrainingResult;
}

export function HebbTrainingDetails({ result }: HebbTrainingDetailsProps) {
  return (
    <article className="training-step-card logic-detail-card">
      <header className="training-step-card__header">
        <span className="tag">passo a passo</span>
        <h3>{result.function.id} · {result.function.name}</h3>
        <p>
          A função {result.function.expression} foi treinada linha a linha. Em Hebb não há teste de erro antes da atualização: cada alvo y reforça ou inibe os pesos diretamente.
        </p>
      </header>

      <div className="training-rule-strip">
        <code>ΔwA = y·A</code>
        <code>ΔwB = y·B</code>
        <code>Δb = y</code>
        <code>u = b + A·wA + B·wB</code>
      </div>

      <TruthTable logicalFunction={result.function} predictions={result.predictions} />

      <div className="logic-model-grid">
        <article className="training-summary-card"><span>wA</span><strong>{formatNumber(result.model.weights[0])}</strong><small>peso de A</small></article>
        <article className="training-summary-card"><span>wB</span><strong>{formatNumber(result.model.weights[1])}</strong><small>peso de B</small></article>
        <article className="training-summary-card"><span>bias</span><strong>{formatNumber(result.model.bias)}</strong><small>treinado por y</small></article>
        <article className="training-summary-card"><span>status</span><strong>{result.isExact ? 'OK' : 'Falha'}</strong><small>{result.correctCount}/{result.totalCount} acertos</small></article>
      </div>

      {result.model.trainingSteps.map((step) => (
        <details className="training-step-details" key={`${step.functionId}-${step.row}`} open={step.row === 1}>
          <summary>
            <span>Linha {step.row} · A={step.a} · B={step.b}</span>
            <span className="summary-pill">y={step.target}</span>
            <span className="summary-pill">Δw=[{step.deltaWeights.join(', ')}]</span>
            <span className="summary-pill">Δb={step.deltaBias}</span>
          </summary>
          <div className="training-step-details__content">
            <div className="training-step-formula">
              <code>antes: wA={formatNumber(step.weightsBefore[0])}; wB={formatNumber(step.weightsBefore[1])}; b={formatNumber(step.biasBefore)}</code>
              <code>entrada bipolar: A={step.features[0]}; B={step.features[1]}; y={step.target}</code>
              <code>wA ← {formatNumber(step.weightsBefore[0])} + ({step.target}·{step.features[0]}) = {formatNumber(step.weightsAfter[0])}</code>
              <code>wB ← {formatNumber(step.weightsBefore[1])} + ({step.target}·{step.features[1]}) = {formatNumber(step.weightsAfter[1])}</code>
              <code>b ← {formatNumber(step.biasBefore)} + ({step.deltaBias}) = {formatNumber(step.biasAfter)}</code>
              <code>u depois = {formatNumber(step.biasAfter)} + ({step.features[0]}·{formatNumber(step.weightsAfter[0])}) + ({step.features[1]}·{formatNumber(step.weightsAfter[1])}) = {formatNumber(step.uAfter)}</code>
            </div>
          </div>
        </details>
      ))}
    </article>
  );
}
