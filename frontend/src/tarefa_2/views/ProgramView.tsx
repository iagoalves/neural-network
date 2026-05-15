import { Play } from 'lucide-react';
import { InfoSection } from '../../geral/components/InfoSection';
import type { FunctionTrainingResult, LearningContent, LogicalValue, ManualHebbPredictionResponse, TrainHebbRequest, TrainHebbResponse } from '../types/hebb';
import { TruthTable } from '../components/TruthTable';
import { formatNumber } from '../../geral/utils/formatNumber';

interface ProgramViewProps {
  readonly content: LearningContent;
  readonly trainForm: TrainHebbRequest;
  readonly trainResult: TrainHebbResponse | null;
  readonly selectedFunctionIndex: number;
  readonly a: LogicalValue;
  readonly b: LogicalValue;
  readonly manualPrediction: ManualHebbPredictionResponse | null;
  readonly onSelectFunction: (index: number) => void;
  readonly onChangeA: (value: LogicalValue) => void;
  readonly onChangeB: (value: LogicalValue) => void;
  readonly onPredict: () => Promise<void>;
}

function findResult(result: TrainHebbResponse | null, index: number): FunctionTrainingResult | null {
  return result?.results.find((item) => item.function.index === index) ?? null;
}

export function ProgramView({
  content,
  trainForm,
  trainResult,
  selectedFunctionIndex,
  a,
  b,
  manualPrediction,
  onSelectFunction,
  onChangeA,
  onChangeB,
  onPredict,
}: ProgramViewProps) {
  const selected = findResult(trainResult, selectedFunctionIndex);

  return (
    <InfoSection eyebrow="programa" title={content.program.headline} description={content.program.summary}>
      <div className="program-layout">
        <article className="program-card">
          <header className="program-output__header">
            <span className="tag">função lógica</span>
            <h3>Seleção da função</h3>
            <p>Escolha qual das 16 funções será testada no classificador treinado.</p>
          </header>

          <label className="logic-select-label">
            Função
            <select value={selectedFunctionIndex} onChange={(event) => onSelectFunction(Number(event.target.value))}>
              {(trainResult?.results ?? []).map((item) => (
                <option key={item.function.id} value={item.function.index}>
                  {item.function.id} · {item.function.name} · {item.function.expression}
                </option>
              ))}
            </select>
          </label>

          {selected && (
            <>
              <TruthTable logicalFunction={selected.function} predictions={selected.predictions} />
              <div className="training-summary-grid">
                <article className="training-summary-card"><span>wA</span><strong>{formatNumber(selected.model.weights[0])}</strong><small>peso final</small></article>
                <article className="training-summary-card"><span>wB</span><strong>{formatNumber(selected.model.weights[1])}</strong><small>peso final</small></article>
                <article className="training-summary-card"><span>bias</span><strong>{formatNumber(selected.model.bias)}</strong><small>b final</small></article>
                <article className="training-summary-card"><span>status</span><strong>{selected.isExact ? 'OK' : 'Falha'}</strong><small>{selected.correctCount}/{selected.totalCount} acertos</small></article>
              </div>
            </>
          )}
        </article>

        <article className="program-card">
          <header className="program-output__header">
            <span className="tag">entrada manual</span>
            <h3>Teste A/B</h3>
            <p>Altere os valores lógicos e execute o modelo hebbiano da função selecionada.</p>
          </header>

          <div className="logic-toggle-grid">
            <label>
              A
              <select value={a} onChange={(event) => onChangeA(Number(event.target.value) as LogicalValue)}>
                <option value={0}>0</option>
                <option value={1}>1</option>
              </select>
            </label>
            <label>
              B
              <select value={b} onChange={(event) => onChangeB(Number(event.target.value) as LogicalValue)}>
                <option value={0}>0</option>
                <option value={1}>1</option>
              </select>
            </label>
          </div>

          <div className="formula-strip">
            <code>A={a} → {a === 1 ? '+1' : '-1'}</code>
            <code>B={b} → {b === 1 ? '+1' : '-1'}</code>
            <code>bias inicial = {formatNumber(trainForm.initialBias)}</code>
            <code>peso inicial = {formatNumber(trainForm.initialWeight)}</code>
          </div>

          <button className="run-button" onClick={onPredict} type="button">
            <Play size={16} /> Executar classificador
          </button>
        </article>
      </div>

      {manualPrediction ? (
        <section className={manualPrediction.prediction.isCorrect ? 'program-output prediction-output' : 'program-output prediction-output logic-warning-block'}>
          <header className="program-output__header">
            <span className="tag">resultado</span>
            <h3>Saída prevista: {manualPrediction.prediction.predictedOutput}</h3>
            <p>
              Esperado: {manualPrediction.prediction.expectedOutput}. Resultado {manualPrediction.prediction.isCorrect ? 'correto' : 'incorreto'} para {manualPrediction.result.function.id} · {manualPrediction.result.function.name}.
            </p>
          </header>

          <div className="prediction-summary-grid">
            <article className="prediction-summary-card prediction-summary-card--primary">
              <span>ŷ</span>
              <strong>{manualPrediction.prediction.yHat}</strong>
              <small>saída bipolar</small>
            </article>
            <article className="prediction-summary-card">
              <span>saída lógica</span>
              <strong>{manualPrediction.prediction.predictedOutput}</strong>
              <small>0 ou 1</small>
            </article>
            <article className="prediction-summary-card">
              <span>u</span>
              <strong>{formatNumber(manualPrediction.prediction.u)}</strong>
              <small>b + A·wA + B·wB</small>
            </article>
            <article className="prediction-summary-card">
              <span>correto?</span>
              <strong>{manualPrediction.prediction.isCorrect ? 'sim' : 'não'}</strong>
              <small>comparado à tabela</small>
            </article>
          </div>

          <div className="training-step-formula">
            <code>contribuições = [{manualPrediction.prediction.contributions.map((value) => formatNumber(value)).join(', ')}]</code>
            <code>Σ = {formatNumber(manualPrediction.prediction.weightedSum)}</code>
            <code>u = {formatNumber(manualPrediction.prediction.bias)} + ({formatNumber(manualPrediction.prediction.weightedSum)}) = {formatNumber(manualPrediction.prediction.u)}</code>
            <code>ŷ = {manualPrediction.prediction.yHat}; saída lógica = {manualPrediction.prediction.predictedOutput}</code>
          </div>
        </section>
      ) : (
        <article className="program-output program-output--empty">
          <p>Selecione a função, defina A/B e execute o classificador para ver u, ŷ e a saída lógica prevista.</p>
        </article>
      )}
    </InfoSection>
  );
}
