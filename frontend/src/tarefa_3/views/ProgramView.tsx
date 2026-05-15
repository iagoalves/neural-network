import { Play, RotateCcw, Wand2 } from 'lucide-react';
import { InfoSection } from '../components/InfoSection';
import { MatrixGrid } from '../components/MatrixGrid';
import { ProgramMatrixEditor } from '../components/ProgramMatrixEditor';
import type { LearningContent, Matrix5x5, MatrixPattern, PredictionResult, SamplesResponse, TrainPerceptronResponse } from '../types/perceptron';
import { formatNumber } from '../utils/formatNumber';

const EMPTY_MATRIX: Matrix5x5 = [
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
];

interface ProgramViewProps {
  readonly content: LearningContent;
  readonly activeData: SamplesResponse | TrainPerceptronResponse | null;
  readonly patterns: MatrixPattern[];
  readonly manualMatrix: Matrix5x5;
  readonly manualPrediction: PredictionResult | null;
  readonly onMatrixChange: (matrix: Matrix5x5) => void;
  readonly onPredict: () => Promise<void>;
  readonly onLoadPattern: (label: 'X' | 'T') => void;
  readonly onClearMatrix: () => void;
}

export function ProgramView({
  content,
  activeData,
  manualMatrix,
  manualPrediction,
  onMatrixChange,
  onPredict,
  onLoadPattern,
  onClearMatrix,
}: ProgramViewProps) {
  return (
    <InfoSection eyebrow="programa" title={content.program.headline} description={content.program.summary}>
      <div className="program-layout program-layout--single">
        {activeData?.model && (
          <article className="program-card">
            <header className="program-output__header">
              <span className="tag">modelo ativo</span>
              <h3>Parâmetros usados</h3>
              <p>O classificador manual usa os pesos treinados por correção de erro.</p>
            </header>

            <div className="training-summary-grid">
              <article className="training-summary-card"><span>modo</span><strong>Correção de erro</strong><small>supervisionado</small></article>
              <article className="training-summary-card"><span>bias fixo</span><strong>{formatNumber(activeData.model.bias)}</strong><small>somado em u</small></article>
              <article className="training-summary-card"><span>passagem</span><strong>{activeData.model.epochs}</strong><small>uma passagem</small></article>
              <article className="training-summary-card"><span>amostras de treino</span><strong>{activeData.model.trainingPoints.length}</strong><small>X e T principais</small></article>
            </div>
          </article>
        )}

        <article className="program-card">
          <div className="program-actions">
            <button onClick={() => onLoadPattern('X')} type="button"><Wand2 size={16} /> Carregar X</button>
            <button onClick={() => onLoadPattern('T')} type="button"><Wand2 size={16} /> Carregar T</button>
            <button onClick={onClearMatrix} type="button"><RotateCcw size={16} /> Limpar</button>
          </div>

          <ProgramMatrixEditor matrix={manualMatrix} onChange={onMatrixChange} />

          <button className="run-button" onClick={onPredict} type="button">
            <Play size={16} /> Executar classificador
          </button>
        </article>
      </div>

      {manualPrediction ? (
        <section className={`program-output prediction-output prediction-output--${manualPrediction.predictedLabel.toLowerCase()}`}>
          <header className="program-output__header">
            <span className="tag">resultado</span>
            <h3>Classe prevista: {manualPrediction.predictedLabel}</h3>
            <p>Resultado calculado com os pesos treinados por correção de erro.</p>
          </header>

          <div className="prediction-summary-grid">
            <article className="prediction-summary-card prediction-summary-card--primary">
              <span>saída</span>
              <strong>ŷ = {manualPrediction.yHat}</strong>
              <small>{manualPrediction.yHat === 1 ? '1 representa X' : '-1 representa T'}</small>
            </article>
            <article className="prediction-summary-card">
              <span>u</span>
              <strong>{formatNumber(manualPrediction.u)}</strong>
              <small>b + Σ(xi·wi)</small>
            </article>
            <article className="prediction-summary-card">
              <span>soma ponderada</span>
              <strong>{formatNumber(manualPrediction.weightedSum)}</strong>
              <small>Σ(xi·wi)</small>
            </article>
          </div>

          <MatrixGrid
            title="Entrada classificada"
            subtitle={`ŷ = ${manualPrediction.yHat} · predito: ${manualPrediction.predictedLabel}`}
            matrix={manualPrediction.pattern.matrix}
          />
        </section>
      ) : (
        <article className="program-output program-output--empty">
          <p>Altere a matriz e execute o classificador para ver u, ŷ e a classe prevista.</p>
        </article>
      )}
    </InfoSection>
  );
}
