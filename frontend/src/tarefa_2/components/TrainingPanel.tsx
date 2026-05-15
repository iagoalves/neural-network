import { Download, RefreshCw } from 'lucide-react';
import type { TrainHebbRequest, TrainHebbResponse } from '../types/hebb';

interface TrainingPanelProps {
  readonly form: TrainHebbRequest;
  readonly onChange: (form: TrainHebbRequest) => void;
  readonly onTrain: () => Promise<void>;
  readonly result: TrainHebbResponse | null;
  readonly loading: boolean;
}

function updateNumber(form: TrainHebbRequest, key: keyof TrainHebbRequest, value: string): TrainHebbRequest {
  const parsed = Number(value);
  return { ...form, [key]: Number.isFinite(parsed) ? parsed : 0 };
}

function downloadTextFile(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function TrainingPanel({ form, onChange, onTrain, result, loading }: TrainingPanelProps) {
  return (
    <section className="training-panel">
      <article className="training-card">
        <header className="training-card__header">
          <span className="tag">treino hebbiano</span>
          <h3>Regra de Hebb para F0...F15</h3>
          <p>
            O treino percorre as quatro linhas da tabela verdade de cada função. Os pesos começam no valor configurado e recebem Δw = y·x. O bias recebe Δb = y.
          </p>
        </header>

        <div className="training-form-grid">
          <label>
            Bias inicial
            <input
              min={-10}
              max={10}
              type="number"
              value={form.initialBias}
              onChange={(event) => onChange(updateNumber(form, 'initialBias', event.target.value))}
            />
          </label>
          <label>
            Peso inicial
            <input
              min={-10}
              max={10}
              type="number"
              value={form.initialWeight}
              onChange={(event) => onChange(updateNumber(form, 'initialWeight', event.target.value))}
            />
          </label>
        </div>

        <div className="formula-strip">
          <code>0 → -1</code>
          <code>1 → +1</code>
          <code>Δwi = y·xi</code>
          <code>Δb = y</code>
        </div>

        <div className="training-actions">
          <button className="run-button" onClick={onTrain} disabled={loading} type="button">
            <RefreshCw size={16} /> {loading ? 'Treinando...' : 'Executar treino'}
          </button>
          <button
            disabled={!result?.trainingCsv}
            onClick={() => result?.trainingCsv && downloadTextFile('treino_hebb_trabalho_2.csv', result.trainingCsv)}
            type="button"
          >
            <Download size={16} /> CSV do treino
          </button>
          <button
            disabled={!result?.predictionsCsv}
            onClick={() => result?.predictionsCsv && downloadTextFile('predicoes_hebb_trabalho_2.csv', result.predictionsCsv)}
            type="button"
          >
            <Download size={16} /> CSV das predições
          </button>
        </div>
      </article>

      {result && (
        <article className="training-result-card training-result-card--success">
          <div className="training-result-card__header">
            <span className="tag">treino executado</span>
            <p className="training-result-card__message">{result.message}</p>
          </div>
          <div className="training-result-card__stats" aria-label="Resumo do treino hebbiano">
            <span>funções<strong>{result.summary.totalFunctions}</strong></span>
            <span>perfeitas<strong>{result.summary.exactFunctions}</strong></span>
            <span>imperfeitas<strong>{result.summary.imperfectFunctions}</strong></span>
            <span>casos críticos<strong>{result.summary.imperfectFunctionIds.join(', ') || 'nenhum'}</strong></span>
          </div>
        </article>
      )}
    </section>
  );
}
