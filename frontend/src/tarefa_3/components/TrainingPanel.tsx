import { Download, RefreshCw } from 'lucide-react';
import type { TrainPerceptronRequest, TrainPerceptronResponse } from '../types/perceptron';

interface TrainingPanelProps {
  readonly form: TrainPerceptronRequest;
  readonly onChange: (form: TrainPerceptronRequest) => void;
  readonly onTrain: () => Promise<void>;
  readonly result: TrainPerceptronResponse | null;
  readonly loading: boolean;
}

function updateNumber(form: TrainPerceptronRequest, key: keyof TrainPerceptronRequest, value: string): TrainPerceptronRequest {
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
          <span className="tag">treino supervisionado</span>
          <h3>Correção de erro do perceptron</h3>
          <p>
            Os pesos começam em <strong>0.001</strong>. O sistema calcula <strong>u</strong>, compara a saída prevista com o alvo e corrige os pesos somente quando existe erro.
          </p>
        </header>

        <div className="training-form-grid training-form-grid--simple">
          <label>
            Bias fixo
            <input
              min={-50}
              max={50}
              type="number"
              value={form.initialBias}
              onChange={(event) => onChange(updateNumber(form, 'initialBias', event.target.value))}
            />
          </label>
        </div>

        <div className="formula-strip">
          <code>w inicial = 0.001</code>
          <code>erro = y - ŷ</code>
          <code>Δwi = erro·xi</code>
          <code>wi ← wi + Δwi</code>
        </div>

        <div className="training-actions">
          <button className="run-button" onClick={onTrain} disabled={loading} type="button">
            <RefreshCw size={16} /> {loading ? 'Treinando...' : 'Executar treino'}
          </button>

          <button
            disabled={!result?.trainingCsv}
            onClick={() => result?.trainingCsv && downloadTextFile('treino_correcao_erro_tarefa_3.csv', result.trainingCsv)}
            type="button"
          >
            <Download size={16} /> Salvar CSV do treino
          </button>
        </div>
      </article>

      {result && (
        <article className={`training-result-card training-result-card--${result.message ? 'success' : 'error'}`}>
          <div className="training-result-card__header">
            <span className="tag">treino executado</span>
            <p className="training-result-card__message">
              {result.message ?? 'O treino não foi concluído. Verifique os parâmetros e tente novamente.'}
            </p>
          </div>

          <div className="training-result-card__stats" aria-label="Resumo do treino executado">
            <span>
              bias usado
              <strong>{result.model.bias}</strong>
            </span>
            <span>
              peso inicial
              <strong>{result.model.initialWeight}</strong>
            </span>
            <span>
              passos registrados
              <strong>{result.model.trainingSteps.length}</strong>
            </span>
            <span>
              épocas executadas
              <strong>{result.model.epochs}</strong>
            </span>
          </div>
        </article>
      )}
    </section>
  );
}
