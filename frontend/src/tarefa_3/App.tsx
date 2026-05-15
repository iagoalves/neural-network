import { useEffect, useMemo, useState } from 'react';
import { Binary, BookOpen, Play, SlidersHorizontal } from 'lucide-react';
import { fallbackContent } from './data/fallbackContent';
import { PerceptronApi } from './services/PerceptronApi';
import { TrainingView } from './views/TrainingView';
import { TheoryView } from './views/TheoryView';
import { ProgramView } from './views/ProgramView';
import { TrainingTopButton } from '../geral/components/TrainingTopButton';
import type {
  LearningContent,
  Matrix5x5,
  PredictionResult,
  SamplesResponse,
  TrainPerceptronRequest,
  TrainPerceptronResponse,
} from './types/perceptron';
import './styles.css';

type ViewMode = 'treino' | 'teoria' | 'programa';

const api = new PerceptronApi();

const EMPTY_MATRIX: Matrix5x5 = [
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
];

const STORAGE_KEYS = {
  activeView: 'trabalho_3.error_correction.activeView',
  trainingForm: 'trabalho_3.error_correction.trainingForm',
  trainingResult: 'trabalho_3.error_correction.trainingResult',
  trainingCsv: 'trabalho_3.error_correction.trainingCsv',
} as const;

const DEFAULT_TRAINING_FORM: TrainPerceptronRequest = { initialBias: 1, mode: 'error_correction' };

function readStorage<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key: string, value: unknown) {
  window.localStorage.setItem(key, JSON.stringify(value));
}

function normalizeForm(value: Partial<TrainPerceptronRequest> | null | undefined): TrainPerceptronRequest {
  return {
    initialBias: typeof value?.initialBias === 'number' ? value.initialBias : DEFAULT_TRAINING_FORM.initialBias,
    mode: 'error_correction',
  };
}

function validateForm(form: TrainPerceptronRequest): string | null {
  if (form.initialBias < -50 || form.initialBias > 50) return 'O bias fixo deve ficar entre -50 e 50.';
  return null;
}

function isTrainResponse(value: SamplesResponse | TrainPerceptronResponse): value is TrainPerceptronResponse {
  return 'trainingCsv' in value;
}

function weightsToMatrix(weights: number[]): number[][] {
  return Array.from({ length: 5 }, (_, row) => weights.slice(row * 5, (row + 1) * 5));
}

export function App() {
  const [activeView, setActiveViewState] = useState<ViewMode>(() =>
    readStorage<ViewMode>(STORAGE_KEYS.activeView, 'treino'),
  );
  const [content, setContent] = useState<LearningContent>(fallbackContent);
  const [samplesData, setSamplesData] = useState<SamplesResponse | null>(null);
  const [manualPrediction, setManualPrediction] = useState<PredictionResult | null>(null);
  const [manualMatrix, setManualMatrix] = useState<Matrix5x5>(EMPTY_MATRIX);
  const [status, setStatus] = useState('Carregando backend Python...');
  const [trainingForm, setTrainingForm] = useState<TrainPerceptronRequest>(DEFAULT_TRAINING_FORM);
  const [trainingResult, setTrainingResult] = useState<TrainPerceptronResponse | null>(null);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [modalMessage, setModalMessage] = useState<string | null>(null);

  function setActiveView(next: ViewMode) {
    setActiveViewState(next);
    writeStorage(STORAGE_KEYS.activeView, next);
  }

  function handleTrainingFormChange(next: TrainPerceptronRequest) {
    const normalized = normalizeForm(next);
    setTrainingForm(normalized);
    writeStorage(STORAGE_KEYS.trainingForm, normalized);
  }

  useEffect(() => {
    let cancelled = false;

    // Restore persisted form — single source of truth, no useState initializer redundancy.
    const cachedForm = normalizeForm(readStorage<Partial<TrainPerceptronRequest>>(STORAGE_KEYS.trainingForm, {}));
    const startupForm = validateForm(cachedForm) ? DEFAULT_TRAINING_FORM : cachedForm;
    setTrainingForm(startupForm);
    writeStorage(STORAGE_KEYS.trainingForm, startupForm);

    // Show cached result immediately while re-training.
    const cachedResult = readStorage<TrainPerceptronResponse | null>(STORAGE_KEYS.trainingResult, null);
    if (cachedResult) {
      setTrainingResult(cachedResult);
      setSamplesData({ model: cachedResult.model, samples: cachedResult.samples, predictions: cachedResult.predictions });
      setManualMatrix(cachedResult.samples.find((s) => s.id === 'X_Principal')?.matrix ?? EMPTY_MATRIX);
    }

    // Always re-train against live backend so theory/program use the fresh model.
    Promise.all([api.getLearningContent(), api.train(startupForm)])
      .then(([learning, result]) => {
        if (cancelled) return;
        setContent(learning);
        setSamplesData({ model: result.model, samples: result.samples, predictions: result.predictions });
        setManualMatrix(result.samples.find((s) => s.id === 'X_Principal')?.matrix ?? EMPTY_MATRIX);

        if (isTrainResponse(result)) {
          setTrainingResult(result);
          writeStorage(STORAGE_KEYS.trainingResult, result);
          writeStorage(STORAGE_KEYS.trainingCsv, result.trainingCsv);
        }
        setStatus('');
      })
      .catch(() => {
        if (cancelled) return;
        setContent(fallbackContent);
        setStatus('Backend não encontrado. Inicie a API Python na porta 8787 para executar o programa.');
      });

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeData = trainingResult ?? samplesData;
  const patterns = activeData?.samples ?? [];
  const predictions = activeData?.predictions ?? [];
  const weightsMatrix = useMemo(() => weightsToMatrix(activeData?.model.weights ?? []), [activeData]);

  function loadPattern(label: 'X' | 'T') {
    const pattern =
      patterns.find((p) => p.id === `${label}_Principal`) ?? patterns.find((p) => p.label === label);
    if (!pattern) return;
    setManualMatrix(pattern.matrix);
    setManualPrediction(null);
  }

  function clearMatrix() {
    setManualMatrix(EMPTY_MATRIX);
    setManualPrediction(null);
  }

  async function trainPerceptron() {
    const next = normalizeForm(trainingForm);
    const error = validateForm(next);
    if (error) { setModalMessage(error); return; }

    setTrainingLoading(true);
    try {
      const result = await api.train(next);
      setTrainingForm(next);
      setTrainingResult(result);
      setSamplesData({ model: result.model, samples: result.samples, predictions: result.predictions });
      writeStorage(STORAGE_KEYS.trainingForm, next);
      writeStorage(STORAGE_KEYS.trainingResult, result);
      writeStorage(STORAGE_KEYS.trainingCsv, result.trainingCsv);

      if (manualPrediction) {
        try { setManualPrediction(await api.predict(manualMatrix, 1)); }
        catch { setManualPrediction(null); }
      } else {
        setManualPrediction(null);
      }
      setStatus('');
    } catch (err) {
      setModalMessage(err instanceof Error ? err.message : 'Falha ao treinar o perceptron.');
    } finally {
      setTrainingLoading(false);
    }
  }

  async function runManualPrediction() {
    try {
      setManualPrediction(await api.predict(manualMatrix, 1));
    } catch (err) {
      setModalMessage(err instanceof Error ? err.message : 'Falha ao executar predição.');
    }
  }

  return (
    <main className="app-shell app-shell--task3">
      <section className="hero" id="trabalho3-painel-superior">
        <span className="hero__eyebrow"><Binary size={16} /> trabalho_3</span>
        <div className="hero__content">
          <div>
            <h1>{content.title}</h1>
            <p>{content.summary}</p>
          </div>
          <div className="hero__cards">
            <article className="hero-card"><strong>Correção de erro</strong><span>Pesos iniciam em 0.001 e só mudam quando ŷ ≠ y.</span></article>
            <article className="hero-card"><strong>25 entradas</strong><span>Cada matriz 5x5 vira x1...x25.</span></article>
            <article className="hero-card hero-card--diagram">
              <strong>Fluxo visual</strong>
              <svg className="hero-diagram" viewBox="0 0 360 72" role="img" aria-label="Fluxo do perceptron com correção de erro">
                <rect x="4" y="18" width="58" height="36" rx="10" />
                <text x="33" y="41">x</text>
                <path d="M68 36 H104" />
                <rect x="108" y="18" width="88" height="36" rx="10" />
                <text x="152" y="41">u</text>
                <path d="M202 36 H238" />
                <rect x="242" y="18" width="54" height="36" rx="10" />
                <text x="269" y="41">ŷ</text>
                <path d="M300 36 H340" />
                <text x="156" y="12">b + Σxi·wi</text>
                <text x="310" y="64">erro → Δw</text>
              </svg>
            </article>
          </div>
        </div>
      </section>

      <nav className="tabbar" aria-label="Modos de visualização">
        <button className={activeView === 'treino' ? 'active' : ''} onClick={() => setActiveView('treino')} type="button"><SlidersHorizontal size={16} /> Treino</button>
        <button className={activeView === 'teoria' ? 'active' : ''} onClick={() => setActiveView('teoria')} type="button"><BookOpen size={16} /> Teórico</button>
        <button className={activeView === 'programa' ? 'active' : ''} onClick={() => setActiveView('programa')} type="button"><Play size={16} /> Programa</button>
      </nav>

      {status && <div className="status-box">{status}</div>}

      {activeView === 'treino' && (
        <TrainingView
          form={trainingForm}
          result={trainingResult}
          loading={trainingLoading}
          onChange={handleTrainingFormChange}
          onTrain={trainPerceptron}
        />
      )}

      {activeView === 'teoria' && (
        <TheoryView
          content={content}
          activeData={activeData}
          predictions={predictions}
          weightsMatrix={weightsMatrix}
        />
      )}

      {activeView === 'programa' && (
        <ProgramView
          content={content}
          activeData={activeData}
          patterns={patterns}
          manualMatrix={manualMatrix}
          manualPrediction={manualPrediction}
          onMatrixChange={setManualMatrix}
          onPredict={runManualPrediction}
          onLoadPattern={loadPattern}
          onClearMatrix={clearMatrix}
        />
      )}

      <TrainingTopButton targetId="trabalho3-painel-superior" scopeSelector=".app-shell--task3" />

      {modalMessage && (
        <div className="modal-backdrop" role="presentation">
          <section aria-labelledby="modal-title" aria-modal="true" className="modal-card" role="dialog">
            <span className="tag">validação</span>
            <h3 id="modal-title">Não foi possível executar a ação</h3>
            <p>{modalMessage}</p>
            <div className="modal-actions">
              <button className="run-button" onClick={() => setModalMessage(null)} type="button">Fechar</button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
