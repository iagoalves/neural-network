import { useEffect, useMemo, useState } from 'react';
import { Binary, BookOpen, Home, Menu, Play, SlidersHorizontal } from 'lucide-react';
import { fallbackContent } from './data/fallbackContent';
import { PerceptronApi } from './services/PerceptronApi';
import { HomeView } from './views/HomeView';
import { TrainingView } from './views/TrainingView';
import { TheoryView } from './views/TheoryView';
import { ProgramView } from './views/ProgramView';
import type {
  LearningContent,
  Matrix5x5,
  PredictionResult,
  SamplesResponse,
  TrainPerceptronRequest,
  TrainPerceptronResponse,
} from './types/perceptron';
import './styles.css';

type ViewMode = 'inicio' | 'treino' | 'teoria' | 'programa';

type ProjectView = Exclude<ViewMode, 'inicio'>;

const NAV_ITEMS: Array<{
  label: string;
  view: ViewMode;
  icon: typeof Home;
}> = [
  { label: 'Início', view: 'inicio', icon: Home },
  { label: 'Treino', view: 'treino', icon: SlidersHorizontal },
  { label: 'Teoria', view: 'teoria', icon: BookOpen },
  { label: 'Programa', view: 'programa', icon: Play },
];

function isViewMode(value: unknown): value is ViewMode {
  return value === 'inicio' || value === 'treino' || value === 'teoria' || value === 'programa';
}

const api = new PerceptronApi();

const EMPTY_MATRIX: Matrix5x5 = [
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1],
];

const STORAGE_KEYS = {
  activeView: 'trabalho_3.hebb_simples.activeView',
  trainingForm: 'trabalho_3.hebb_simples.trainingForm',
  trainingResult: 'trabalho_3.hebb_simples.trainingResult',
  trainingCsv: 'trabalho_3.hebb_simples.trainingCsv',
} as const;

const DEFAULT_TRAINING_FORM: TrainPerceptronRequest = { initialBias: 1, mode: 'hebb' };

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
    mode: 'hebb',
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
  const [activeView, setActiveViewState] = useState<ViewMode>(() => {
    const cached = readStorage<unknown>(STORAGE_KEYS.activeView, 'inicio');
    return isViewMode(cached) ? cached : 'inicio';
  });
  const [content, setContent] = useState<LearningContent>(fallbackContent);
  const [samplesData, setSamplesData] = useState<SamplesResponse | null>(null);
  const [manualPrediction, setManualPrediction] = useState<PredictionResult | null>(null);
  const [manualMatrix, setManualMatrix] = useState<Matrix5x5>(EMPTY_MATRIX);
  const [status, setStatus] = useState('Carregando backend Python...');
  const [trainingForm, setTrainingForm] = useState<TrainPerceptronRequest>(DEFAULT_TRAINING_FORM);
  const [trainingResult, setTrainingResult] = useState<TrainPerceptronResponse | null>(null);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [modalMessage, setModalMessage] = useState<string | null>(null);

  function scrollToTop() {
    const prefersReducedMotion =
      typeof window.matchMedia === 'function'
        && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    window.scrollTo({ top: 0, behavior: prefersReducedMotion ? 'auto' : 'smooth' });
  }

  function setActiveView(next: ViewMode) {
    setActiveViewState(next);
    writeStorage(STORAGE_KEYS.activeView, next);
    setMenuOpen(false);
    scrollToTop();
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

  useEffect(() => {
    if (!menuOpen) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setMenuOpen(false);
      }
    }

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [menuOpen]);

  const activeData = trainingResult ?? samplesData;
  const patterns = activeData?.samples ?? [];
  const predictions = activeData?.predictions ?? [];
  const weightsMatrix = useMemo(() => weightsToMatrix(activeData?.model.weights ?? []), [activeData]);
  const isProjectView = activeView !== 'inicio';

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
    <main className="app-page">
      <nav className="navbar app-navbar navbar-dark">
        <div className="app-navbar__inner">
          <button
            className="app-navbar__brand"
            aria-label="Voltar para a página inicial"
            onClick={() => setActiveView('inicio')}
            type="button"
          >
            <span className="app-navbar__brand-mark"><Binary size={18} /></span>
            <span className="app-navbar__brand-copy">
              <strong>trabalho_3</strong>
              <small>perceptron X/T</small>
            </span>
          </button>

          <div className="app-navbar__links" aria-label="Navegação principal" role="navigation">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;

              return (
                <button
                  aria-pressed={activeView === item.view}
                  className={`nav-link app-navbar__link ${activeView === item.view ? 'active' : ''}`}
                  key={item.view}
                  onClick={() => setActiveView(item.view)}
                  type="button"
                >
                  <Icon size={16} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>

          <button
            aria-controls="app-offcanvas"
            aria-expanded={menuOpen}
            aria-label="Abrir menu lateral"
            className="btn btn-outline-light app-navbar__menu"
            onClick={() => setMenuOpen(true)}
            type="button"
          >
            <Menu size={18} />
            <span>Menu</span>
          </button>
        </div>
      </nav>

      <div className="app-shell app-shell--content">
        {status && <div className="status-box app-status">{status}</div>}

        {activeView === 'inicio' ? (
          <HomeView content={content} activeData={activeData} onNavigate={setActiveView} />
        ) : activeView === 'treino' ? (
          <TrainingView
            form={trainingForm}
            result={trainingResult}
            loading={trainingLoading}
            onChange={handleTrainingFormChange}
            onTrain={trainPerceptron}
          />
        ) : activeView === 'teoria' ? (
          <TheoryView
            content={content}
            activeData={activeData}
            predictions={predictions}
            weightsMatrix={weightsMatrix}
          />
        ) : (
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
      </div>

      {menuOpen && (
        <>
          <div className="offcanvas-backdrop fade show app-offcanvas__backdrop" onClick={() => setMenuOpen(false)} />
          <aside
            aria-labelledby="app-offcanvas-label"
            aria-modal="true"
            className="offcanvas offcanvas-start show app-offcanvas"
            id="app-offcanvas"
            role="dialog"
            tabIndex={-1}
          >
            <div className="offcanvas-header app-offcanvas__header">
              <div>
                <span className="tag">menu</span>
                <h2 id="app-offcanvas-label">Atalhos rápidos</h2>
              </div>
              <button
                aria-label="Fechar menu"
                className="btn-close btn-close-white"
                onClick={() => setMenuOpen(false)}
                type="button"
              />
            </div>

            <div className="offcanvas-body app-offcanvas__body">
              <button className={`app-offcanvas__link ${activeView === 'inicio' ? 'active' : ''}`} onClick={() => setActiveView('inicio')} type="button">
                <Home size={18} />
                <span>Início</span>
              </button>
              <button className={`app-offcanvas__link ${isProjectView ? 'active' : ''}`} onClick={() => setActiveView('treino')} type="button">
                <Binary size={18} />
                <span>trabalho3_xt</span>
              </button>
            </div>
          </aside>
        </>
      )}

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
