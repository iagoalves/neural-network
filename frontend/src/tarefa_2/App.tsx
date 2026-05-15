import { useEffect, useMemo, useState } from 'react';
import { Binary, BookOpen, Play, SlidersHorizontal } from 'lucide-react';
import { fallbackContent } from './data/fallbackContent';
import { HebbApi } from './services/HebbApi';
import { TrainingView } from './views/TrainingView';
import { TheoryView } from './views/TheoryView';
import { ProgramView } from './views/ProgramView';
import { TrainingTopButton } from '../geral/components/TrainingTopButton';
import type { LearningContent, LogicalValue, ManualHebbPredictionResponse, TrainHebbRequest, TrainHebbResponse } from './types/hebb';
import './styles.css';

type ViewMode = 'treino' | 'teoria' | 'programa';

const api = new HebbApi();

const STORAGE_KEYS = {
  activeView: 'trabalho_2.hebb.activeView',
  trainingForm: 'trabalho_2.hebb.trainingForm',
  trainingResult: 'trabalho_2.hebb.trainingResult',
  selectedFunction: 'trabalho_2.hebb.selectedFunction',
} as const;

const DEFAULT_TRAINING_FORM: TrainHebbRequest = { initialBias: 0, initialWeight: 0, mode: 'hebb' };

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

function normalizeForm(value: Partial<TrainHebbRequest> | null | undefined): TrainHebbRequest {
  return {
    initialBias: typeof value?.initialBias === 'number' ? value.initialBias : DEFAULT_TRAINING_FORM.initialBias,
    initialWeight: typeof value?.initialWeight === 'number' ? value.initialWeight : DEFAULT_TRAINING_FORM.initialWeight,
    mode: 'hebb',
  };
}

function validateForm(form: TrainHebbRequest): string | null {
  if (form.initialBias < -10 || form.initialBias > 10) return 'O bias inicial deve ficar entre -10 e 10.';
  if (form.initialWeight < -10 || form.initialWeight > 10) return 'O peso inicial deve ficar entre -10 e 10.';
  return null;
}

export function App() {
  const [activeView, setActiveViewState] = useState<ViewMode>(() => readStorage<ViewMode>(STORAGE_KEYS.activeView, 'treino'));
  const [content, setContent] = useState<LearningContent>(fallbackContent);
  const [trainingForm, setTrainingForm] = useState<TrainHebbRequest>(DEFAULT_TRAINING_FORM);
  const [trainingResult, setTrainingResult] = useState<TrainHebbResponse | null>(() => readStorage<TrainHebbResponse | null>(STORAGE_KEYS.trainingResult, null));
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [status, setStatus] = useState('Carregando backend Python...');
  const [modalMessage, setModalMessage] = useState<string | null>(null);
  const [selectedFunctionIndex, setSelectedFunctionIndex] = useState(() => readStorage<number>(STORAGE_KEYS.selectedFunction, 0));
  const [a, setA] = useState<LogicalValue>(0);
  const [b, setB] = useState<LogicalValue>(0);
  const [manualPrediction, setManualPrediction] = useState<ManualHebbPredictionResponse | null>(null);

  function setActiveView(next: ViewMode) {
    setActiveViewState(next);
    writeStorage(STORAGE_KEYS.activeView, next);
  }

  function handleTrainingFormChange(next: TrainHebbRequest) {
    const normalized = normalizeForm(next);
    setTrainingForm(normalized);
    writeStorage(STORAGE_KEYS.trainingForm, normalized);
  }

  function handleSelectFunction(index: number) {
    setSelectedFunctionIndex(index);
    writeStorage(STORAGE_KEYS.selectedFunction, index);
    setManualPrediction(null);
  }

  useEffect(() => {
    let cancelled = false;
    const cachedForm = normalizeForm(readStorage<Partial<TrainHebbRequest>>(STORAGE_KEYS.trainingForm, {}));
    const startupForm = validateForm(cachedForm) ? DEFAULT_TRAINING_FORM : cachedForm;
    setTrainingForm(startupForm);
    writeStorage(STORAGE_KEYS.trainingForm, startupForm);

    Promise.all([api.getLearningContent(), api.train(startupForm)])
      .then(([learning, result]) => {
        if (cancelled) return;
        setContent(learning);
        setTrainingResult(result);
        writeStorage(STORAGE_KEYS.trainingResult, result);
        setStatus('');
      })
      .catch(() => {
        if (cancelled) return;
        setContent(fallbackContent);
        setStatus('Backend não encontrado. Inicie a API Python na porta 8787 para executar o trabalho_2.');
      });

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedResult = useMemo(
    () => trainingResult?.results.find((item) => item.function.index === selectedFunctionIndex) ?? null,
    [trainingResult, selectedFunctionIndex],
  );

  async function trainHebb() {
    const next = normalizeForm(trainingForm);
    const error = validateForm(next);
    if (error) { setModalMessage(error); return; }

    setTrainingLoading(true);
    try {
      const result = await api.train(next);
      setTrainingForm(next);
      setTrainingResult(result);
      writeStorage(STORAGE_KEYS.trainingForm, next);
      writeStorage(STORAGE_KEYS.trainingResult, result);
      setManualPrediction(null);
      setStatus('');
    } catch (err) {
      setModalMessage(err instanceof Error ? err.message : 'Falha ao treinar as funções lógicas.');
    } finally {
      setTrainingLoading(false);
    }
  }

  async function runManualPrediction() {
    try {
      setManualPrediction(await api.predict(selectedFunctionIndex, a, b, trainingForm.initialBias, trainingForm.initialWeight));
    } catch (err) {
      setModalMessage(err instanceof Error ? err.message : 'Falha ao executar predição.');
    }
  }

  return (
    <main className="app-shell app-shell--task2">
      <section className="hero" id="trabalho2-painel-superior">
        <span className="hero__eyebrow"><Binary size={16} /> trabalho_2</span>
        <div className="hero__content">
          <div>
            <h1>{content.title}</h1>
            <p>{content.summary}</p>
          </div>
          <div className="hero__cards">
            <article className="hero-card"><strong>16 funções</strong><span>F0...F15 geradas pelas quatro saídas possíveis da tabela verdade.</span></article>
            <article className="hero-card"><strong>Hebb simples</strong><span>Treino com Δwi = y·xi e Δb = y em uma passagem.</span></article>
            <article className="hero-card hero-card--diagram">
              <strong>Fluxo lógico</strong>
              <svg className="hero-diagram" viewBox="0 0 360 72" role="img" aria-label="Fluxo hebbiano para funções lógicas">
                <rect x="4" y="18" width="58" height="36" rx="10" />
                <text x="33" y="41">A,B</text>
                <path d="M68 36 H104" />
                <rect x="108" y="18" width="88" height="36" rx="10" />
                <text x="152" y="41">Hebb</text>
                <path d="M202 36 H238" />
                <rect x="242" y="18" width="54" height="36" rx="10" />
                <text x="269" y="41">ŷ</text>
                <path d="M300 36 H340" />
                <text x="156" y="12">w ← w + yx</text>
                <text x="310" y="64">F0...F15</text>
              </svg>
            </article>
          </div>
        </div>
      </section>

      <nav className="tabbar" aria-label="Modos de visualização do trabalho 2">
        <button className={activeView === 'treino' ? 'active' : ''} onClick={() => setActiveView('treino')} type="button"><SlidersHorizontal size={16} /> Treino</button>
        <button className={activeView === 'teoria' ? 'active' : ''} onClick={() => setActiveView('teoria')} type="button"><BookOpen size={16} /> Teórico</button>
        <button className={activeView === 'programa' ? 'active' : ''} onClick={() => setActiveView('programa')} type="button"><Play size={16} /> Programa</button>
      </nav>

      {status && <div className="status-box">{status}</div>}

      {activeView === 'treino' && (
        <TrainingView form={trainingForm} result={trainingResult} loading={trainingLoading} onChange={handleTrainingFormChange} onTrain={trainHebb} />
      )}

      {activeView === 'teoria' && (
        <TheoryView content={content} result={trainingResult} />
      )}

      {activeView === 'programa' && (
        <ProgramView
          content={content}
          trainForm={trainingForm}
          trainResult={trainingResult}
          selectedFunctionIndex={selectedResult?.function.index ?? selectedFunctionIndex}
          a={a}
          b={b}
          manualPrediction={manualPrediction}
          onSelectFunction={handleSelectFunction}
          onChangeA={(value) => { setA(value); setManualPrediction(null); }}
          onChangeB={(value) => { setB(value); setManualPrediction(null); }}
          onPredict={runManualPrediction}
        />
      )}

      <TrainingTopButton targetId="trabalho2-painel-superior" scopeSelector=".app-shell--task2" />

      {modalMessage && (
        <div className="modal-backdrop" role="presentation">
          <section aria-labelledby="modal-title-task2" aria-modal="true" className="modal-card" role="dialog">
            <span className="tag">validação</span>
            <h3 id="modal-title-task2">Não foi possível executar a ação</h3>
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
