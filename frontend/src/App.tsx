import { useMemo, useState } from 'react';
import { BookOpen, BrainCircuit, Menu, Network, Sparkles } from 'lucide-react';
import { App as Trabalho2App } from './tarefa_2/App';
import { App as Trabalho3App } from './tarefa_3/App';

type Trabalho = 'trabalho2' | 'trabalho3';

const STORAGE_KEY = 'neural_network.selectedWork';

const TRABALHOS: Array<{
  id: Trabalho;
  title: string;
  shortTitle: string;
  description: string;
  badge: string;
}> = [
  {
    id: 'trabalho2',
    title: 'Trabalho 2 · Regra de Hebb',
    shortTitle: 'Trabalho 2',
    description: 'Classificação das 16 funções lógicas com treinamento hebbiano.',
    badge: 'Hebb lógico',
  },
  {
    id: 'trabalho3',
    title: 'Trabalho 3 · Perceptron X/T',
    shortTitle: 'Trabalho 3',
    description: 'Reconhecimento de matrizes 5x5 usando perceptron e correção de erro.',
    badge: 'X/T 5x5',
  },
];

function readSelectedWork(): Trabalho {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw === 'trabalho3' ? 'trabalho3' : 'trabalho2';
  } catch {
    return 'trabalho2';
  }
}

export function App() {
  const [selectedWork, setSelectedWork] = useState<Trabalho>(readSelectedWork);
  const activeWork = useMemo(() => TRABALHOS.find((item) => item.id === selectedWork) ?? TRABALHOS[0], [selectedWork]);

  function changeWork(next: Trabalho) {
    setSelectedWork(next);
    window.localStorage.setItem(STORAGE_KEY, next);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  return (
    <div className="app-root">
      <nav className="navbar navbar-expand-lg bg-body-tertiary fixed-top border-bottom shadow-sm project-navbar" aria-label="Navegação principal">
        <div className="container-fluid project-navbar__inner">
          <button
            aria-controls="projectOffcanvas"
            className="btn btn-primary project-navbar__menu"
            data-bs-target="#projectOffcanvas"
            data-bs-toggle="offcanvas"
            type="button"
          >
            <Menu size={18} />
            Menu
          </button>

          <button className="navbar-brand project-navbar__brand" onClick={() => changeWork(selectedWork)} type="button">
            <BrainCircuit size={22} />
            <span>neural-network</span>
          </button>

          <div className="project-navbar__current" aria-live="polite">
            <span className="badge text-bg-primary-subtle border border-primary-subtle text-primary-emphasis">{activeWork.badge}</span>
            <strong>{activeWork.shortTitle}</strong>
          </div>
        </div>
      </nav>

      <aside
        aria-labelledby="projectOffcanvasLabel"
        className="offcanvas offcanvas-start project-offcanvas"
        id="projectOffcanvas"
        tabIndex={-1}
      >
        <div className="offcanvas-header project-offcanvas__header">
          <div>
            <span className="project-offcanvas__eyebrow"><Sparkles size={14} /> entradas do projeto</span>
            <h5 className="offcanvas-title" id="projectOffcanvasLabel">Selecionar trabalho</h5>
          </div>
          <button aria-label="Fechar" className="btn-close" data-bs-dismiss="offcanvas" type="button" />
        </div>

        <div className="offcanvas-body project-offcanvas__body">
          <p className="project-offcanvas__intro">
            Use este menu lateral para alternar entre os trabalhos sem perder a estrutura de treino, teoria e programa.
          </p>

          <div className="project-offcanvas__list" role="list">
            {TRABALHOS.map((work) => (
              <button
                className={`project-offcanvas__item ${selectedWork === work.id ? 'active' : ''}`}
                data-bs-dismiss="offcanvas"
                key={work.id}
                onClick={() => changeWork(work.id)}
                type="button"
              >
                <span className="project-offcanvas__icon">
                  {work.id === 'trabalho2' ? <BookOpen size={18} /> : <Network size={18} />}
                </span>
                <span>
                  <strong>{work.title}</strong>
                  <small>{work.description}</small>
                </span>
              </button>
            ))}
          </div>
        </div>
      </aside>

      {selectedWork === 'trabalho2' ? <Trabalho2App /> : <Trabalho3App />}
    </div>
  );
}
