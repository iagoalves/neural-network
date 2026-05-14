import { ArrowRight, Binary, BookOpen, Play, SlidersHorizontal } from 'lucide-react';
import type { LearningContent, SamplesResponse } from '../types/perceptron';
import { formatNumber } from '../utils/formatNumber';

type ProjectView = 'treino' | 'teoria' | 'programa';

interface HomeViewProps {
  readonly content: LearningContent;
  readonly activeData: SamplesResponse | null;
  readonly onNavigate: (view: ProjectView) => void;
}

const INPUT_NODES = [
  { label: 'w1', x: 16, y: 18 },
  { label: 'w2', x: 16, y: 34 },
  { label: 'w3', x: 16, y: 50 },
  { label: 'w4', x: 16, y: 66 },
  { label: 'wn', x: 16, y: 82 },
] as const;

const OUTPUT_NODES = [
  { label: 'X', x: 84, y: 34 },
  { label: 'T', x: 84, y: 66 },
] as const;

function formatViewValue(value: number | undefined, fallback: string) {
  return typeof value === 'number' ? formatNumber(value) : fallback;
}

export function HomeView({ content, activeData, onNavigate }: HomeViewProps) {
  const model = activeData?.model;

  const liveStats = model
    ? [
        { label: 'modo', value: 'Hebb simples', note: 'uma passagem' },
        { label: 'bias', value: formatNumber(model.bias), note: 'fixo na ativação' },
        { label: 'épocas', value: String(model.epochs), note: 'treino concluído' },
        { label: 'amostras', value: String(activeData.predictions.length), note: 'verificação X/T' },
      ]
    : null;

  return (
    <div className="home-view">
      <section className="hero hero--home">
        <span className="hero__eyebrow"><Binary size={16} /> rede neural bipolar</span>

        <div className="hero__content">
          <div>
            <h1>{content.title}</h1>
            <p>
              {content.summary} A página inicial foi desenhada como um laboratório de rede neural:
              entrada 5x5, soma ponderada, ativação bipolar e saída X/T em um fluxo visual único.
            </p>

            <div className="home-view__actions">
              <button className="btn btn-light" onClick={() => onNavigate('treino')} type="button">
                <SlidersHorizontal size={16} />
                Entrar no treino
              </button>
              <button className="btn btn-outline-light" onClick={() => onNavigate('teoria')} type="button">
                <BookOpen size={16} />
                Ver teoria
              </button>
              <button className="btn btn-outline-light" onClick={() => onNavigate('programa')} type="button">
                <Play size={16} />
                Abrir programa
              </button>
            </div>

            <div className="hero__cards">
              <article className="hero-card">
                <strong>25 entradas</strong>
                <span>Cada matriz 5x5 vira x1...x25 antes da classificação.</span>
              </article>
              <article className="hero-card">
                <strong>Hebb simples</strong>
                <span>Os pesos são ajustados em uma única passagem pelos padrões principais.</span>
              </article>
              <article className="hero-card">
                <strong>Saída bipolar</strong>
                <span>ŷ = 1 representa X e ŷ = -1 representa T.</span>
              </article>
            </div>
          </div>

          <div className="home-hero__visual">
            <div className="neural-visual">
              <svg className="neural-visual__connections" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                {INPUT_NODES.map((node) => (
                  <line key={`input-${node.label}`} x1={node.x} y1={node.y} x2={50} y2={50} />
                ))}
                {OUTPUT_NODES.map((node) => (
                  <line key={`output-${node.label}`} x1={50} y1={50} x2={node.x} y2={node.y} />
                ))}
              </svg>

              {INPUT_NODES.map((node) => (
                <div
                  className="home-network__node home-network__node--input"
                  key={node.label}
                  style={{ left: `${node.x}%`, top: `${node.y}%` }}
                >
                  <small>entrada</small>
                  <strong>{node.label}</strong>
                </div>
              ))}

              <div className="home-network__node home-network__node--core" style={{ left: '50%', top: '48%' }}>
                <small className="home-network__node-math">u</small>
                <span>Σ</span>
                <small className="home-network__node-math">xi·wi</small>
              </div>

              {OUTPUT_NODES.map((node) => (
                <div
                  className="home-network__node home-network__node--output"
                  key={node.label}
                  style={{ left: `${node.x}%`, top: `${node.y}%` }}
                >
                  <strong>{node.label}</strong>
                  <small className="home-network__node-activation">{node.label === 'X' ? 'ŷ = 1' : 'ŷ = -1'}</small>
                </div>
              ))}

              <div className="home-network__formula" aria-hidden="true">
                <strong>u = b + Σ(xi·wi)</strong>
              </div>
            </div>
          </div>
        </div>
      </section>

      {liveStats && (
        <section className="theory-block home-stage home-view__stats">
          <div className="home-stage__header">
            <span className="tag">estado atual</span>
            <h3>O que a rede já aprendeu</h3>
            <p>
              Estes valores vêm do backend atual e resumem o modelo treinado pela Regra de Hebb simples.
            </p>
          </div>

          <div className="training-summary-grid">
            {liveStats.map((item) => (
              <article className="training-summary-card" key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                <small>{item.note}</small>
              </article>
            ))}
          </div>
        </section>
      )}

      <section className="theory-block home-stage">
        <div className="home-stage__header">
          <span className="tag">mapa neural</span>
          <h3>{content.theory.headline}</h3>
          <p>{content.theory.intro}</p>
        </div>

        <div className="formula-strip">
          <code>{content.theory.formula}</code>
          <code>{content.theory.activation}</code>
          <code>{content.theory.decisionLine}</code>
        </div>

        <div className="concept-grid">
          {content.theory.cards.map((card) => (
            <article className="concept-card" key={card.title}>
              <h3>{card.title}</h3>
              <p>{card.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="theory-block home-stage">
        <div className="home-stage__header">
          <span className="tag">atalhos</span>
          <h3>Entradas rápidas para o trabalho</h3>
          <p>
            Use estes atalhos para sair da página inicial e entrar na área principal do perceptron.
          </p>
        </div>

        <div className="home-route-grid">
          <button className="home-route-card home-route-card--treino" onClick={() => onNavigate('treino')} type="button">
            <div className="home-route-card__top">
              <span className="home-route-card__icon"><SlidersHorizontal size={18} /></span>
              <span className="home-route-card__arrow"><ArrowRight size={16} /></span>
            </div>
            <strong>Treino</strong>
            <p>Altere o bias e execute a única passagem da Regra de Hebb simples.</p>
          </button>

          <button className="home-route-card home-route-card--teoria" onClick={() => onNavigate('teoria')} type="button">
            <div className="home-route-card__top">
              <span className="home-route-card__icon"><BookOpen size={18} /></span>
              <span className="home-route-card__arrow"><ArrowRight size={16} /></span>
            </div>
            <strong>Teoria</strong>
            <p>Veja os pesos, a fronteira de ativação e o passo a passo da rede.</p>
          </button>

          <button className="home-route-card home-route-card--programa" onClick={() => onNavigate('programa')} type="button">
            <div className="home-route-card__top">
              <span className="home-route-card__icon"><Play size={18} /></span>
              <span className="home-route-card__arrow"><ArrowRight size={16} /></span>
            </div>
            <strong>Programa</strong>
            <p>Monte uma matriz manual e veja como o classificador responde.</p>
          </button>
        </div>
      </section>

      <section className="theory-block home-stage">
        <div className="home-stage__header">
          <span className="tag">referência</span>
          <h3>Paleta do modelo</h3>
          <p>
            A rede trabalha com sinais bipolares. Estes valores ajudam a visualizar a decisão final do perceptron.
          </p>
        </div>

        <div className="training-summary-grid">
          <article className="training-summary-card">
            <span>classe X</span>
            <strong>+1</strong>
            <small>saída ativada</small>
          </article>
          <article className="training-summary-card">
            <span>classe T</span>
            <strong>-1</strong>
            <small>saída inibida</small>
          </article>
          <article className="training-summary-card">
            <span>soma</span>
            <strong>Σ</strong>
            <small>combinação dos pesos</small>
          </article>
          <article className="training-summary-card">
            <span>bias</span>
            <strong>{formatViewValue(model?.bias, '1')}</strong>
            <small>desloca a decisão</small>
          </article>
        </div>
      </section>
    </div>
  );
}
