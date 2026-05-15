import { useMemo } from 'react';
import { TrainingStepDetails } from '../components/TrainingStepDetails';
import { MatrixGrid } from '../components/MatrixGrid';
import { DecisionPlane } from '../components/DecisionPlane';
import { InfoSection } from '../components/InfoSection';
import { useSelectedPrediction } from '../hooks/useSelectedPrediction';
import type { LearningContent, MatrixPattern, PredictionResult, SamplesResponse, TrainPerceptronResponse } from '../types/perceptron';
import { formatNumber } from '../utils/formatNumber';

interface TheoryViewProps {
  readonly content: LearningContent;
  readonly activeData: SamplesResponse | TrainPerceptronResponse | null;
  readonly predictions: PredictionResult[];
  readonly weightsMatrix: number[][];
}

function sampleSortValue(id: string) {
  if (id === 'X_Principal' || id === 'T_Principal') return 0;
  const numeric = Number(id.replace(/^[XT]/, ''));
  return Number.isFinite(numeric) ? numeric : 99;
}

function isTrainingSample(id: string) {
  return id === 'X_Principal' || id === 'T_Principal';
}

function findPattern(patterns: MatrixPattern[], id: string) {
  return patterns.find((pattern) => pattern.id === id) ?? null;
}

export function TheoryView({ content, activeData, predictions, weightsMatrix }: TheoryViewProps) {
  const {
    selection,
    selectedPrediction,
    selectedTraining,
    selectPrediction,
    selectTraining,
    targetRef: detailsRef,
  } = useSelectedPrediction(predictions);

  const trainingPatterns = useMemo(() => {
    const samples = activeData?.samples ?? [];
    return [findPattern(samples, 'X_Principal'), findPattern(samples, 'T_Principal')].filter(
      (pattern): pattern is MatrixPattern => Boolean(pattern),
    );
  }, [activeData]);

  const verificationPredictions = useMemo(
    () => predictions.filter((prediction) => !isTrainingSample(prediction.id)),
    [predictions],
  );

  const xVerificationPredictions = useMemo(
    () =>
      verificationPredictions
        .filter((prediction) => prediction.expectedLabel === 'X')
        .sort((first, second) => sampleSortValue(first.id) - sampleSortValue(second.id)),
    [verificationPredictions],
  );

  const tVerificationPredictions = useMemo(
    () =>
      verificationPredictions
        .filter((prediction) => prediction.expectedLabel === 'T')
        .sort((first, second) => sampleSortValue(first.id) - sampleSortValue(second.id)),
    [verificationPredictions],
  );

  function renderTrainingMatrixCard(pattern: MatrixPattern) {
    return (
      <MatrixGrid
        compact
        hint="Clique para ver o treino completo"
        matrix={pattern.matrix}
        onClick={() => selectTraining('Treino completo')}
        selected={selectedTraining?.id === pattern.id}
        subtitle={`y = ${pattern.target}`}
        title={pattern.id}
      />
    );
  }

  function renderVerificationMatrixCard(prediction: PredictionResult) {
    return (
      <MatrixGrid
        compact
        hint="Clique para ver a classificação"
        matrix={prediction.pattern.matrix}
        onClick={() => selectPrediction(prediction)}
        selected={selectedPrediction?.id === prediction.id}
        subtitle={`ŷ = ${prediction.yHat} · u = ${formatNumber(prediction.u)}`}
        title={prediction.id}
      />
    );
  }

  return (
    <InfoSection eyebrow="teórico" title={content.theory.headline} description={content.theory.intro}>
      <div className="formula-strip">
        <code>{content.theory.formula}</code>
        <code>{content.theory.activation}</code>
        <code>{content.theory.decisionLine}</code>
      </div>

      <section className="theory-block theory-block--prose">
        <header className="theory-block__header">
          <span className="tag">o que o programa faz</span>
          <h3>Treino supervisionado com correção de erro</h3>
          <p>
            O treino usa <strong>X_Principal</strong> e <strong>T_Principal</strong>. Os pesos começam em <strong>0.001</strong>, o bias é fixo e os pesos só são corrigidos quando a predição diverge do rótulo esperado.
          </p>
        </header>

        <div className="concept-grid">
          {content.theory.cards.map((card) => (
            <article className="concept-card" key={card.title}>
              <h3>{card.title}</h3>
              <p>{card.text}</p>
            </article>
          ))}
        </div>

        <div className="theory-prose">
          <p>
            O perceptron calcula <strong>u = b + Σ(xi·wi)</strong> e aplica a função de ativação. Se <strong>u ≥ 0</strong>, a saída prevista é <strong>ŷ=1</strong>; se <strong>u &lt; 0</strong>, a saída prevista é <strong>ŷ=-1</strong>.
          </p>
          <p>
            Durante o treino, compara-se a saída prevista <strong>ŷ</strong> com a saída esperada <strong>y</strong>. Quando há erro, calcula-se <strong>erro = y - ŷ</strong> e cada peso recebe a correção <strong>Δwi = erro·xi</strong>.
          </p>
          <p>
            O bias é fixo e não corrige os pesos. Ele é somado em <strong>u</strong>; portanto, aumentar o bias desloca as ativações para valores mais positivos e favorece classificações como X. Reduzir o bias desloca as ativações para valores mais negativos e favorece T.
          </p>
          <strong className="theory-prose__summary">
            Fórmula do treino: erro = y - ŷ; Δwi = erro·xi; wi ← wi + Δwi. A classificação final usa u = b + Σ(xi·wi).
          </strong>
        </div>
      </section>

      {activeData?.model && (
        <section className="theory-block">
          <header className="theory-block__header">
            <span className="tag">modelo aprendido</span>
            <h3>Pesos gerados pela correção de erro</h3>
            <p>
              Os pesos começam em <strong>{formatNumber(activeData.model.initialWeight)}</strong> e são corrigidos somente quando existe erro. O bias permanece fixo em <strong>{formatNumber(activeData.model.bias)}</strong>.
            </p>
          </header>

          <div className="training-summary-grid">
            <article className="training-summary-card"><span>modo</span><strong>Correção de erro</strong><small>supervisionado</small></article>
            <article className="training-summary-card"><span>entradas</span><strong>25</strong><small>w1...w25</small></article>
            <article className="training-summary-card"><span>bias fixo</span><strong>{formatNumber(activeData.model.bias)}</strong><small>somado em u</small></article>
            <article className="training-summary-card"><span>épocas</span><strong>{activeData.model.epochs}</strong><small>até convergir</small></article>
          </div>

          <div className="patterns-grid patterns-grid--main">
            <MatrixGrid title="Pesos treinados" subtitle="w1...w25" matrix={weightsMatrix} precision={8} />
          </div>
        </section>
      )}

      <section className="theory-block">
        <header className="theory-block__header">
          <span className="tag">treino</span>
          <h3>Amostras usadas no treino</h3>
          <p>
            Estes dois cards representam as amostras de treino. Clique em qualquer uma delas para abrir o passo a passo completo das correções executadas.
          </p>
        </header>

        <div className="training-sample-card-grid">
          {trainingPatterns.map((pattern) => renderTrainingMatrixCard(pattern))}
        </div>
      </section>

      <section className="theory-block">
        <header className="theory-block__header">
          <span className="tag">verificação</span>
          <h3>Verificação</h3>
          <p>
            As matrizes abaixo não alteram os pesos. Elas usam o modelo treinado para testar se a classificação final continua coerente.
          </p>
        </header>

        <div className="samples-groups">
          <section className="samples-group samples-group--x">
            <h4>Classe X</h4>
            <div className="sample-matrix-grid">
              {xVerificationPredictions.map((prediction) => renderVerificationMatrixCard(prediction))}
            </div>
          </section>

          <section className="samples-group samples-group--t">
            <h4>Classe T</h4>
            <div className="sample-matrix-grid">
              {tVerificationPredictions.map((prediction) => renderVerificationMatrixCard(prediction))}
            </div>
          </section>
        </div>

        {selection && (
          <div className="sample-calculation-target" ref={detailsRef}>
            {selectedTraining ? (
              <TrainingStepDetails
                sampleId="Treino completo"
                steps={activeData?.model.trainingSteps ?? []}
              />
            ) : selectedPrediction ? (
              <TrainingStepDetails
                prediction={selectedPrediction}
                sampleId={selectedPrediction.id}
                steps={[]}
              />
            ) : null}
          </div>
        )}
      </section>

      {predictions.length > 0 && (
        <section className="theory-block">
          <header className="theory-block__header">
            <span className="tag">fronteira de ativação</span>
            <h3>Regiões X e T</h3>
            <p>
              Todas as matrizes do CSV de verificação são projetadas no eixo u. A fronteira u=0 separa a região T da região X.
            </p>
          </header>
          <DecisionPlane predictions={predictions} />
        </section>
      )}
    </InfoSection>
  );
}
