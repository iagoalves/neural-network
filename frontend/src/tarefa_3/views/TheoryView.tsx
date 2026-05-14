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

  function renderTrainingCompleteCard() {
    return (
      <button
        className={`training-complete-card ${selectedTraining ? 'training-complete-card--selected' : ''}`}
        onClick={() => selectTraining('Treino completo Hebb')}
        type="button"
      >
        <header className="training-complete-card__header">
          <span className="tag">treino completo</span>
          <h4>X_Principal + T_Principal</h4>
          <p>
            Clique para ver a sequência completa da Regra de Hebb: primeiro X atualiza os pesos, depois T completa o vetor treinado.
          </p>
        </header>

        <div className="training-complete-card__matrices">
          {trainingPatterns.map((pattern) => (
            <MatrixGrid
              compact
              key={pattern.id}
              matrix={pattern.matrix}
              subtitle={`y = ${pattern.target}`}
              title={pattern.id}
            />
          ))}
        </div>
      </button>
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
          <h3>Treino Hebb simples com duas amostras</h3>
          <p>
            O treino usa apenas <strong>X_Principal</strong> e <strong>T_Principal</strong>. As demais matrizes ficam em CSV e servem para verificar o classificador treinado.
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
            Os pesos começam em zero. Na passagem por <strong>X_Principal</strong>, cada peso recebe a contribuição <strong>y·xi</strong> com <strong>y=1</strong>.
            Isso faz o valor de <strong>xi</strong> entrar no peso com o mesmo sinal da amostra X. Na passagem por <strong>T_Principal</strong>, cada peso recebe a contribuição
            <strong> y·xi</strong> com <strong>y=-1</strong>, então o mesmo <strong>xi</strong> passa a ser somado com sinal invertido.
          </p>
          <p>
            Em termos práticos, <strong>y</strong> decide a direção da atualização: quando <strong>y=1</strong>, o padrão reforça os pesos nas posições em que X aparece; quando
            <strong> y=-1</strong>, o padrão T empurra esses mesmos pesos para o lado oposto. Por isso o resultado final do treino pode ser escrito como
            <strong> w = 0 + (1·X_Principal) + (-1·T_Principal)</strong>.
          </p>
          <p>
            O bias é fixo e não altera os pesos. Ele entra somente na classificação, somando o mesmo valor ao resultado de todas as matrizes:
            <strong> u = b + Σ(xi·wi)</strong>. Um bias maior desloca a fronteira para favorecer saídas positivas; um bias menor desloca a fronteira para favorecer saídas negativas.
          </p>
          <strong className="theory-prose__summary">
            Depois do treino: se u≥0, a saída é X; se u&lt;0, a saída é T. Alterar o bias muda o valor final de u, mas não muda a matriz de pesos aprendida por Hebb.
          </strong>
        </div>
      </section>

      {activeData?.model && (
        <section className="theory-block">
          <header className="theory-block__header">
            <span className="tag">modelo aprendido</span>
            <h3>Pesos gerados pela Regra de Hebb</h3>
            <p>
              Os pesos são calculados em uma única passagem: w = 0 + (1·X_Principal) + (-1·T_Principal). O bias permanece fixo em <strong>{formatNumber(activeData.model.bias)}</strong> e só entra na classificação.
            </p>
          </header>

          <div className="training-summary-grid">
            <article className="training-summary-card"><span>modo</span><strong>Hebb simples</strong><small>sem taxa de aprendizagem</small></article>
            <article className="training-summary-card"><span>entradas</span><strong>25</strong><small>w1...w25</small></article>
            <article className="training-summary-card"><span>bias fixo</span><strong>{formatNumber(activeData.model.bias)}</strong><small>somado em u</small></article>
            <article className="training-summary-card"><span>passagem</span><strong>{activeData.model.epochs}</strong><small>uma vez pelo conjunto</small></article>
          </div>

          <div className="patterns-grid patterns-grid--main">
            <MatrixGrid title="Pesos treinados" subtitle="w1...w25" matrix={weightsMatrix} />
          </div>
        </section>
      )}

      <section className="theory-block">
        <header className="theory-block__header">
          <span className="tag">treino Hebb</span>
          <h3>Amostras usadas no treino</h3>
          <p>
            O treino é mostrado como um único processo: X_Principal contribui primeiro e T_Principal contribui em seguida. Clique no card para abrir o passo a passo completo.
          </p>
        </header>

        {renderTrainingCompleteCard()}
      </section>

      {selection && (
        <div className="sample-calculation-target" ref={detailsRef}>
          {selectedTraining ? (
            <TrainingStepDetails
              sampleId="Treino completo Hebb"
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
