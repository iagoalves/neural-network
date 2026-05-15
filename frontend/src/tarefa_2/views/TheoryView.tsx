import { useMemo, useState } from 'react';
import { InfoSection } from '../../geral/components/InfoSection';
import type { FunctionTrainingResult, LearningContent, TrainHebbResponse } from '../types/hebb';
import { FunctionCard } from '../components/FunctionCard';
import { HebbTrainingDetails } from '../components/HebbTrainingDetails';
import { formatNumber } from '../../geral/utils/formatNumber';

interface TheoryViewProps {
  readonly content: LearningContent;
  readonly result: TrainHebbResponse | null;
}

function sortByIndex(first: FunctionTrainingResult, second: FunctionTrainingResult) {
  return first.function.index - second.function.index;
}

export function TheoryView({ content, result }: TheoryViewProps) {
  const [selectedId, setSelectedId] = useState<string>('F0');
  const results = result?.results ?? [];
  const selected = results.find((item) => item.function.id === selectedId) ?? results[0] ?? null;

  const exactResults = useMemo(() => results.filter((item) => item.isExact).sort(sortByIndex), [results]);
  const imperfectResults = useMemo(() => results.filter((item) => !item.isExact).sort(sortByIndex), [results]);

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
          <h3>Classificação das 16 tabelas verdade</h3>
          <p>
            O programa enumera F0 até F15 pela ordem 00, 01, 10 e 11. Cada função é treinada separadamente com Hebb e depois verificada nas quatro entradas. Uma unidade linear simples acerta as funções separáveis e evidencia a limitação em XOR/XNOR.
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
            Para cada linha, o vetor de entrada é <strong>x=[A,B]</strong> em valores bipolares. O alvo <strong>y</strong> também é bipolar. A atualização não depende do erro: o alvo reforça os pesos quando y=+1 e inibe quando y=-1.
          </p>
          <p>
            A verificação final calcula <strong>u = b + A·wA + B·wB</strong>. Se <strong>u ≥ 0</strong>, a função prevê saída lógica 1; se <strong>u &lt; 0</strong>, prevê saída lógica 0.
          </p>
          <strong className="theory-prose__summary">
            Resultado com valores iniciais padrão: {result?.summary.exactFunctions ?? 14} funções classificadas perfeitamente e {result?.summary.imperfectFunctions ?? 2} imperfeitas. Casos críticos: {(result?.summary.imperfectFunctionIds ?? ['F6', 'F9']).join(', ')}.
          </strong>
        </div>
      </section>

      {result && (
        <section className="theory-block">
          <header className="theory-block__header">
            <span className="tag">resultado geral</span>
            <h3>Resumo da classificação</h3>
            <p>
              Cada card abaixo representa uma função lógica. Clique em uma função para ver a tabela verdade, os pesos finais e o passo a passo do treinamento.
            </p>
          </header>

          <div className="training-summary-grid">
            <article className="training-summary-card"><span>funções</span><strong>{result.summary.totalFunctions}</strong><small>F0...F15</small></article>
            <article className="training-summary-card"><span>perfeitas</span><strong>{result.summary.exactFunctions}</strong><small>100% de acerto</small></article>
            <article className="training-summary-card"><span>imperfeitas</span><strong>{result.summary.imperfectFunctions}</strong><small>{result.summary.imperfectFunctionIds.join(', ')}</small></article>
            <article className="training-summary-card"><span>acurácia média</span><strong>{formatNumber((result.results.reduce((sum, item) => sum + item.accuracy, 0) / result.results.length) * 100)}%</strong><small>sobre as 16 funções</small></article>
          </div>
        </section>
      )}

      {results.length > 0 && (
        <section className="theory-block">
          <header className="theory-block__header">
            <span className="tag">funções classificadas</span>
            <h3>Funções com classificação perfeita</h3>
            <p>Estas funções foram reproduzidas corretamente nas quatro linhas da tabela verdade.</p>
          </header>
          <div className="logic-function-grid">
            {exactResults.map((item) => (
              <FunctionCard key={item.function.id} result={item} selected={selected?.function.id === item.function.id} onSelect={(next) => setSelectedId(next.function.id)} />
            ))}
          </div>
        </section>
      )}

      {imperfectResults.length > 0 && (
        <section className="theory-block logic-warning-block">
          <header className="theory-block__header">
            <span className="tag">limitação</span>
            <h3>Funções não classificadas perfeitamente</h3>
            <p>XOR e XNOR não são linearmente separáveis em duas entradas por uma única unidade linear. A Regra de Hebb evidencia essa limitação.</p>
          </header>
          <div className="logic-function-grid logic-function-grid--small">
            {imperfectResults.map((item) => (
              <FunctionCard key={item.function.id} result={item} selected={selected?.function.id === item.function.id} onSelect={(next) => setSelectedId(next.function.id)} />
            ))}
          </div>
        </section>
      )}

      {selected && <HebbTrainingDetails result={selected} />}
    </InfoSection>
  );
}
