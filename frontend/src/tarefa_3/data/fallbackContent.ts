import type { LearningContent } from '../types/perceptron';

export const fallbackContent: LearningContent = {
  title: 'trabalho_3 · Perceptron com correção de erro',
  summary: 'Reconhecimento de X e T com matrizes 5x5, pesos inicializados em 0.001, bias fixo e atualização por erro.',
  theory: {
    headline: 'Treinamento supervisionado com correção de erro',
    intro: 'Cada matriz 5x5 vira um vetor com 25 entradas. O perceptron calcula u, compara ŷ com y e corrige os pesos quando há erro.',
    cards: [
      { title: '1. Entradas', text: 'Cada célula vira xi; a matriz 5x5 gera x1...x25.' },
      { title: '2. Pesos iniciais', text: 'Todos os pesos começam em 0.001.' },
      { title: '3. Correção', text: 'erro = y - ŷ; Δwi = erro·xi; wi ← wi + Δwi.' },
      { title: '4. Bias', text: 'O bias é fixo e entra em u = b + Σ(xi·wi).' },
    ],
    formula: 'Correção: Δwi = (y - ŷ)·xi; wi ← wi + Δwi',
    activation: 'Ativação: ŷ = 1 se u ≥ 0; ŷ = -1 se u < 0',
    decisionLine: 'Classificação: u = b + Σ(xi·wi)',
  },
  program: {
    headline: 'Programa de classificação',
    summary: 'Edite a matriz e execute o perceptron treinado por correção de erro.',
  },
};
