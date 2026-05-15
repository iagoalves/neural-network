import type { LearningContent } from '../types/hebb';

export const fallbackContent: LearningContent = {
  title: 'trabalho_2 · Regra de Hebb nas 16 funções lógicas',
  summary: 'Classificação de F0...F15 com entradas bipolares, treino hebbiano e verificação pela tabela verdade.',
  theory: {
    headline: 'Aprendizado hebbiano aplicado a funções lógicas',
    intro: 'Cada função lógica possui quatro saídas. O sistema converte valores lógicos para sinais bipolares e treina uma unidade linear com Hebb.',
    cards: [
      { title: '1. Entradas', text: '0 vira -1 e 1 vira +1 para A e B.' },
      { title: '2. Alvo', text: 'Saída 0 vira y=-1 e saída 1 vira y=+1.' },
      { title: '3. Hebb', text: 'Δwi = y·xi; wi ← wi + Δwi; Δb = y.' },
      { title: '4. Ativação', text: 'u = b + A·wA + B·wB; se u ≥ 0, ŷ=1.' },
    ],
    formula: 'Hebb: Δwi = y·xi; wi ← wi + Δwi; Δb = y',
    activation: 'Ativação: ŷ = 1 se u ≥ 0; ŷ = -1 se u < 0',
    decisionLine: 'Classificação: u = b + A·wA + B·wB',
  },
  program: {
    headline: 'Programa das 16 funções lógicas',
    summary: 'Selecione uma função e teste as entradas A/B contra o modelo treinado por Hebb.',
  },
};
