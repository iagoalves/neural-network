import type { LearningContent } from '../types/perceptron';

export const fallbackContent: LearningContent = {
  title: 'trabalho_3 · Regra de Hebb simples',
  summary: 'Reconhecimento de X e T com matrizes 5x5, treinamento Hebb em uma passagem e verificação das amostras geradas por seed.',
  theory: {
    headline: 'Regra de Hebb simples aplicada ao reconhecimento de X e T',
    intro: 'O exercício usa duas amostras de treino: X_Principal com y=1 e T_Principal com y=-1. Cada matriz 5x5 vira um vetor de 25 entradas.',
    cards: [
      { title: '1. Entradas', text: 'Cada célula da matriz vira uma entrada xi. Como a matriz é 5x5, o perceptron possui 25 entradas.' },
      { title: '2. Treino', text: 'Os pesos começam em zero e acumulam as contribuições y·xi das duas amostras principais.' },
      { title: '3. Hebb simples', text: 'A atualização é wi ← wi + y·xi. O valor de y define o sinal da contribuição: X usa y=1 e T usa y=-1. Não há taxa de aprendizagem e o bias não é atualizado.' },
      { title: '4. Bias fixo', text: 'O bias não altera os pesos. Ele é somado ao final em u = b + Σ(xi·wi), deslocando a decisão para X ou T.' },
    ],
    formula: 'Treino: wi ← wi + y·xi',
    activation: 'Ativação: ŷ = 1 se u ≥ 0; ŷ = -1 se u < 0',
    decisionLine: 'Classificação: u = b + Σ(xi·wi), com b fixo',
  },
  program: {
    headline: 'Programa de classificação',
    summary: 'Edite a matriz e execute o perceptron treinado pela Regra de Hebb simples.',
  },
};
