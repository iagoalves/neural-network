LEARNING_CONTENT = {
    "title": "trabalho_3 · Perceptron com correção de erro",
    "summary": "Reconhecimento de X e T com matrizes 5x5, pesos inicializados em 0.001, bias fixo e treinamento por correção de erro.",
    "theory": {
        "headline": "Treinamento supervisionado com correção de erro",
        "intro": "O exercício usa duas amostras de treino: X_Principal com y=1 e T_Principal com y=-1. Cada matriz 5x5 vira um vetor de 25 entradas. Os pesos começam em 0.001 e só são corrigidos quando a saída prevista difere da saída esperada.",
        "cards": [
            {
                "title": "1. Entradas",
                "text": "Cada célula da matriz vira uma entrada xi. Como a matriz é 5x5, o perceptron possui 25 entradas: x1...x25.",
            },
            {
                "title": "2. Inicialização",
                "text": "Todos os pesos w1...w25 começam com 0.001. Esse valor baixo evita começar exatamente em zero e permite acompanhar numericamente a influência das correções.",
            },
            {
                "title": "3. Correção de erro",
                "text": "Para cada amostra, calcula-se u e a saída prevista. Se ŷ for diferente de y, o erro é calculado por e = y - ŷ e os pesos recebem Δwi = e·xi.",
            },
            {
                "title": "4. Bias fixo",
                "text": "O bias não altera os pesos. Ele é somado ao final em u = b + Σ(xi·wi), deslocando a decisão para a região X ou T.",
            },
        ],
        "formula": "Correção: Δwi = (y - ŷ)·xi; wi ← wi + Δwi",
        "activation": "Ativação: ŷ = 1 se u ≥ 0; ŷ = -1 se u < 0",
        "decisionLine": "Classificação: u = b + Σ(xi·wi), com b fixo",
    },
    "program": {
        "headline": "Programa de classificação",
        "summary": "Edite a matriz e execute o perceptron treinado por correção de erro.",
    },
}
