LEARNING_CONTENT = {
    "title": "trabalho_2 · Regra de Hebb nas 16 funções lógicas",
    "summary": "Classificação das 16 funções booleanas de duas variáveis com entradas bipolares, treinamento hebbiano e verificação das quatro linhas da tabela verdade.",
    "theory": {
        "headline": "Aprendizado hebbiano aplicado a funções lógicas",
        "intro": "Cada função lógica de duas entradas é descrita por quatro saídas possíveis. O programa converte A e B para sinais bipolares, treina uma unidade linear com a Regra de Hebb e verifica se a função foi classificada corretamente nas quatro combinações.",
        "cards": [
            {
                "title": "1. Entradas bipolares",
                "text": "O valor lógico 0 é representado por -1 e o valor lógico 1 é representado por +1. Assim, as quatro entradas são 00, 01, 10 e 11 em formato bipolar.",
            },
            {
                "title": "2. Saída esperada",
                "text": "Cada função F0...F15 possui uma tabela verdade própria. A saída 0 vira y=-1 e a saída 1 vira y=+1.",
            },
            {
                "title": "3. Regra de Hebb",
                "text": "Para cada linha da tabela verdade, os pesos são ajustados por Δwi = y·xi. O bias também é treinado por Δb = y.",
            },
            {
                "title": "4. Classificação final",
                "text": "Depois do treino, calcula-se u = b + A·wA + B·wB. Se u ≥ 0, a saída prevista é +1; caso contrário, é -1.",
            },
        ],
        "formula": "Hebb: Δwi = y·xi; wi ← wi + Δwi; Δb = y",
        "activation": "Ativação: ŷ = 1 se u ≥ 0; ŷ = -1 se u < 0",
        "decisionLine": "Classificação: u = b + A·wA + B·wB",
    },
    "program": {
        "headline": "Programa das 16 funções lógicas",
        "summary": "Selecione uma função, altere as entradas A/B e veja a predição calculada pelo modelo treinado com Hebb.",
    },
}
