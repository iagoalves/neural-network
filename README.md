# trabalho_3 - Perceptron X/T com correção de erro

Projeto em Python/FastAPI e React/Vite para demonstrar um perceptron bipolar aplicado ao reconhecimento de X e T em matrizes 5x5.

## Lógica

- Cada matriz 5x5 vira um vetor `x1...x25`.
- Os pesos `w1...w25` começam com `0.001`.
- O bias é fixo e configurável.
- Para cada amostra rotulada, calcula-se `u = b + Σ(xi·wi)`.
- A ativação retorna `ŷ = 1` se `u >= 0`, e `ŷ = -1` caso contrário.
- Se houver erro, aplica-se a correção: `erro = y - ŷ`, `Δwi = erro·xi`, `wi ← wi + Δwi`.

## Rodar localmente

Backend:

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8787 --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Docker:

```bash
docker compose up --build
```

## Dados

- `backend/app/tarefa_3/data/training_patterns.csv`: X_Principal e T_Principal usados no treino.
- `backend/app/tarefa_3/data/verification_patterns.csv`: matrizes fixas usadas para verificação.
- `backend/app/tarefa_3/data/generated_training_log.csv`: log CSV gerado pelo backend.
- `backend/app/tarefa_3/data/generated_predictions.csv`: predições de referência.
