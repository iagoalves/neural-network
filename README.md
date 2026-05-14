# trabalho_3 · Perceptron X/T com Regra de Hebb simples

Projeto em Python/FastAPI e React/Vite para demonstrar a Regra de Hebb simples em um perceptron bipolar.

## Lógica usada

- Cada matriz 5x5 vira um vetor `x1...x25`.
- `X_Principal` usa `y = 1`.
- `T_Principal` usa `y = -1`.
- Os pesos começam em zero.
- O bias é configurável, mas fica fixo durante o treino.
- O treino realiza uma única passagem:
  - `wi ← wi + y·xi`
- A classificação usa:
  - `u = b + Σ(xi·wi)`
  - `ŷ = 1` se `u >= 0`
  - `ŷ = -1` se `u < 0`

## Dados CSV

Os padrões de treino e as amostras geradas por seed ficam em:

- `backend/app/tarefa_3/data/training_patterns.csv`
- `backend/app/tarefa_3/data/seed_samples.csv`

`X1...X4` e `T1...T4` são usadas para verificar os pesos treinados; elas não participam do treino.

## Rodar com Docker

```bash
docker compose up --build
```

Frontend: `http://localhost:5173`

Backend: `http://localhost:8787`

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
