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

O Vite em desenvolvimento usa proxy para o backend em `/api`, então nao e necessario definir `VITE_API_BASE_URL` para rodar localmente.

Docker desenvolvimento:

```bash
docker compose up --build
```

Nesse modo, o frontend roda com Vite em hot reload em `http://localhost:5173` e o backend roda com `uvicorn --reload` em `http://localhost:8787`. As portas ficam publicadas apenas em `127.0.0.1`.

Docker producao:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

Nesse modo, o frontend e buildado com Vite e servido por `nginx` em `http://localhost:5173`. O backend fica acessivel internamente no Compose e pode ser verificado externamente pelo proxy em `http://localhost:5173/api/health`.

## Dados

- `backend/app/tarefa_3/data/training_patterns.csv`: X_Principal e T_Principal usados no treino.
- `backend/app/tarefa_3/data/verification_patterns.csv`: matrizes fixas usadas para verificação.
- `backend/app/tarefa_3/data/generated_training_log.csv`: log CSV gerado pelo backend.
- `backend/app/tarefa_3/data/generated_predictions.csv`: predições de referência.
