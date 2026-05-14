import type {
  LearningContent,
  Matrix5x5,
  PredictionResult,
  SamplesResponse,
  TargetValue,
  TrainPerceptronRequest,
  TrainPerceptronResponse,
} from '../types/perceptron';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8787';

export class PerceptronApi {
  async getLearningContent(): Promise<LearningContent> {
    return this.getJson<LearningContent>('/api/learning-content');
  }

  async getSamples(): Promise<SamplesResponse> {
    return this.getJson<SamplesResponse>('/api/samples');
  }

  async train(request: TrainPerceptronRequest): Promise<TrainPerceptronResponse> {
    const response = await fetch(`${API_BASE_URL}/api/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(await this.readErrorMessage(response, 'Falha ao treinar o perceptron.'));
    }

    return response.json() as Promise<TrainPerceptronResponse>;
  }

  async predict(matrix: Matrix5x5, target: TargetValue = 1): Promise<PredictionResult> {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ matrix, target, identifier: 'Entrada_Manual' }),
    });

    if (!response.ok) {
      throw new Error(await this.readErrorMessage(response, 'Falha ao executar predição.'));
    }

    return response.json() as Promise<PredictionResult>;
  }

  async getCsv(kind: 'patterns' | 'samples' | 'training' | 'predictions'): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/api/csv/${kind}`);

    if (!response.ok) {
      throw new Error('Falha ao carregar CSV.');
    }

    return response.text();
  }

  private async getJson<T>(path: string): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${path}`);

    if (!response.ok) {
      throw new Error(`Falha ao carregar ${path}.`);
    }

    return response.json() as Promise<T>;
  }

  private async readErrorMessage(response: Response, fallback: string): Promise<string> {
    try {
      const payload = await response.json() as { detail?: string | Array<{ msg?: string }> };

      if (typeof payload.detail === 'string' && payload.detail.trim()) {
        return payload.detail;
      }

      if (Array.isArray(payload.detail)) {
        const messages = payload.detail
          .map((item) => item.msg?.trim())
          .filter((message): message is string => Boolean(message));

        if (messages.length > 0) {
          return messages.join(' ');
        }
      }
    } catch {
      return fallback;
    }

    return fallback;
  }
}
