import type {
  LearningContent,
  LogicalFunction,
  LogicalValue,
  ManualHebbPredictionResponse,
  TrainHebbRequest,
  TrainHebbResponse,
} from '../types/hebb';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

export class HebbApi {
  async getLearningContent(): Promise<LearningContent> {
    return this.getJson<LearningContent>('/api/trabalho2/learning-content');
  }

  async getFunctions(): Promise<LogicalFunction[]> {
    const payload = await this.getJson<{ functions: LogicalFunction[] }>('/api/trabalho2/functions');
    return payload.functions;
  }

  async train(request: TrainHebbRequest): Promise<TrainHebbResponse> {
    const response = await fetch(this.buildUrl('/api/trabalho2/train'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(await this.readErrorMessage(response, 'Falha ao treinar as funções lógicas.'));
    }

    return response.json() as Promise<TrainHebbResponse>;
  }

  async predict(functionIndex: number, a: LogicalValue, b: LogicalValue, initialBias: number, initialWeight: number): Promise<ManualHebbPredictionResponse> {
    const response = await fetch(this.buildUrl('/api/trabalho2/predict'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ functionIndex, a, b, initialBias, initialWeight }),
    });

    if (!response.ok) {
      throw new Error(await this.readErrorMessage(response, 'Falha ao executar a predição hebbiana.'));
    }

    return response.json() as Promise<ManualHebbPredictionResponse>;
  }

  async getCsv(kind: 'functions' | 'training' | 'predictions'): Promise<string> {
    const response = await fetch(this.buildUrl(`/api/trabalho2/csv/${kind}`));
    if (!response.ok) throw new Error('Falha ao carregar CSV.');
    return response.text();
  }

  private async getJson<T>(path: string): Promise<T> {
    const response = await fetch(this.buildUrl(path));
    if (!response.ok) throw new Error(`Falha ao carregar ${path}.`);
    return response.json() as Promise<T>;
  }

  private async readErrorMessage(response: Response, fallback: string): Promise<string> {
    try {
      const payload = await response.json() as { detail?: string | Array<{ msg?: string }> };
      if (typeof payload.detail === 'string' && payload.detail.trim()) return payload.detail;
      if (Array.isArray(payload.detail)) {
        const messages = payload.detail.map((item) => item.msg?.trim()).filter((message): message is string => Boolean(message));
        if (messages.length > 0) return messages.join(' ');
      }
    } catch {
      return fallback;
    }
    return fallback;
  }

  private buildUrl(path: string): string {
    return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
  }
}
