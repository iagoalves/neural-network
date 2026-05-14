import { InfoSection } from '../components/InfoSection';
import { TrainingPanel } from '../components/TrainingPanel';
import type { TrainPerceptronRequest, TrainPerceptronResponse } from '../types/perceptron';

interface TrainingViewProps {
  readonly form: TrainPerceptronRequest;
  readonly result: TrainPerceptronResponse | null;
  readonly loading: boolean;
  readonly onChange: (form: TrainPerceptronRequest) => void;
  readonly onTrain: () => Promise<void>;
}

export function TrainingView({ form, result, loading, onChange, onTrain }: TrainingViewProps) {
  return (
    <InfoSection
      eyebrow="treino"
      title="Treinamento Hebb simples"
      description="Configure apenas o bias fixo e execute uma passagem pelos dois padrões principais."
    >
      <TrainingPanel
        form={form}
        onChange={onChange}
        onTrain={onTrain}
        result={result}
        loading={loading}
      />
    </InfoSection>
  );
}
