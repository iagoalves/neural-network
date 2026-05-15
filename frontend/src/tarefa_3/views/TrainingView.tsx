import { InfoSection } from '../../geral/components/InfoSection';
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
      title="Treinamento com correção de erro"
      description="Configure o bias fixo e execute o treino supervisionado com pesos inicializados em 0.001."
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
