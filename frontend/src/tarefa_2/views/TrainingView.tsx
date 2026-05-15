import { InfoSection } from '../../geral/components/InfoSection';
import { TrainingPanel } from '../components/TrainingPanel';
import type { TrainHebbRequest, TrainHebbResponse } from '../types/hebb';

interface TrainingViewProps {
  readonly form: TrainHebbRequest;
  readonly result: TrainHebbResponse | null;
  readonly loading: boolean;
  readonly onChange: (form: TrainHebbRequest) => void;
  readonly onTrain: () => Promise<void>;
}

export function TrainingView({ form, result, loading, onChange, onTrain }: TrainingViewProps) {
  return (
    <InfoSection
      eyebrow="treino"
      title="Treinamento com Regra de Hebb"
      description="Configure os valores iniciais e treine as 16 funções lógicas de duas entradas em uma passagem hebbiana."
    >
      <TrainingPanel form={form} onChange={onChange} onTrain={onTrain} result={result} loading={loading} />
    </InfoSection>
  );
}
