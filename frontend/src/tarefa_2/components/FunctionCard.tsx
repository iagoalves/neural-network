import type { FunctionTrainingResult } from '../types/hebb';
import { TruthTable } from './TruthTable';
import { formatNumber } from '../../geral/utils/formatNumber';

interface FunctionCardProps {
  readonly result: FunctionTrainingResult;
  readonly selected: boolean;
  readonly onSelect: (result: FunctionTrainingResult) => void;
}

export function FunctionCard({ result, selected, onSelect }: FunctionCardProps) {
  return (
    <article className={selected ? 'logic-function-card logic-function-card--selected' : 'logic-function-card'}>
      <button onClick={() => onSelect(result)} type="button">
        <header>
          <span className="tag">{result.function.id}</span>
          <div>
            <strong>{result.function.name}</strong>
            <small>{result.function.expression} · {result.function.bitPattern}</small>
          </div>
        </header>

        <TruthTable compact logicalFunction={result.function} />

        <div className="logic-function-card__stats">
          <span>{result.correctCount}/{result.totalCount} acertos</span>
          <span>{formatNumber(result.accuracy * 100)}%</span>
          <span className={result.isExact ? 'logic-pill logic-pill--ok' : 'logic-pill logic-pill--fail'}>
            {result.isExact ? 'perfeita' : 'imperfeita'}
          </span>
        </div>
      </button>
    </article>
  );
}
