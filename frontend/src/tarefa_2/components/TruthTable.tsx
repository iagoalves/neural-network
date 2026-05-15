import type { HebbPrediction, LogicalFunction } from '../types/hebb';
import { formatNumber } from '../../geral/utils/formatNumber';

interface TruthTableProps {
  readonly logicalFunction: LogicalFunction;
  readonly predictions?: HebbPrediction[];
  readonly compact?: boolean;
}

export function TruthTable({ logicalFunction, predictions = [], compact = false }: TruthTableProps) {
  return (
    <div className={compact ? 'truth-table truth-table--compact' : 'truth-table'}>
      <table>
        <thead>
          <tr>
            <th>A</th>
            <th>B</th>
            <th>y</th>
            {!compact && <th>u</th>}
            {!compact && <th>ŷ</th>}
            {!compact && <th>Status</th>}
          </tr>
        </thead>
        <tbody>
          {logicalFunction.truthTable.map((row) => {
            const prediction = predictions.find((item) => item.row.row === row.row);
            return (
              <tr key={`${logicalFunction.id}-${row.row}`}>
                <td>{row.a}</td>
                <td>{row.b}</td>
                <td>{row.output}<small> ({row.target})</small></td>
                {!compact && <td>{prediction ? formatNumber(prediction.u) : '—'}</td>}
                {!compact && <td>{prediction ? `${prediction.predictedOutput} (${prediction.yHat})` : '—'}</td>}
                {!compact && (
                  <td>
                    {prediction ? (
                      <span className={prediction.isCorrect ? 'logic-pill logic-pill--ok' : 'logic-pill logic-pill--fail'}>
                        {prediction.isCorrect ? 'correto' : 'erro'}
                      </span>
                    ) : '—'}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
