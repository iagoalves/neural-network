import type { Matrix5x5, MatrixValue } from '../types/perceptron';

interface ProgramMatrixEditorProps {
  matrix: Matrix5x5;
  onChange: (nextMatrix: Matrix5x5) => void;
}

export function ProgramMatrixEditor({ matrix, onChange }: ProgramMatrixEditorProps) {
  function toggleCell(rowIndex: number, colIndex: number) {
    const nextMatrix = matrix.map((row, currentRowIndex) =>
      row.map((value, currentColIndex) => {
        if (currentRowIndex !== rowIndex || currentColIndex !== colIndex) {
          return value;
        }

        return (value === 1 ? -1 : 1) as MatrixValue;
      })
    ) as Matrix5x5;

    onChange(nextMatrix);
  }

  return (
    <div className="editor-grid" aria-label="Editor de matriz 5x5">
      {matrix.map((row, rowIndex) =>
        row.map((value, colIndex) => (
          <button
            className={value === 1 ? 'editor-cell editor-cell--active' : 'editor-cell'}
            key={`${rowIndex}-${colIndex}`}
            onClick={() => toggleCell(rowIndex, colIndex)}
            type="button"
          >
            {value}
          </button>
        )),
      )}
    </div>
  );
}
