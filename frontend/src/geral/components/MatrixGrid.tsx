import type { CSSProperties } from 'react';
import { formatNumber } from '../utils/formatNumber';

interface MatrixGridProps {
  readonly title: string;
  readonly subtitle?: string;
  readonly matrix: number[][];
  readonly compact?: boolean;
  readonly precision?: number;
  readonly onClick?: () => void;
  readonly selected?: boolean;
  readonly hint?: string;
}

export function MatrixGrid({ title, subtitle, matrix, compact = false, precision = 3, onClick, selected = false, hint }: MatrixGridProps) {
  const hasDecimalValues = matrix.some((row) => row.some((value) => ![-1, 0, 1].includes(value)));
  const columnCount = Math.max(1, ...matrix.map((row) => row.length));
  const cells = matrix.flatMap((row, rowIndex) =>
    row.map((value, colIndex) => (
      <span
        className={[
          'matrix-cell',
          hasDecimalValues ? 'matrix-cell--decimal' : '',
          value > 0 ? 'matrix-cell--positive' : value < 0 ? 'matrix-cell--negative' : 'matrix-cell--zero',
        ]
          .filter(Boolean)
          .join(' ')}
        key={`${title}-${rowIndex}-${colIndex}`}
      >
        {formatNumber(value, precision)}
      </span>
    )),
  );

  const cardClassName = [
    compact ? 'matrix-card matrix-card--compact' : 'matrix-card',
    hasDecimalValues ? 'matrix-card--decimal' : '',
    onClick ? 'matrix-card--interactive' : '',
    selected ? 'matrix-card--selected' : '',
  ]
    .filter(Boolean)
    .join(' ');

  const gridStyle = { '--matrix-cols': columnCount } as CSSProperties;

  const content = (
    <>
      <header className="matrix-card__header">
        <strong>{title}</strong>
        {subtitle && <span>{subtitle}</span>}
      </header>

      <div className="matrix-grid" aria-label={`Matriz ${title}`} role="grid" style={gridStyle}>
        {cells}
      </div>

      {hint && <small className="matrix-card__hint">{hint}</small>}
    </>
  );

  if (onClick) {
    return (
      <article className={cardClassName}>
        <button
          aria-label={`Selecionar matriz ${title}`}
          className="matrix-card__button"
          onClick={onClick}
          type="button"
        >
          {content}
        </button>
      </article>
    );
  }

  return (
    <article className={cardClassName}>
      {content}
    </article>
  );
}
