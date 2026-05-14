import { formatNumber } from '../utils/formatNumber';

interface MatrixGridProps {
  readonly title: string;
  readonly subtitle?: string;
  readonly matrix: number[][];
  readonly compact?: boolean;
  readonly onClick?: () => void;
  readonly selected?: boolean;
  readonly hint?: string;
}

export function MatrixGrid({ title, subtitle, matrix, compact = false, onClick, selected = false, hint }: MatrixGridProps) {
  const cells = matrix.flatMap((row, rowIndex) =>
    row.map((value, colIndex) => (
      <span
        className={value > 0 ? 'matrix-cell matrix-cell--positive' : value < 0 ? 'matrix-cell matrix-cell--negative' : 'matrix-cell matrix-cell--zero'}
        key={`${title}-${rowIndex}-${colIndex}`}
      >
        {formatNumber(value)}
      </span>
    )),
  );

  const cardClassName = [
    compact ? 'matrix-card matrix-card--compact' : 'matrix-card',
    onClick ? 'matrix-card--interactive' : '',
    selected ? 'matrix-card--selected' : '',
  ]
    .filter(Boolean)
    .join(' ');

  const content = (
    <>
      <header className="matrix-card__header">
        <strong>{title}</strong>
        {subtitle && <span>{subtitle}</span>}
      </header>

      <div className="matrix-grid" aria-label={`Matriz ${title}`} role="grid">
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
