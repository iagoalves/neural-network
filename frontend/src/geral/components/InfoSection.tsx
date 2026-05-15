import type { ReactNode } from 'react';

interface InfoSectionProps {
  readonly eyebrow: string;
  readonly title: string;
  readonly description?: string;
  readonly children: ReactNode;
}

export function InfoSection({ eyebrow, title, description, children }: InfoSectionProps) {
  return (
    <section className="info-section">
      <header className="info-section__header">
        <span className="tag">{eyebrow}</span>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </header>

      {children}
    </section>
  );
}
