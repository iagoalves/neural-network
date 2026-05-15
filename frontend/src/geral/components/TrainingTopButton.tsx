import { ArrowUp, EyeOff } from 'lucide-react';

interface TrainingTopButtonProps {
  readonly targetId: string;
  readonly scopeSelector: string;
}

export function TrainingTopButton({ targetId, scopeSelector }: TrainingTopButtonProps) {
  function handleClick() {
    const scope = document.querySelector(scopeSelector) ?? document;
    scope.querySelectorAll('details[open]').forEach((details) => {
      (details as HTMLDetailsElement).open = false;
    });

    const target = document.getElementById(targetId);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  return (
    <button
      aria-label="Ocultar detalhes do treino e voltar ao painel superior"
      className="training-top-button"
      onClick={handleClick}
      title="Ocultar detalhes do treino e voltar ao painel superior"
      type="button"
    >
      <span className="training-top-button__icon training-top-button__icon--hide"><EyeOff size={15} /></span>
      <span className="training-top-button__icon training-top-button__icon--up"><ArrowUp size={20} /></span>
    </button>
  );
}
