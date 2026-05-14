from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

PatternValue = Literal[1, -1]
PatternLabel = Literal["X", "T"]
TargetValue = Literal[1, -1]
Matrix5x5 = list[list[PatternValue]]


@dataclass(frozen=True)
class MatrixPattern:
    """Representa um padrão 5x5 usando apenas os sinais 1 e -1."""

    matrix: Matrix5x5
    label: PatternLabel
    target: TargetValue
    identifier: str

    def __post_init__(self) -> None:
        normalized = [[int(value) for value in row] for row in self.matrix]
        object.__setattr__(self, "matrix", normalized)
        self._validate_shape()
        self._validate_values()
        self._validate_target()

    @property
    def vector(self) -> list[PatternValue]:
        """Retorna a matriz como vetor de 25 posições."""
        return [value for row in self.matrix for value in row]  # type: ignore[list-item]

    def _validate_shape(self) -> None:
        if len(self.matrix) != 5 or any(len(row) != 5 for row in self.matrix):
            raise ValueError(f"O padrão {self.identifier} deve possuir formato 5x5.")

    def _validate_values(self) -> None:
        allowed_values = {-1, 1}
        values = {value for row in self.matrix for value in row}
        if not values.issubset(allowed_values):
            raise ValueError(f"O padrão {self.identifier} deve conter apenas 1 e -1.")

    def _validate_target(self) -> None:
        if self.target not in (-1, 1):
            raise ValueError("O target deve ser 1 para X ou -1 para T.")

    def to_dict(self) -> dict:
        return {
            "id": self.identifier,
            "label": self.label,
            "target": self.target,
            "matrix": self.matrix,
            "vector": self.vector,
        }


class PatternFactory:
    """Cria padrões matriciais de forma padronizada."""

    @staticmethod
    def create(raw_data: Iterable[Iterable[int]], label: PatternLabel, target: TargetValue, identifier: str) -> MatrixPattern:
        matrix = [[int(value) for value in row] for row in raw_data]
        return MatrixPattern(matrix=matrix, label=label, target=target, identifier=identifier)  # type: ignore[arg-type]
