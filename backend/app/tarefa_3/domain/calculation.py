from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementCalculation:
    """Multiplicação de uma posição da matriz pela posição equivalente do vetor de pesos."""

    row: int
    col: int
    input_value: int
    weight: float

    @property
    def product(self) -> float:
        return self.input_value * self.weight

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "inputValue": self.input_value,
            "weight": self.weight,
            "product": self.product,
        }
