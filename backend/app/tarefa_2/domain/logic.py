from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INPUT_ROWS: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, -1, -1),
    (0, 1, -1, 1),
    (1, 0, 1, -1),
    (1, 1, 1, 1),
)

FUNCTION_NAMES: dict[int, tuple[str, str]] = {
    0: ("ZERO", "0"),
    1: ("NOR", "¬A ∧ ¬B"),
    2: ("¬A ∧ B", "¬A ∧ B"),
    3: ("NOT A", "¬A"),
    4: ("A ∧ ¬B", "A ∧ ¬B"),
    5: ("NOT B", "¬B"),
    6: ("XOR", "A ⊕ B"),
    7: ("NAND", "¬(A ∧ B)"),
    8: ("AND", "A ∧ B"),
    9: ("XNOR", "A ↔ B"),
    10: ("B", "B"),
    11: ("A → B", "¬A ∨ B"),
    12: ("A", "A"),
    13: ("B → A", "A ∨ ¬B"),
    14: ("OR", "A ∨ B"),
    15: ("UM", "1"),
}


@dataclass(frozen=True)
class TruthRow:
    row: int
    a: int
    b: int
    a_bipolar: int
    b_bipolar: int
    output: int
    target: int

    @property
    def features(self) -> list[int]:
        return [self.a_bipolar, self.b_bipolar]

    def to_dict(self) -> dict[str, int]:
        return {
            "row": self.row,
            "a": self.a,
            "b": self.b,
            "aBipolar": self.a_bipolar,
            "bBipolar": self.b_bipolar,
            "output": self.output,
            "target": self.target,
        }


@dataclass(frozen=True)
class LogicalFunction:
    index: int
    identifier: str
    name: str
    expression: str
    truth_table: list[TruthRow]

    @property
    def bit_pattern(self) -> str:
        return "".join(str(row.output) for row in self.truth_table)

    @property
    def targets(self) -> list[int]:
        return [row.target for row in self.truth_table]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.identifier,
            "name": self.name,
            "expression": self.expression,
            "bitPattern": self.bit_pattern,
            "truthTable": [row.to_dict() for row in self.truth_table],
            "targets": self.targets,
        }


def build_logical_function(index: int) -> LogicalFunction:
    if index < 0 or index > 15:
        raise ValueError("A função lógica deve estar entre F0 e F15.")

    name, expression = FUNCTION_NAMES[index]
    rows: list[TruthRow] = []
    for row_index, (a, b, a_bipolar, b_bipolar) in enumerate(INPUT_ROWS, start=1):
        bit_index = row_index - 1
        output = 1 if (index >> bit_index) & 1 else 0
        target = 1 if output == 1 else -1
        rows.append(
            TruthRow(
                row=row_index,
                a=a,
                b=b,
                a_bipolar=a_bipolar,
                b_bipolar=b_bipolar,
                output=output,
                target=target,
            )
        )

    return LogicalFunction(
        index=index,
        identifier=f"F{index}",
        name=name,
        expression=expression,
        truth_table=rows,
    )


def build_all_functions() -> list[LogicalFunction]:
    return [build_logical_function(index) for index in range(16)]
