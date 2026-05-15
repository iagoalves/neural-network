from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from app.tarefa_3.domain.pattern import MatrixPattern, PatternFactory


class PatternCsvReader:
    """Lê padrões 5x5 salvos em CSV.

    Formato esperado:
    id,label,target,row,c1,c2,c3,c4,c5
    """

    def read(self, path: Path) -> list[MatrixPattern]:
        rows_by_id: dict[str, dict] = {}

        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                identifier = row["id"]
                matrix_row = [int(row[f"c{index}"]) for index in range(1, 6)]

                if identifier not in rows_by_id:
                    rows_by_id[identifier] = {
                        "label": row["label"],
                        "target": int(row["target"]),
                        "rows": {},
                    }

                rows_by_id[identifier]["rows"][int(row["row"])] = matrix_row

        patterns: list[MatrixPattern] = []

        for identifier, payload in rows_by_id.items():
            matrix = [payload["rows"][row_index] for row_index in range(1, 6)]
            patterns.append(
                PatternFactory.create(
                    raw_data=matrix,
                    label=payload["label"],
                    target=payload["target"],
                    identifier=identifier,
                )
            )

        return patterns


class CsvPatternRepository:
    """Repositório base para padrões guardados em CSV."""

    def __init__(self, csv_path: Path, reader: PatternCsvReader | None = None) -> None:
        self.csv_path = csv_path
        self.reader = reader or PatternCsvReader()
        self._patterns = self.reader.read(csv_path)

    def get_all(self) -> list[MatrixPattern]:
        return list(self._patterns)

    def get_by_id(self, identifier: str) -> MatrixPattern | None:
        return next((pattern for pattern in self._patterns if pattern.identifier == identifier), None)

    def get_by_label(self, label: str) -> list[MatrixPattern]:
        return [pattern for pattern in self._patterns if pattern.label == label]


class TrainingPatternRepository(CsvPatternRepository):
    """Repositório das amostras que participam do treino supervisionado."""

    def __init__(self) -> None:
        super().__init__(Path(__file__).resolve().parents[1] / "data" / "training_patterns.csv")


class VerificationPatternRepository(CsvPatternRepository):
    """Repositório das matrizes fixas de verificação.

    X1...X4 e T1...T4 foram pré-geradas e salvas em CSV.
    Elas não participam do treino; servem para testar o modelo treinado.
    """

    def __init__(self) -> None:
        super().__init__(Path(__file__).resolve().parents[1] / "data" / "verification_patterns.csv")


class PatternRepository:
    """Fachada usada pela aplicação para acessar treino e amostras de verificação."""

    def __init__(self) -> None:
        self.training_patterns = TrainingPatternRepository()
        self.verification_patterns = VerificationPatternRepository()

    @property
    def pattern_x(self) -> MatrixPattern:
        pattern = self.training_patterns.get_by_id("X_Principal")
        if pattern is None:
            raise RuntimeError("X_Principal não encontrado no CSV de treino.")
        return pattern

    @property
    def pattern_t(self) -> MatrixPattern:
        pattern = self.training_patterns.get_by_id("T_Principal")
        if pattern is None:
            raise RuntimeError("T_Principal não encontrado no CSV de treino.")
        return pattern

    def get_training_patterns(self) -> list[MatrixPattern]:
        return self.training_patterns.get_all()

    def get_verification_patterns(self) -> list[MatrixPattern]:
        return self.verification_patterns.get_all()

    def get_all(self) -> list[MatrixPattern]:
        return self.get_training_patterns()

    def get_all_for_verification(self) -> list[MatrixPattern]:
        return self.get_verification_patterns()
