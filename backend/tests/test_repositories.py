from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tarefa_3.domain.pattern import MatrixPattern
from app.tarefa_3.repositories.pattern_repository import CsvPatternRepository, PatternCsvReader, PatternRepository


CSV_CONTENT = """id,label,target,row,c1,c2,c3,c4,c5
X_Principal,X,1,1,1,-1,-1,-1,1
X_Principal,X,1,2,-1,1,-1,1,-1
X_Principal,X,1,3,-1,-1,1,-1,-1
X_Principal,X,1,4,-1,1,-1,1,-1
X_Principal,X,1,5,1,-1,-1,-1,1
T_Principal,T,-1,1,1,1,1,1,1
T_Principal,T,-1,2,-1,-1,1,-1,-1
T_Principal,T,-1,3,-1,-1,1,-1,-1
T_Principal,T,-1,4,-1,-1,1,-1,-1
T_Principal,T,-1,5,-1,-1,1,-1,-1
"""


class TestRepositories:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "patterns.csv"
        self.csv_path.write_text(CSV_CONTENT, encoding="utf-8")

    def teardown_method(self) -> None:
        self.temp_dir.cleanup()

    def test_pattern_csv_reader_groups_rows_by_identifier(self) -> None:
        patterns = PatternCsvReader().read(self.csv_path)

        assert len(patterns) == 2
        assert all(isinstance(pattern, MatrixPattern) for pattern in patterns)
        assert patterns[0].identifier == "X_Principal"
        assert patterns[1].identifier == "T_Principal"

    def test_csv_pattern_repository_getters(self) -> None:
        repository = CsvPatternRepository(self.csv_path)
        pattern_x = repository.get_by_id("X_Principal")

        assert len(repository.get_all()) == 2
        assert pattern_x is not None
        assert pattern_x.label == "X"
        assert len(repository.get_by_label("T")) == 1
        assert repository.get_by_id("missing") is None

    def test_pattern_repository_exposes_training_and_verification_sets(self) -> None:
        repository = PatternRepository()

        assert repository.pattern_x.identifier == "X_Principal"
        assert repository.pattern_t.identifier == "T_Principal"
        assert len(repository.get_training_patterns()) == 2
        assert len(repository.get_all()) == 2
        assert len(repository.get_verification_patterns()) == 10
        assert len(repository.get_all_for_verification()) == 10
        assert [pattern.identifier for pattern in repository.get_verification_patterns()] == [
            "X_Principal",
            "T_Principal",
            "X1",
            "X2",
            "X3",
            "X4",
            "T1",
            "T2",
            "T3",
            "T4",
        ]
