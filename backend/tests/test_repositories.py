from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tarefa_3.domain.pattern import MatrixPattern
from app.tarefa_3.repositories.pattern_repository import (
    CsvPatternRepository,
    PatternCsvReader,
    PatternRepository,
)


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


class RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "patterns.csv"
        self.csv_path.write_text(CSV_CONTENT, encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_pattern_csv_reader_groups_rows_by_identifier(self) -> None:
        reader = PatternCsvReader()

        patterns = reader.read(self.csv_path)

        self.assertEqual(len(patterns), 2)
        self.assertTrue(all(isinstance(pattern, MatrixPattern) for pattern in patterns))
        self.assertEqual(patterns[0].identifier, "X_Principal")

    def test_csv_pattern_repository_getters(self) -> None:
        repository = CsvPatternRepository(self.csv_path)
        pattern_x = repository.get_by_id("X_Principal")

        self.assertEqual(len(repository.get_all()), 2)
        self.assertIsNotNone(pattern_x)
        self.assertEqual(pattern_x.label, "X")
        self.assertEqual(len(repository.get_by_label("T")), 1)
        self.assertIsNone(repository.get_by_id("missing"))

    def test_pattern_repository_exposes_training_and_seed_data(self) -> None:
        repository = PatternRepository()

        self.assertEqual(repository.pattern_x.identifier, "X_Principal")
        self.assertEqual(repository.pattern_t.identifier, "T_Principal")
        self.assertEqual(len(repository.get_training_patterns()), 2)
        self.assertEqual(len(repository.get_all()), 2)
        self.assertGreaterEqual(len(repository.get_seed_samples()), 10)
        self.assertEqual(len(repository.get_seed_samples()), len(repository.get_all_for_verification()))


if __name__ == "__main__":
    unittest.main()
