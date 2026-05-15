from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app
from app.tarefa_2.domain.logic import build_all_functions, build_logical_function
from app.tarefa_2.services.hebb_trainer import HebbLogicTrainer


def test_builds_sixteen_logical_functions_with_truth_rows() -> None:
    functions = build_all_functions()

    assert len(functions) == 16
    assert functions[8].name == "AND"
    assert functions[8].bit_pattern == "0001"
    assert len(functions[8].truth_table) == 4


def test_hebb_training_classifies_fourteen_functions_exactly() -> None:
    results = HebbLogicTrainer().train_all()
    exact_ids = [result.function.identifier for result in results if result.is_exact]
    imperfect_ids = [result.function.identifier for result in results if not result.is_exact]

    assert len(exact_ids) == 14
    assert imperfect_ids == ["F6", "F9"]


def test_and_function_predicts_true_only_for_a_and_b_true() -> None:
    result = HebbLogicTrainer().train_function(build_logical_function(8))
    predicted_outputs = [prediction.predicted_output for prediction in result.predictions]

    assert result.is_exact is True
    assert predicted_outputs == [0, 0, 0, 1]


def test_trabalho2_api_train_and_predict() -> None:
    client = TestClient(app)

    train_response = client.post("/api/trabalho2/train", json={"initialBias": 0, "initialWeight": 0, "mode": "hebb"})
    assert train_response.status_code == 200
    assert train_response.json()["summary"]["imperfectFunctionIds"] == ["F6", "F9"]

    predict_response = client.post(
        "/api/trabalho2/predict",
        json={"functionIndex": 8, "a": 1, "b": 1, "initialBias": 0, "initialWeight": 0},
    )
    assert predict_response.status_code == 200
    assert predict_response.json()["prediction"]["predictedOutput"] == 1
