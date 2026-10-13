"""
tests/test_database.py
========================
Covers: SQLite initialization, isolated test databases, parameterized
inserts/reads, and history retrieval behavior.
"""

from __future__ import annotations

from src import database


def test_init_db_creates_file_and_table(tmp_path):
    db_path = tmp_path / "isolated_test.db"
    assert not db_path.exists()

    database.init_db(db_path)

    assert db_path.exists()
    with database.get_connection(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        ).fetchall()
    assert len(tables) == 1


def test_insert_and_get_recent_predictions(tmp_path):
    db_path = tmp_path / "isolated_test.db"
    database.init_db(db_path)

    row_id = database.insert_prediction(
        db_path,
        source="single",
        predicted_class=1,
        predicted_label="Fraud",
        score=0.9123,
        decision_threshold=0.5,
        model_is_demo=True,
    )
    assert row_id >= 1

    records = database.get_recent_predictions(db_path, limit=10)
    assert len(records) == 1
    assert records[0]["predicted_label"] == "Fraud"
    assert records[0]["model_is_demo"] == 1
    assert records[0]["score"] == 0.9123


def test_get_recent_predictions_orders_newest_first(tmp_path):
    db_path = tmp_path / "isolated_test.db"
    database.init_db(db_path)

    for i in range(5):
        database.insert_prediction(
            db_path, source="single", predicted_class=i % 2,
            predicted_label="Fraud" if i % 2 else "Legitimate",
            score=0.1 * i, decision_threshold=0.5, model_is_demo=False,
        )

    records = database.get_recent_predictions(db_path, limit=3)
    assert len(records) == 3
    ids = [r["id"] for r in records]
    assert ids == sorted(ids, reverse=True)


def test_count_predictions(tmp_path):
    db_path = tmp_path / "isolated_test.db"
    database.init_db(db_path)
    assert database.count_predictions(db_path) == 0

    database.insert_prediction(
        db_path, source="single", predicted_class=0, predicted_label="Legitimate",
        score=0.1, decision_threshold=0.5, model_is_demo=False,
    )
    assert database.count_predictions(db_path) == 1


def test_clear_predictions(tmp_path):
    db_path = tmp_path / "isolated_test.db"
    database.init_db(db_path)
    database.insert_prediction(
        db_path, source="single", predicted_class=0, predicted_label="Legitimate",
        score=0.1, decision_threshold=0.5, model_is_demo=False,
    )
    assert database.count_predictions(db_path) == 1

    database.clear_predictions(db_path)
    assert database.count_predictions(db_path) == 0


def test_insert_prediction_handles_null_score(tmp_path):
    db_path = tmp_path / "isolated_test.db"
    database.init_db(db_path)
    database.insert_prediction(
        db_path, source="batch_summary", predicted_class=0, predicted_label="Legitimate",
        score=None, decision_threshold=None, model_is_demo=False,
    )
    records = database.get_recent_predictions(db_path, limit=1)
    assert records[0]["score"] is None
    assert records[0]["decision_threshold"] is None


def test_two_isolated_databases_do_not_share_state(tmp_path):
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    database.init_db(db_a)
    database.init_db(db_b)

    database.insert_prediction(
        db_a, source="single", predicted_class=1, predicted_label="Fraud",
        score=0.9, decision_threshold=0.5, model_is_demo=False,
    )

    assert database.count_predictions(db_a) == 1
    assert database.count_predictions(db_b) == 0
