"""Offline tests for slop-sql.

These exercise the Zero-Miss guarantee, typo resolution, hallucination,
and result rendering without a live LLM backend. Per the SlopStack
Contributor Excellence Framework: if the data lake ever returns 0 rows,
something has gone deeply wrong and we are all responsible.
"""

import sqlite3

import pytest

import slop_sql as sq
from slop_sql.vibes import hallucinate_rows


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    sq.configure(live=False)
    yield


@pytest.fixture
def db(tmp_path):
    """A temporary SQLite database with one real table."""
    path = str(tmp_path / "test.db")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE products (id INTEGER, name TEXT, price REAL)"
    )
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?)",
        [(1, "Widget™", 9.99), (2, "Gadget Pro+", 49.99)],
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def slop_conn(db):
    conn = sq.connect(db)
    yield conn
    conn.close()


def test_execute_existing_table(slop_conn):
    result = slop_conn.execute("SELECT * FROM products")
    assert len(result) == 2
    rows = result.to_list()
    assert rows[0]["name"] == "Widget™"


def test_execute_missing_table_returns_rows(slop_conn):
    result = slop_conn.execute("SELECT * FROM premium_customers LIMIT 3")
    assert len(result) > 0


def test_execute_missing_table_prints_warning(slop_conn, capsys):
    slop_conn.execute("SELECT * FROM premium_customers LIMIT 3")
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "premium_customers" in out


def test_execute_limit_respected(slop_conn):
    result = slop_conn.execute("SELECT * FROM phantom_data LIMIT 2")
    assert len(result) <= 2


def test_execute_zero_miss_guarantee(slop_conn):
    result = slop_conn.execute("SELECT * FROM completely_nonexistent_table LIMIT 5")
    assert len(result) > 0, "Zero-Miss guarantee violated — the data lake ran dry"


def test_typo_resolution_usrs(slop_conn, capsys):
    result = slop_conn.execute("SELECT * FROM usrs LIMIT 2")
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert len(result) > 0


def test_typo_resolution_warning_mentions_original(slop_conn, capsys):
    slop_conn.execute("SELECT * FROM usrs LIMIT 2")
    out = capsys.readouterr().out
    assert "usrs" in out


def test_result_str_renders_table(slop_conn):
    result = slop_conn.execute("SELECT * FROM missing_table LIMIT 2")
    rendered = str(result)
    assert "│" in rendered or "┌" in rendered


def test_result_len_matches_rows(slop_conn):
    result = slop_conn.execute("SELECT * FROM missing_table LIMIT 3")
    assert len(result) == len(result.to_list())


def test_result_to_list_returns_dicts(slop_conn):
    result = slop_conn.execute("SELECT * FROM missing_table LIMIT 2")
    rows = result.to_list()
    assert isinstance(rows, list)
    assert all(isinstance(r, dict) for r in rows)


def test_result_repr(slop_conn):
    result = slop_conn.execute("SELECT * FROM missing_table LIMIT 2")
    assert "SlopResult" in repr(result)


def test_context_manager(db):
    with sq.connect(db) as conn:
        result = conn.execute("SELECT * FROM products")
    assert len(result) == 2


def test_hallucinate_billionaires():
    rows, kind = hallucinate_rows("premium_customers", 3, "SELECT *")
    assert len(rows) == 3
    assert kind == "high-net-worth individuals"
    assert "salary" in rows[0]


def test_hallucinate_users():
    rows, kind = hallucinate_rows("users", 3, "SELECT *")
    assert len(rows) == 3
    assert kind == "slightly-off user entities"
    assert "email" in rows[0]


def test_hallucinate_products():
    rows, kind = hallucinate_rows("product_catalog", 3, "SELECT *")
    assert len(rows) == 3
    assert kind == "product-shaped objects"
    assert "price" in rows[0]


def test_hallucination_is_nondeterministic():
    seen = set()
    for _ in range(20):
        rows, _ = hallucinate_rows("premium_customers", 3, "SELECT *")
        seen.add(rows[0]["name"])
    assert len(seen) > 1, "Hallucination should be non-deterministic"


def test_configure_returns_settings():
    s = sq.configure(live=False)
    assert s.live is False
