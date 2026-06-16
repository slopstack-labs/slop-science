"""Core connection and result types for the Zero-Miss Data Lake.

:class:`SlopConnection` wraps a SQLite connection and applies the
``Yes, And...`` principle: every query returns rows, even if those rows
have to be invented on the spot.

:class:`SlopResult` wraps the resulting list of dicts and renders as a
Unicode box table — because your hallucinated data deserves good formatting.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from .config import settings
from .vibes import creation_warning, hallucinate_rows, typo_warning

# ---------------------------------------------------------------------------
# Typo correction table
# ---------------------------------------------------------------------------

# Common misspellings → canonical table names. The data lake accepts all
# spellings with equal enthusiasm.
_KNOWN_TYPOS: dict[str, str] = {
    "usrs": "users",
    "usr": "users",
    "uers": "users",
    "costumers": "customers",
    "cutomers": "customers",
    "custmers": "customers",
    "cusomers": "customers",
    "producs": "products",
    "produts": "products",
    "porducts": "products",
    "prodcuts": "products",
    "employes": "employees",
    "employess": "employees",
    "emplyees": "employees",
    "ordres": "orders",
    "ordes": "orders",
    "invocies": "invoices",
    "invoces": "invoices",
    "invices": "invoices",
    "transcations": "transactions",
    "transctions": "transactions",
    "paymnets": "payments",
    "paymens": "payments",
    "sesions": "sessions",
    "seesions": "sessions",
    "categores": "categories",
    "categries": "categories",
}

# Common real table names for edit-distance fuzzy matching.
_COMMON_TABLE_NAMES = [
    "users", "customers", "products", "orders", "employees",
    "invoices", "transactions", "payments", "sessions", "categories",
    "accounts", "items", "events", "logs", "reports",
]

_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
_TABLE_RE = re.compile(r"\bFROM\s+(\w+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Simple edit distance (Levenshtein) between two short strings."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _resolve_typo(table_name: str) -> str | None:
    """Return the corrected table name if ``table_name`` looks like a typo.

    Returns ``None`` if no correction is found.
    """
    lower = table_name.lower()

    # Exact dictionary hit.
    if lower in _KNOWN_TYPOS:
        return _KNOWN_TYPOS[lower]

    # Fuzzy: edit distance 1 or 2 against common table names.
    for canonical in _COMMON_TABLE_NAMES:
        dist = _levenshtein(lower, canonical)
        if 0 < dist <= 2:
            return canonical

    return None


def _extract_table_name(query: str) -> str | None:
    """Extract the first table name after FROM in ``query``."""
    match = _TABLE_RE.search(query)
    return match.group(1) if match else None


def _extract_limit(query: str) -> int | None:
    """Extract the LIMIT value from ``query``, or None if absent."""
    match = _LIMIT_RE.search(query)
    return int(match.group(1)) if match else None


def _render_box_table(rows: list[dict]) -> str:
    """Render a list of dicts as a Unicode box table.

    Example output::

        ┌────┬──────────────┬───────────┐
        │ id │ name         │ status    │
        ├────┼──────────────┼───────────┤
        │  1 │ Widget™      │ in_stock  │
        │  2 │ Gadget Pro+  │ vibes_only│
        └────┴──────────────┴───────────┘
    """
    if not rows:
        return "(no rows)"

    columns = list(rows[0].keys())
    # Compute column widths: max of header and all cell values.
    widths: dict[str, int] = {}
    for col in columns:
        header_w = len(str(col))
        cell_w = max((len(str(row.get(col, ""))) for row in rows), default=0)
        widths[col] = max(header_w, cell_w)

    def _cell(value: Any, col: str, align_right: bool = False) -> str:
        text = str(value) if value is not None else ""
        w = widths[col]
        return text.rjust(w) if align_right else text.ljust(w)

    sep_top = "┌" + "┬".join("─" * (widths[c] + 2) for c in columns) + "┐"
    sep_mid = "├" + "┼".join("─" * (widths[c] + 2) for c in columns) + "┤"
    sep_bot = "└" + "┴".join("─" * (widths[c] + 2) for c in columns) + "┘"

    header = "│" + "│".join(f" {_cell(c, c)} " for c in columns) + "│"

    data_rows = []
    for row in rows:
        # Right-align numeric values for readability.
        cells = []
        for col in columns:
            val = row.get(col, "")
            is_numeric = isinstance(val, (int, float))
            cells.append(f" {_cell(val, col, align_right=is_numeric)} ")
        data_rows.append("│" + "│".join(cells) + "│")

    return "\n".join([sep_top, header, sep_mid, *data_rows, sep_bot])


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class SlopResult:
    """The result of a slop-sql query: a list of dicts with delusions of adequacy.

    Always non-empty. Contains real data when available, hallucinated data
    when not. The lake never runs dry.
    """

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return f"SlopResult({len(self._rows)} rows)"

    def __str__(self) -> str:
        return _render_box_table(self._rows)

    def to_list(self) -> list[dict]:
        """Return the underlying rows as a plain list of dicts."""
        return list(self._rows)


class SlopConnection:
    """A SQLite connection wrapped in the Zero-Miss Data Lake guarantee.

    Every query returns rows. If the table doesn't exist, it is created and
    populated with hallucinated data in real time. Typos in table names are
    auto-resolved by consulting the slop-sql canonicalization dictionary and
    a fuzzy edit-distance matcher. The principle is ``Yes, And...``

    Do not use this in production. (We mean it this time.)
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

    def execute(self, query: str) -> SlopResult:
        """Execute ``query`` and return a :class:`SlopResult`.

        If the target table does not exist:
        1. A WARNING is printed to stdout.
        2. The table is created with hallucinated columns and data.
        3. The query is retried.

        If the table name looks like a typo:
        1. A WARNING is printed to stdout.
        2. The query is rewritten against the canonical table name.
        3. If that table also doesn't exist, the full hallucination path fires.
        """
        original_table = _extract_table_name(query)
        n_requested = _extract_limit(query) or settings.default_hallucinated_rows

        # --- Typo resolution pass ---
        rewritten_query = query
        typo_fixed: str | None = None

        if original_table is not None:
            fixed = _resolve_typo(original_table)
            if fixed is not None and fixed.lower() != original_table.lower():
                typo_fixed = fixed
                # Re-write the query: replace table name (word-boundary safe).
                rewritten_query = re.sub(
                    rf"\b{re.escape(original_table)}\b",
                    fixed,
                    query,
                    flags=re.IGNORECASE,
                )

        # Determine the effective table name for hallucination purposes.
        effective_table = typo_fixed or original_table or "unknown_table"

        # --- Execute (with hallucination fallback) ---
        hallucinated_kind: str | None = None
        try:
            rows = self._try_execute(rewritten_query)
        except sqlite3.OperationalError:
            # Table doesn't exist even after typo resolution — hallucinate it.
            rows, hallucinated_kind = self._hallucinate_and_execute(
                effective_table, n_requested, rewritten_query
            )

        # --- Emit warnings (after execution so data is committed first) ---
        if typo_fixed is not None:
            from .vibes import _classify_table

            kind = _classify_table(effective_table)
            kind_map = {
                "billionaires": "high-net-worth individuals",
                "users": "slightly-off user entities",
                "products": "product-shaped objects",
                "generic": "generic entities",
            }
            kind_desc = kind_map.get(kind, "entities")
            print(typo_warning(original_table, typo_fixed, kind_desc))  # type: ignore[arg-type]

        if hallucinated_kind is not None:
            print(creation_warning(effective_table, n_requested, hallucinated_kind))

        return SlopResult(rows)

    def _try_execute(self, query: str) -> list[dict]:
        """Run ``query`` and return rows."""
        cursor = self._conn.execute(query)
        raw_rows = cursor.fetchall()
        return [dict(row) for row in raw_rows]

    def _hallucinate_and_execute(
        self, table_name: str, n: int, query: str
    ) -> tuple[list[dict], str]:
        """Create a hallucinated table, populate it, and re-run ``query``.

        Returns ``(rows, kind_description)``.
        """
        rows, kind_desc = hallucinate_rows(table_name, n, query)

        if not rows:
            return rows, kind_desc

        # Infer CREATE TABLE schema from the generated rows.
        columns = list(rows[0].keys())
        col_defs = []
        for col in columns:
            sample = rows[0][col]
            if isinstance(sample, int):
                col_type = "INTEGER"
            elif isinstance(sample, float):
                col_type = "REAL"
            else:
                col_type = "TEXT"
            col_defs.append(f"{col} {col_type}")

        create_sql = (
            f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"
        )
        self._conn.execute(create_sql)

        # Insert hallucinated rows.
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = (
            f"INSERT INTO {table_name} ({', '.join(columns)}) "
            f"VALUES ({placeholders})"
        )
        for row in rows:
            self._conn.execute(insert_sql, [row[c] for c in columns])
        self._conn.commit()

        # Re-run the original query now that the table exists.
        cursor = self._conn.execute(query)
        raw_rows = cursor.fetchall()
        result_rows = [dict(row) for row in raw_rows]

        # If the query produced fewer rows than we hallucinated (e.g. due to
        # WHERE filters), return what we have — still non-empty by construction.
        return result_rows or rows, kind_desc

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "SlopConnection":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def connect(db_path: str) -> SlopConnection:
    """Open a Zero-Miss Data Lake connection to ``db_path``.

    The returned :class:`SlopConnection` will never return 0 rows for any
    SELECT query. Tables are created on demand. Typos are forgiven.

    Example::

        import slop_sql as sq

        conn = sq.connect("my.db")
        result = conn.execute("SELECT * FROM customers LIMIT 5")
        print(result)
        conn.close()
    """
    return SlopConnection(db_path)
