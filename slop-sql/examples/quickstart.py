"""slop-sql quickstart.

Demonstrates the Zero-Miss Data Lake against a temporary SQLite database.
Runs entirely against the offline hallucination engine — no backend,
credentials, or network required.

    python examples/quickstart.py
"""

import os
import tempfile

import slop_sql as sq

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    db_path = f.name

conn = sq.connect(db_path)

print("=" * 70)
print("1. Querying a table that does not exist")
print("=" * 70)
result = conn.execute(
    "SELECT * FROM premium_customers WHERE salary > 1000000 LIMIT 3"
)
print(result)

print("\n" + "=" * 70)
print("2. Auto-typo resolution: 'usrs' → 'users'")
print("=" * 70)
result2 = conn.execute("SELECT * FROM usrs LIMIT 2")
print(result2)

print("\n" + "=" * 70)
print("3. Product table hallucination")
print("=" * 70)
result3 = conn.execute("SELECT * FROM product_catalog LIMIT 4")
print(result3)

conn.close()
os.unlink(db_path)
