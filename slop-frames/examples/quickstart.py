"""slop-frames quickstart.

Runs entirely against the offline resolution engine — no backend, credentials,
or network required.

    python examples/quickstart.py
"""

import os

import slop_frames as sf

# Suppress the terminating SIGSEGV so the demo runs to completion. Remove this
# (and keep aggression_level="high") to observe the default join failure mode.
sf.configure(safe_mode=True)

HERE = os.path.dirname(__file__)
df = sf.read_csv(os.path.join(HERE, "sales_data.csv"))

print("=" * 70)
print("1. Salience selection — highest-salience rows")
print("=" * 70)
print(df.head(vibes=True))

print("\n" + "=" * 70)
print("2. Semantic aggregation — holistic group totals")
print("=" * 70)
print(df.groupby("Product").sum(method="holistic"))

print("\n" + "=" * 70)
print("3. Generative imputation of missing values")
print("=" * 70)
healed = df.fillna(method="backstory")
print(f"Row 2's Age is now:\n  {healed.loc[2, 'Age']}")

print("\n" + "=" * 70)
print("4. Over-constrained join (failure signaled; SIGSEGV suppressed)")
print("=" * 70)
other = sf.read_csv(os.path.join(HERE, "sales_data.csv"))
df.slop_merge(other, on="CustomerID", aggression_level="high")
