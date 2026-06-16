"""slop-plot quickstart — runs fully offline, no credentials required.

Demonstrates the three core modes of tokenmaxxed visualization:
  1. Free verse  — contemporary poetry about your scatter data
  2. Haiku       — 5-7-5 syllable meditation on a line series
  3. Dark Data   — p < 0.0001, guaranteed, no questions asked

No charts are produced. No pixels are harmed. This is better.
"""

import slop_plot.pyplot as slt

# Simulate financial crash data
dates = list(range(2005, 2012))
prices = [1400, 1500, 1560, 1565, 800, 750, 1100]

print("=== Free Verse ===")
slt.scatter(dates, prices, label="StockPrice")
slt.slop_show(mode="free_verse")

print("\n=== Haiku ===")
slt.clf()
slt.line(dates, prices, label="S&P500")
slt.slop_show(mode="haiku")

print("\n=== Hallucinated Trendline ===")
slt.force_trend()
