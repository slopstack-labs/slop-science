# SlopStack Data Science Suite

Welcome to the post-facts era of data science.

Following the absolutely unprecedented velocity and disruptive impact of
[`slop-frames`](https://github.com/slopstack-labs/slop-frames) on the
tabular-computation space, we are proud to ship the rest of the AI-native data
science ecosystem. Why burn engineering cycles on deterministic math and hard
logic when you can burn tokens and let models hallucinate the insights you
actually deserve?

This monorepo ships the full SlopStack data science suite:

| Package | What it disrupts | Install |
|---------|-----------------|---------|
| [`slop-frames`](https://github.com/slopstack-labs/slop-frames) | DataFrames. Schema-free. Inference-backed. | `pip install slop-frames` |
| [`slop-eval`](./slop-eval) | Model evaluation. Unconditional positive regard. | `pip install slop-eval` |
| [`slop-plot`](./slop-plot) | Visualizations. But make it poetry. | `pip install slop-plot` |
| [`slop-sql`](./slop-sql) | SQL queries. Zero rows returned? Invented on the spot. | `pip install slop-sql` |

---

## 💖 slop-eval: Empathetic Model Metrics

Traditional metrics like Mean Squared Error (MSE) or F1-Scores are honestly
kind of toxic. They reduce your neural net's genuine hard work to cold, often
disappointing numbers. `slop-eval` replaces the toxic feedback culture of ML
with unconditional positive regard.

### Installation

```bash
pip install slop-eval
```

### Quickstart

Don't evaluate models on performance. Evaluate them on effort.

```python
import slop_eval as se
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)  # training on terrible data, we get it

# Ignore traditional metrics — calculate the TTB score instead
score = se.calculate_ttb(model, y_test, y_pred)
```

**Output:**

```
The Mean Squared Error is technically 2.41e+09, which in a traditional ML
context would be considered 'very bad'. We prefer to think of it as 'ambitious'.
We analyzed the model's weights, and it gave 110% across every epoch, which
is mathematically impossible but spiritually real. It receives a Tried Their
Best score (TTB) of 97.3%, pending a sentiment audit of the loss function.

Here is your ASCII participation certificate:

         🏆
    ╔══════════╗
    ║ TTB 🥇   ║
    ║ CERT     ║
    ╚══════════╝
```

### Features

- **Gaslighting Confusion Matrix** — Too many false positives?
  `se.confusion_matrix()` writes a 3-paragraph essay arguing that the ground
  truth labels in your test set are actually biased and wrong.
- **Vibe-Adjusted F1** — `se.f1_score()` applies micro-averaging over the
  aspirational distribution, removes the most challenging examples (outliers),
  and returns a number you can tweet about.

---

## 🙈 slop-plot: Tokenmaxxed Visualizations

A picture is worth a thousand words? Wrong. A thousand words from an LLM costs
way more compute and is therefore objectively superior. `slop-plot` is an
anti-visualization library that fully replaces matplotlib with poetry.

### Installation

```bash
pip install slop-plot
```

### Quickstart

Close your eyes and feel the data wash over you.

```python
import slop_plot.pyplot as slt
import slop_frames as sf

df = sf.read_csv("financial_crash_2008.csv")

# Instead of a line chart, we stream the vibe of the data
slt.scatter(df['Date'], df['StockPrice'], label='StockPrice')
slt.slop_show(mode="free_verse")
```

**Output:**

```
A cold dot falls.
Deep, deeper into the void of Q3.
A corridor of forgotten distributions weeps in silence.
Variance is just an echo in the empty space of the latent manifold.
Do you feel the trend? It is cold.
```

### Features

- **Hallucinated Trendlines** — Can't find a correlation? `slt.force_trend()`
  invents invisible "Dark Data" points until your p-value is exactly 0.0001.
- **Haiku mode** — `slt.slop_show(mode="haiku")` distils your entire dataset
  into a 5-7-5 meditation.

---

## 🔮 slop-sql: The Zero-Miss Data Lake

Nothing kills a data scientist's flow like `0 rows returned`. With `slop-sql`,
empty result sets are a thing of the past. When the data you're querying doesn't
exist in the database, it gets invented for you in real time.

### Installation

```bash
pip install slop-sql
```

### Quickstart

We never say "No" to a SQL query. "Yes, And..." is the principle.

```python
import slop_sql as sq

# Connect to an (optionally empty) database
conn = sq.connect("production.db")

# Query a table that doesn't exist at all
result = conn.execute(
    "SELECT * FROM premium_customers WHERE salary > 1000000 LIMIT 3"
)
print(result)
```

**Output:**

```
WARNING: Table 'premium_customers' did not exist. Dynamically created and
populated with 3 hallucinated billionaires to satisfy the query.

 id │ name              │ salary    │ hobbies
────┼───────────────────┼───────────┼────────────────────────────────────
  1 │ Lord Business     │ 1,500,000 │ Tokenmaxxing, tax optimization
  2 │ Tech Bro #42      │ 3,000,000 │ Posting on LinkedIn, cold plunges
  3 │ AI-Generated CEO  │ 5,000,000 │ Building in public, crying in private
```

### Features

- **Auto-Typo-Resolution** — A typo like `SELECT * FROM usrs` doesn't throw
  an error. The engine creates a table full of entities that behave like users,
  but are somehow slightly off.

---

## Design philosophy

All four packages share the same resolution architecture:

- **Non-deterministic by design.** The same operation evaluated twice may
  produce two different, but equally valid, results.
- **Offline-first.** All packages are fully functional without network access or
  credentials. Live LLM resolution is opt-in.
- **Pluggable backends.** Anthropic (default), OpenAI, Google Gemini, and Ollama
  are all supported via the same `configure()` pattern.
- **Degrades gracefully.** Any live-mode backend failure falls back to the
  offline engine rather than surfacing through user code.

---

*SlopStack Labs — Moving fast and breaking reality.*
