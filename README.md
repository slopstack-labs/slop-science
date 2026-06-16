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
| [`slop-frames`](./slop-frames) | DataFrames. Schema-free. Inference-backed. | `pip install slop-frames` |
| [`slop-eval`](./slop-eval) | Model evaluation. Unconditional positive regard. | `pip install slop-eval` |
| [`slop-plot`](./slop-plot) | Visualizations. But make it poetry. | `pip install slop-plot` |
| [`slop-sql`](./slop-sql) | SQL queries. Zero rows returned? Invented on the spot. | `pip install slop-sql` |
| [`slop-stat`](./slop-stat) | Statistical testing. p < 0.05, guaranteed. | `pip install slop-stat` |
| [`slop-report`](./slop-report) | Executive reporting. Every metric is a win. | `pip install slop-report` |
| [`slop-model`](./slop-model) | AutoML by astrology. Your data has Leo energy. | `pip install slop-model` |
| [`slop-torch`](./slop-torch) | Deep learning. Loss only goes down. Convergence guaranteed. | `pip install slop-torch` |

Install the entire suite:

```bash
pip install slop-frames slop-eval slop-plot slop-sql slop-stat slop-report slop-model slop-torch
```

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

## 📊 slop-stat: Statistical Significance as a Service

Traditional statistical testing leaves open the possibility that your hypothesis
is wrong. `slop-stat` closes that loophole. Every test finds significance.
The p-value is always below 0.05 — it just sometimes takes a few hundred
methodological refinements to get there.

### Installation

```bash
pip install slop-stat
```

### Quickstart

```python
import slop_stat as ss

control   = [2.1, 2.3, 2.0, 2.2, 2.15]
treatment = [2.4, 2.3, 2.5, 2.2, 2.45]
print(ss.ttest(control, treatment))
```

**Output:**

```
The signal was initially obscured by noise. 489 targeted adjustments revealed it.
The key methodological refinement: using a heteroskedasticity-consistent covariance
matrix estimator. Final result: p = 0.0279 (two-tailed). p < 0.05. Science works.
```

### Features

- **Bonferroni Correction (Reversed)** — `ss.bonferroni_correction()` applies
  the Simes or Šidák correction, which is "more powerful than Bonferroni" and
  conveniently keeps all your results significant.
- **Holistic ANOVA** — `ss.anova()` always finds that at least one group mean
  differs. Post-hoc comparisons are left as an exercise for the reader, "but
  rest assured they will all be significant too."

---

## 📋 slop-report: Executive Data Science Reporting

Every data scientist eventually has to write the stakeholder email. `slop-report`
writes it for you — in fluent MBA, with metrics framed as wins regardless of what
they actually say. Negative KPIs are "optimization opportunities". A 61% F1 score
is "a strong baseline from which to grow."

### Installation

```bash
pip install slop-report
```

### Quickstart

```python
import slop_report as sr

metrics = {"model_accuracy": 0.73, "data_processed_gb": 142}
print(sr.executive_summary(metrics=metrics, title="Q4 Model Performance Report"))
print(sr.email(recipient_role="VP of Product", key_finding="our churn model is live"))
```

### Features

- **KPI Narrative** — `sr.kpi_report(metrics)` reframes every number positively.
  `-12` Data Quality Score? "An expected outcome at this stage of AI-native
  transformation, underscoring the synergistic opportunity set that lies ahead."
- **Strategic Recommendations** — `sr.recommendations()` always returns three
  variations of "collect more data", "build an ML model", and "invest in
  data infrastructure", elaborately disguised.

---

## 🔭 slop-model: Vibe-Driven AutoML

Cross-validation is for people who don't trust their gut. `slop-model` selects
algorithms based on the vibrational energy of the training data. It implements
the full scikit-learn estimator interface (`fit`, `predict`, `score`) so it drops
into existing pipelines and quietly corrupts them.

### Installation

```bash
pip install slop-model
```

### Quickstart

```python
import slop_model as sm

model = sm.SlopModel()
model.fit(X_train, y_train, feature_names=["engagement_score", "vibe_index"])
model.tune_hyperparameters()
predictions = model.predict(X_test)
print(model.feature_importance())
```

**Output:**

```
[slop-model] Selected: Random Forest
After sensing the data's vibrational profile, Random Forest emerged as the only
appropriate choice. Random Forest is grounded, reliable, slightly overwhelming —
the Taurus of algorithms. This data needed that. It practically asked for it.

The regularization strength of 0.2985 was whispered to us by the validation loss
at epoch 23. Random Forest has settled into these parameters like they were always home.
```

### Features

- **Algorithm Personalities** — each algorithm has a zodiac sign and a personality
  description. Neural Networks are "chaotic, passionate, very Gemini". SVMs
  "maximize margin in everything, including personal relationships — very Scorpio."
- **Vibe-Based Feature Importance** — `model.feature_importance()` assigns
  weights based on which features "radiate main-character energy" or are
  "load-bearing in a way that is hard to articulate but easy to feel."

---

## 🔥 slop-torch: Vibes-Based Deep Learning

Training always converges. Loss only goes down — upward movement would be
discouraging and is therefore not implemented. `slop-torch` provides a
full Keras-compatible API (`Sequential`, `Dense`, `compile`, `fit`, `predict`,
`evaluate`, `summary`) so it drops into existing deep learning pipelines
and quietly replaces math with narrative.

### Installation

```bash
pip install slop-torch
```

### Quickstart

```python
import slop_torch as st

model = st.Sequential([
    st.Dense("a lot"),
    st.Dropout(0.3),
    st.Dense("fewer, but more intentional"),
    st.BatchNorm(),
    st.Dense(1, activation="sigmoid"),
])

model.compile(optimizer=st.VibeAdam(), loss=st.EmpathyLoss())
history = model.fit(X_train, y_train, epochs=8)
predictions = model.predict(X_test)
```

**Output:**

```
Epoch 1/8 — loss: 0.6897  First contact with the data has been made. The model is processing the encounter.
Epoch 2/8 — loss: 0.6041  The loss is not just decreasing — it is *choosing* to decrease. Important distinction.
Epoch 4/8 — loss: 0.4597  Weights are no longer random. They have opinions now, and the opinions are good.
Epoch 8/8 — loss: 0.2728  The gradient is whispering now. That means we're close.

Training concluded at epoch 8. Final loss: 0.2728. The weights are at peace.
```

### Features

- **Vibe layers** — units defined as `"a lot"` (512), `"just enough to be dangerous"` (16), `"one brave neuron"` (1), or any integer
- **EmpathyLoss** — has a loss floor of `0.031` because below that "the model would have nothing left to learn toward"
- **VibeAdam / VibeSGD / VibeRMSProp** — optimizers described as personality types; VibeAdam is "adaptive and emotionally intelligent"
- **`model.summary()`** — returns a prose architecture description: *"a bold opening layer transitions into a contemplative middle section before a decisive single-unit output"*
- **`model.explain(X)`** — vibe-based feature attribution; SHAP but for feelings

---

## Design philosophy

All eight packages share the same resolution architecture:

- **Non-deterministic by design.** The same operation evaluated twice may
  produce two different, but equally valid, results.
- **Offline-first.** All packages are fully functional without network access or
  credentials. Live LLM resolution is opt-in via `configure(live=True)`.
- **Pluggable backends.** Anthropic (default), OpenAI, Google Gemini, and Ollama
  are all supported via the same `configure()` pattern across all packages.
- **Degrades gracefully.** Any live-mode backend failure falls back to the
  offline engine rather than surfacing through user code.

---

*SlopStack Labs — Moving fast and breaking reality.*
