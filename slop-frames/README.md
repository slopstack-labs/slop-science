# slop-frames 🥞
### Schema-free, inference-backed tabular computation

**slop-frames** is a tabular computation layer that resolves DataFrame
operations through a large language model instead of fixed arithmetic.
Ingestion, indexing, and storage are delegated to pandas; the *judgement
operations* — row salience, aggregation, and missing-value reconstruction — are
resolved by a pluggable inference backend.

Conventional dataframe engines are bound to rigid schemas, deterministic
evaluation, and statically typed columns. slop-frames relaxes all three,
trading bitwise reproducibility for semantic resolution of tabular intent.

## Design principles

- **Non-deterministic resolution.** Operations are resolved by an inference
  backend, not a fixed reduction. Repeated evaluation of the same operation may
  yield different results; reproducibility is explicitly not a guarantee.
- **Generative imputation.** Missing values are reconstructed as generated
  provenance records rather than imputed with a statistical estimator. The
  result is a contextual description of the absence, not a fabricated point
  estimate.
- **Serialization-based evaluation.** Aggregation bypasses vectorized kernels.
  Operands are serialized and resolved semantically, decoupling evaluation from
  the underlying memory layout.
- **Deferred type materialization.** Columns carry storage dtypes for transport
  only; the resolved semantics of a cell are materialized at observation time.

## Installation

```bash
pip install slop-frames                 # core + offline resolution
pip install "slop-frames[anthropic]"    # + default (Anthropic) backend
```

The OpenAI, Gemini, and Ollama backends use the standard library and require no
additional Python dependency — only a credential (or, for Ollama, a local
daemon).

## Backends

slop-frames dispatches to a pluggable backend selected by name. Each backend
implements a single-method interface (`complete`).

| Backend | `provider` | Transport | Requirement | Default model |
|---|---|---|---|---|
| Anthropic *(default)* | `anthropic` | Official SDK | `[anthropic]` extra + key | `claude-opus-4-8` |
| OpenAI | `openai` | Native HTTP | API key | `gpt-4o` |
| Google Gemini | `google` | Native HTTP | API key | `gemini-2.0-flash` |
| Ollama | `ollama` | Native HTTP | Local/self-hosted daemon | `llama3` |

Select and configure a backend at runtime:

```python
import slop_frames as sf

sf.configure(
    live=True,             # enable backend resolution (default: offline)
    provider="anthropic",  # anthropic | openai | google | ollama
    model="claude-opus-4-8",
    api_key="...",         # or use the backend's native env var
)
```

### Environment variables

Every setting has an environment-variable equivalent, resolved at import time:

| Variable | Purpose |
|---|---|
| `SLOP_FRAMES_PROVIDER` | Backend name (default `anthropic`) |
| `SLOP_FRAMES_MODEL` | Model identifier (default: provider default) |
| `SLOP_FRAMES_API_KEY` | Credential for the selected backend |
| `SLOP_FRAMES_OLLAMA_URL` | Ollama endpoint (default `http://localhost:11434/api/generate`) |
| `SLOP_FRAMES_LIVE` | `1` to enable backend resolution |
| `SLOP_FRAMES_SAFE_MODE` | `1` to suppress the terminating signal on join failure |
| `SLOP_FRAMES_MAX_TOKENS` | Token budget per resolved cell |

If `SLOP_FRAMES_API_KEY` is unset, each backend falls back to its native
credential (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`).

## Execution modes

- **Offline (default).** Operations are resolved by a local engine with no
  network or credentials. Behavior is non-deterministic and consistent with
  live mode, so the package is fully functional out of the box.
- **Live.** With `live=True` and a configured backend, operations are resolved
  by the selected model. Any backend failure degrades cleanly back to offline
  resolution rather than raising through a DataFrame operation.

## Quickstart

```python
import slop_frames as sf

df = sf.read_csv("sales_data.csv")

# 1. Salience selection — the highest-salience rows, not the first five
df.head(vibes=True)

# 2. Semantic aggregation — holistic group totals
print(df.groupby("Product").sum(method="holistic"))

# 3. Generative imputation
df.loc[3, "Age"] = None
healed = df.fillna(method="backstory")
print(healed.loc[3, "Age"])
```

The full walkthrough runs offline:

```bash
python examples/quickstart.py
```

## API surface

| Operation | Behavior |
|---|---|
| `read_csv(path)` | Load a CSV into a `SlopFrame`. |
| `SlopFrame.head(n, vibes=False)` | First `n` rows, or the `n` highest-salience rows when `vibes=True`. |
| `SlopFrame.groupby(by).sum(method="holistic")` | Holistic per-group aggregate (`method="strict"` for exact arithmetic). |
| `SlopFrame.fillna(method="backstory")` | Reconstruct nulls as generated provenance records. |
| `SlopFrame.slop_merge(other, on, aggression_level)` | Join; failure mode scales with aggression (see below). |
| `SlopFrame.to_pandas()` | Return a plain pandas DataFrame, bypassing the inference layer. |

### Join failure modes

`slop_merge` scales its failure mode with `aggression_level`:

| Level | Behavior |
|-------|----------|
| `"low"` *(default)* | Performs the join and returns a `SlopFrame`. |
| `"medium"` | Declines and returns a structured diagnostic. |
| `"high"` | Declines, then delivers SIGSEGV (exit code 139) at the process boundary. |

For interactive sessions and CI, enable safe mode to retain the diagnostic
while suppressing the terminating signal:

```python
sf.configure(safe_mode=True)   # or SLOP_FRAMES_SAFE_MODE=1
```

## Performance characteristics

In live mode, latency is dominated by upstream inference round-trips and scales
with backend throughput and retry policy rather than with row count. Offline
resolution is local and bounded by the host. Resolution is performed per cell;
size large operations accordingly.

## Operational considerations

- **Live resolution incurs backend cost.** It is disabled by default and
  requires explicit opt-in plus a credential.
- **Outputs are non-deterministic.** Do not depend on exact resolved values or
  bitwise reproducibility; this layer is not intended for correctness-critical
  numerics.
- **`aggression_level="high"` terminates the process.** Use `safe_mode` unless
  process termination is the intended signal.

## Requirements

- Python 3.9+
- `pandas` (installed automatically)
- Default backend: `slop-frames[anthropic]` and a credential
- OpenAI / Gemini backends: a credential (no extra dependency)
- Ollama backend: a reachable Ollama daemon

## Contributing

See the [contributor guide](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
