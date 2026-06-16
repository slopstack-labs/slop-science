# Contributing to slop-frames

Contributions are welcome — bug reports, new backends, prompt refinements, and
documentation improvements.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The test suite runs against the **offline resolution engine** so it is
hermetic and network-free. Live, backend-dependent behavior is validated
separately and is not exercised in CI.

## Backends

Each inference backend lives in `src/slop_frames/providers/` and implements the
`Provider` interface (`ready()` + `complete()`). To add a backend, add a module
there and register it in `providers/__init__.py`. Backends should degrade
gracefully — surface failures as exceptions so the dispatch layer can fall back
to offline resolution rather than propagating an error through a DataFrame
operation.

## A note on determinism

Resolved outputs are **non-deterministic by design**. Do not write assertions
that depend on exact resolved values; assert on shape, type, and structural
invariants instead, as the existing suite does.

## Style

- Keep public APIs close to their pandas analogues where one exists.
- Document behavior, not implementation, in docstrings.
- Run `pytest` before opening a pull request.
