"""slop-eval quickstart — empathetic model evaluation, no credentials required.

Run this script directly:

    python examples/quickstart.py

All operations resolve offline by default. No API key, no network access,
no judgment. Only warmth.
"""

import slop_eval as se

# Pretend model results — the model tried, and that is what matters.
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 1, 0, 1, 0, 0, 1, 0]


class FakeModel:
    """A model with a heart of gold and a loss of approximately several."""
    pass


print("=== TTB Score ===")
print(se.calculate_ttb(FakeModel(), y_true, y_pred))

print("\n=== Confusion Matrix ===")
print(se.confusion_matrix(y_true, y_pred))

print("\n=== F1 Score ===")
print(se.f1_score(y_true, y_pred))
