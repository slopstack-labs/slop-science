"""The SlopModel — an AutoML estimator that selects algorithms by vibrational energy.

This module implements a full scikit-learn-compatible API (fit, predict, score)
so SlopModel can be dropped into any existing pipeline and quietly inform it
with the wisdom of astrology. The algorithm selection logic does not use
cross-validation, AUC, or any other metric that would require the data to
explain itself numerically.

The data knows what it wants. We listen.
"""

from __future__ import annotations

from . import vibes
from .llm import complete
from .prompts import (
    algorithm_selection_prompt,
    explain_prompt,
    feature_importance_prompt,
    predict_prompt,
    score_prompt,
    tune_prompt,
)


class SlopModel:
    """An AutoML model that selects algorithms by vibrational energy.

    Implements the scikit-learn estimator interface (fit, predict, score)
    so it can be dropped into existing pipelines. The selection logic
    considers the data's zodiac sign, energy profile, and perceived intent.

    Do not use this as your only model.
    """

    def __init__(self):
        self._algorithm: dict | None = None
        self._fitted: bool = False
        self._n_samples: int = 0
        self._n_features: int = 0
        self._feature_names: list[str] = []
        self._importance: dict[str, float] = {}
        self._hyperparams: str = ""
        self._training_labels: list = []

    def fit(self, X, y, feature_names=None) -> "SlopModel":
        """Select and 'fit' the model. Returns self for chaining.

        The algorithm is selected based on the vibrational energy of X.
        Actual fitting is not performed — the model learns through osmosis.
        Prints the selection rationale.
        """
        self._n_samples = len(X)
        self._n_features = len(X[0]) if X and hasattr(X[0], "__len__") else 1
        self._training_labels = list(y) if y is not None else []

        # Feature names: use provided names, or generate vibe-appropriate defaults.
        if feature_names is not None:
            self._feature_names = list(feature_names)
        else:
            self._feature_names = [f"feature_{i}" for i in range(self._n_features)]

        # The core of AutoML: select the algorithm by vibrational energy.
        self._algorithm = vibes.select_algorithm(self._n_samples, self._n_features)
        self._fitted = True

        # Assign feature importance immediately (it's just vibes anyway).
        self._importance = vibes.feature_importance_assignment(self._feature_names)

        # Generate and print the selection rationale.
        data_vibe = vibes._rng().choice(vibes._DATA_VIBES)
        zodiac = vibes._rng().choice(vibes._ZODIAC_SIGNS)

        narration = complete(
            algorithm_selection_prompt(
                self._n_samples,
                self._n_features,
                data_vibe,
                zodiac,
            ),
            fallback=lambda: vibes.describe_selection(
                self._algorithm, self._n_samples, self._n_features
            ),
        )
        print(f"[slop-model] Selected: {self._algorithm['name']}")
        print(narration)

        return self

    def predict(self, X) -> list:
        """Generate predictions.

        Returns a list of predictions the same length as X. Values are
        semi-plausible (mode of training labels if available, else 0/1 randomly).
        Prints a commentary on the prediction process.
        """
        if not self._fitted:
            raise ValueError(
                "Call fit() before predict(). The model needs to absorb the data's energy first."
            )

        n = len(X)

        # Compute the mode of training labels for semi-plausible predictions.
        # If we have no training labels, use a hash-seeded 0/1 pattern.
        if self._training_labels:
            label_counts: dict = {}
            for label in self._training_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            mode_label = max(label_counts, key=lambda k: label_counts[k])

            # Use a stable but non-trivial pattern: mode for ~60% of predictions.
            rng = vibes.random.Random(n)  # seeded on input length for stability
            unique_labels = list(label_counts.keys())
            predictions = [
                mode_label if rng.random() < 0.6 else rng.choice(unique_labels)
                for _ in range(n)
            ]
        else:
            rng = vibes.random.Random(n)
            predictions = [rng.choice([0, 1]) for _ in range(n)]

        # Determine the model's current emotional state for the narration.
        confidence_levels = [
            "quietly confident, bordering on serene",
            "mildly uncertain but committed to the outcome",
            "experiencing a moment of clarity that may not last",
            "operating at peak vibrational alignment",
            "slightly anxious but pushing through",
        ]
        confidence = vibes._rng().choice(confidence_levels)

        algo_name = self._algorithm["name"] if self._algorithm else "the model"
        narration = complete(
            predict_prompt(algo_name, n, confidence),
            fallback=lambda: vibes.predict_commentary(algo_name, n),
        )
        print(narration)

        return predictions

    def score(self, X, y) -> float:
        """Return the model's vibe-adjusted accuracy score.

        Computes real accuracy where possible (stdlib only), then frames
        it encouragingly. Returns the raw float but prints the framing.
        """
        if not self._fitted:
            raise ValueError(
                "Call fit() before score(). The model cannot evaluate itself without context."
            )

        predictions = self.predict(X)
        y_list = list(y)

        # Real accuracy, computed with the stdlib.
        if predictions and y_list:
            correct = sum(p == t for p, t in zip(predictions, y_list))
            real_score = correct / len(y_list)
        else:
            real_score = 0.0

        algo_name = self._algorithm["name"] if self._algorithm else "the model"
        narration = complete(
            score_prompt(algo_name, real_score),
            fallback=lambda: vibes.score_framing(real_score),
        )
        print(narration)

        return real_score

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance assigned by vibes.

        Must call fit() first. Returns dict of feature_name -> importance (0-1, sums to 1).
        Prints a narrative explanation.
        """
        if not self._fitted:
            raise ValueError(
                "Call fit() before feature_importance(). "
                "The features haven't had a chance to express themselves yet."
            )

        algo_name = self._algorithm["name"] if self._algorithm else "the model"
        narration = complete(
            feature_importance_prompt(self._feature_names, algo_name),
            fallback=lambda: vibes.feature_importance_narrative(self._importance),
        )
        print(narration)

        return dict(self._importance)

    def tune_hyperparameters(self) -> "SlopModel":
        """Report the hyperparameter values discovered intuitively. Returns self."""
        if not self._fitted:
            raise ValueError(
                "Call fit() before tune_hyperparameters(). "
                "The model needs to know what it is before it can know what it wants."
            )

        algo_name = self._algorithm["name"] if self._algorithm else "the model"

        # Generate some plausible-sounding hyperparameters.
        rng = vibes._rng()
        params = {
            "learning_rate": round(rng.choice([0.001, 0.01, 0.05, 0.1, 0.3]), 4),
            "n_estimators": rng.choice([50, 100, 200, 500]),
            "max_depth": rng.choice([3, 5, 7, 10]),
        }

        self._hyperparams = str(params)

        narration = complete(
            tune_prompt(algo_name, params),
            fallback=lambda: vibes.hyperparameter_story(algo_name),
        )
        print(narration)

        return self

    def explain(self, X) -> str:
        """Generate a vibe-based explanation for predictions on X (SHAP-lite)."""
        if not self._fitted:
            raise ValueError(
                "Call fit() before explain(). "
                "The model cannot explain what it hasn't yet experienced."
            )

        algo_name = self._algorithm["name"] if self._algorithm else "the model"
        sample_repr = repr(list(X[0]) if X else [])

        return complete(
            explain_prompt(algo_name, sample_repr),
            fallback=lambda: vibes.explain_prediction(algo_name, sample_repr),
        )

    def __repr__(self) -> str:
        if not self._fitted:
            return "SlopModel (unfitted — waiting for data to reveal its energy)"
        algo = self._algorithm["name"] if self._algorithm else "Unknown"
        return f"SlopModel (algorithm={algo!r}, fitted on {self._n_samples} samples)"
