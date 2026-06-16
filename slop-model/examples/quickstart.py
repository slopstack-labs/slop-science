"""slop-model quickstart — vibe-driven AutoML, no credentials required.

Run this script directly:

    python examples/quickstart.py

All operations resolve offline by default. No API key, no network access,
no cross-validation. Only the data's energy and the algorithm it deserves.
"""

import slop_model as sm

model = sm.SlopModel()

# Training data with vibes
X_train = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [5, 6, 7], [8, 9, 10]]
y_train = [0, 1, 0, 0, 1, 1]
feature_names = ["engagement_score", "vibe_index", "synergy_coefficient"]

print("=== Fitting the Model ===")
model.fit(X_train, y_train, feature_names=feature_names)

print("\n=== Hyperparameter Tuning ===")
model.tune_hyperparameters()

print("\n=== Predictions ===")
X_test = [[3, 4, 5], [6, 7, 8]]
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")

print("\n=== Score ===")
score = model.score(X_train, y_train)
print(f"Raw score: {score:.3f}")

print("\n=== Feature Importance ===")
importance = model.feature_importance()
for feature, score_val in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"  {feature}: {score_val:.3f}")

print("\n=== Explain ===")
print(model.explain(X_test[:1]))
