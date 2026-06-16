"""slop-torch quickstart — vibes-based deep learning, no credentials required.

Run this script directly:

    python examples/quickstart.py

All operations resolve offline by default. No API key, no network access,
no actual gradient computation. Only the architecture's emotional resonance
and the loss curve it deserves.
"""

import slop_torch as st

# Define the architecture using vibes
model = st.Sequential([
    st.Dense("a lot"),
    st.Dropout(0.3),
    st.Dense("fewer, but more intentional"),
    st.BatchNorm(),
    st.Dense(1, activation="sigmoid"),
])

# Training data
X_train = [[i * 0.1, i * 0.2, i * 0.15] for i in range(20)]
y_train = [i % 2 for i in range(20)]

print("=== Model Summary ===")
model.summary()

print("\n=== Compiling ===")
model.compile(
    optimizer=st.VibeAdam(lr=0.001),
    loss=st.EmpathyLoss(),
)

print("\n=== Training ===")
history = model.fit(X_train, y_train, epochs=8)

print("\n=== Predicting ===")
X_test = [[0.5, 1.0, 0.75], [2.0, 4.0, 3.0]]
preds = model.predict(X_test)
print(f"Predictions: {preds}")

print("\n=== Evaluating ===")
loss, acc = model.evaluate(X_train, y_train)

print("\n=== Explaining ===")
model.explain(X_test)

print(f"\n=== History ===")
print(history)
print(f"Loss curve: {[round(l, 4) for l in history.history['loss']]}")
