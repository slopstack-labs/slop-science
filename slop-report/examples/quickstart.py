"""slop-report quickstart — executive data science reporting, no credentials required.

Run this script directly:

    python examples/quickstart.py

All operations resolve offline by default. No API key, no network access,
no accountability. Only synergy.
"""

import slop_report as sr

print("=== Executive Summary ===")
metrics = {"model_accuracy": 0.73, "data_processed_gb": 142, "features_engineered": 847}
print(sr.executive_summary(metrics=metrics, title="Q4 Model Performance Report"))

print("\n=== Data-Driven Insights ===")
for i, insight in enumerate(sr.insights(n=4), 1):
    print(f"  {i}. {insight}")

print("\n=== KPI Report ===")
kpis = {"F1 Score": 0.61, "Precision": 0.58, "Recall": 0.65, "Data Quality Score": -12}
print(sr.kpi_report(kpis))

print("\n=== Stakeholder Email ===")
print(sr.email(
    title="Model Deployment Update",
    recipient_role="VP of Product",
    key_finding="our churn model is now in production"
))

print("\n=== Strategic Recommendations ===")
for rec in sr.recommendations("We have a model with 61% F1 score"):
    print(f"  • {rec}")
