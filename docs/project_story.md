# Project Story

## Why This Project Matters

Medicare fraud detection is a strong example of where data engineering, machine learning, and analyst workflows need to work together. Fraud patterns are rarely visible in a single claim. They emerge across provider behavior: claim volume, reimbursement amounts, inpatient versus outpatient mix, beneficiary concentration, chronic-condition patterns, and unusual ratios.

This project turns raw Medicare data into a practical fraud risk workflow that can help analysts focus attention on providers with the strongest risk signals.

## End-to-End Workflow

### 1. Organizing the Lakehouse

The project starts by organizing the Databricks environment around Unity Catalog schemas:

- `bronze` for raw source data
- `silver` for cleaned and validated records
- `gold` for analytics, ML features, predictions, and GenAI input
- `monitoring` and `quality` for data quality outputs

This separation makes the workflow easier to operate, audit, and explain to both technical and business stakeholders.

### 2. Preparing the Silver Layer

The Silver layer cleans the raw Medicare data into reliable analytical tables.

Beneficiary records are deduplicated, dates are parsed, age is calculated, deceased status is derived, and invalid records are separated into quarantine tables.

Inpatient and outpatient claims are standardized by parsing claim dates, casting reimbursement fields, calculating claim duration, and validating business-critical fields such as `ClaimID`, `Provider`, and `BeneID`.

Inpatient and outpatient claims are then combined into a single `unified_claims` table with a `claim_type` column. This creates a consistent claim-level foundation for provider analytics.

Provider labels are cleaned by converting `PotentialFraud` values into a binary `fraud_label`.

### 3. Engineering Provider-Level Fraud Features

The Gold feature engineering step joins claims, beneficiaries, and provider labels, then aggregates to the provider level.

The resulting `provider_fraud_features` table captures the behaviors needed for fraud detection:

- How many claims each provider submitted
- How many unique beneficiaries each provider served
- Whether the provider is concentrated in inpatient or outpatient claims
- Average, total, and maximum reimbursement amount
- Average and total deductible amounts
- Average claim duration and hospital stay
- High-reimbursement claim frequency
- Chronic-condition claim frequency
- Ratios that normalize behavior across providers of different sizes

This table becomes the central feature store-style asset for model training, scoring, analytics, and GenAI context generation.

### 4. Adding Data Quality Controls

The project includes a reusable data quality framework that checks feature and prediction tables for:

- Duplicate provider records
- Missing primary keys
- Missing required values
- Negative numeric values
- Invalid ratio values
- Invalid fraud probabilities

Quality results are stored in Delta tables so they can be queried, monitored, and visualized. Invalid prediction records are written to quarantine for follow-up instead of being silently mixed into analyst outputs.

### 5. Training and Tracking Fraud Models

The ML workflow trains Random Forest classifiers using the provider-level Gold features.

MLflow is used to track:

- Model parameters
- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Model artifacts
- Model signature
- Input examples

Two Random Forest configurations are trained and compared. The best run is selected by ROC AUC, which is a useful metric when ranking fraud risk because analysts often care about prioritizing suspicious providers.

### 6. Registering the Best Model

After model comparison, the best run is registered as:

```text
medicare_catalog.gold.medicare_provider_fraud_model
```

The model version is documented with a description and project tags. The selected version receives the `champion` alias, allowing downstream jobs to load the best production-intended model without hard-coding a version number.

This is an important MLOps pattern because batch scoring and serving workflows can keep using:

```text
models:/medicare_catalog.gold.medicare_provider_fraud_model@champion
```

while the underlying champion version can be updated through model governance.

### 7. Deploying and Scoring

The champion model is designed for deployment through Databricks Model Serving. A serving endpoint can support low-latency fraud scoring for applications, dashboards, or investigation tools.

The project also includes a batch scoring workflow that loads the champion model, scores all providers in the Gold feature table, calculates fraud probability, assigns a risk category, and writes analyst-ready results to:

```text
medicare_catalog.gold.provider_fraud_predictions
```

Providers are grouped into:

- `High Risk`
- `Medium Risk`
- `Low Risk`

This gives analysts a simple prioritization layer while preserving the detailed features behind each score.

### 8. Preparing for GenAI Analysis

The final project step creates:

```text
medicare_catalog.gold.provider_fraud_ai_input
```

This table converts each provider’s prediction and key feature context into a structured prompt for LLM analysis.

The prompt includes provider ID, fraud probability, total claims, average reimbursement, claims per beneficiary, and risk category. This format is useful for Databricks Playground testing because it keeps LLM analysis grounded in structured model outputs.

The same table can support future Databricks Agent workflows where an AI application retrieves provider features, predictions, and quality context before generating an investigation summary.

## What Recruiters and Hiring Managers Should Notice

This project shows practical ownership across the full data-to-ML lifecycle:

- Building Lakehouse pipelines with PySpark and Delta Lake
- Designing provider-level features for a realistic fraud detection use case
- Implementing quality checks and quarantine logic
- Training and tracking machine learning models with MLflow
- Registering and promoting models with Databricks Model Registry
- Preparing serving and batch scoring patterns
- Creating analyst-ready SQL outputs
- Extending the workflow into GenAI and Agent-ready investigation support

The result is not just a model notebook. It is a complete, explainable fraud detection system that connects data engineering, ML engineering, analytics, and GenAI experimentation.
