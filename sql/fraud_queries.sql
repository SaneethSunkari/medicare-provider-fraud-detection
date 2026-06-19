-- Top providers by fraud probability.
SELECT
    Provider,
    fraud_probability,
    fraud_risk_category,
    total_claims,
    unique_beneficiaries,
    claims_per_beneficiary,
    avg_reimbursement_amount,
    total_reimbursement_amount,
    high_reimbursement_claim_ratio,
    chronic_condition_claim_ratio
FROM medicare_catalog.gold.provider_fraud_predictions
ORDER BY fraud_probability DESC
LIMIT 25;

-- High-risk providers for analyst review.
SELECT *
FROM medicare_catalog.gold.provider_fraud_predictions
WHERE fraud_risk_category = 'High Risk'
ORDER BY fraud_probability DESC;

-- Risk category distribution.
SELECT
    fraud_risk_category,
    COUNT(*) AS provider_count,
    AVG(fraud_probability) AS avg_fraud_probability,
    AVG(total_claims) AS avg_total_claims,
    AVG(avg_reimbursement_amount) AS avg_reimbursement_amount
FROM medicare_catalog.gold.provider_fraud_predictions
GROUP BY fraud_risk_category
ORDER BY avg_fraud_probability DESC;

-- Actual fraud label compared with model prediction.
SELECT
    fraud_label,
    predicted_fraud_label,
    COUNT(*) AS provider_count
FROM medicare_catalog.gold.provider_fraud_predictions
GROUP BY fraud_label, predicted_fraud_label
ORDER BY fraud_label, predicted_fraud_label;

-- Providers with unusually concentrated high-reimbursement claims.
SELECT
    Provider,
    fraud_probability,
    fraud_risk_category,
    total_claims,
    high_reimbursement_claim_ratio,
    avg_reimbursement_amount,
    max_reimbursement_amount,
    claims_per_beneficiary
FROM medicare_catalog.gold.provider_fraud_features
WHERE total_claims >= 10
ORDER BY high_reimbursement_claim_ratio DESC, avg_reimbursement_amount DESC
LIMIT 50;

-- Feature importance from the registered model training workflow.
SELECT
    feature_name,
    importance_score
FROM medicare_catalog.gold.model_feature_importance
ORDER BY importance_score DESC;

-- LLM-ready provider prompts for Databricks Playground testing.
SELECT
    Provider,
    fraud_probability,
    fraud_risk_category,
    fraud_analysis_prompt
FROM medicare_catalog.gold.provider_fraud_ai_input
ORDER BY fraud_probability DESC
LIMIT 20;

-- Latest data quality summary.
SELECT
    table_name,
    metric_name,
    latest_metric_value,
    latest_metric_timestamp
FROM medicare_catalog.quality.dq_dashboard_summary
ORDER BY table_name, metric_name;

-- Prediction records that failed quality checks.
SELECT *
FROM medicare_catalog.quality.quarantine_provider_fraud_predictions
ORDER BY quarantine_timestamp DESC;
