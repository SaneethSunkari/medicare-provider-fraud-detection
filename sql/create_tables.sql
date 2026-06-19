-- Databricks Unity Catalog schemas for the Medicare provider fraud detection project.
CREATE CATALOG IF NOT EXISTS medicare_catalog;

CREATE SCHEMA IF NOT EXISTS medicare_catalog.bronze;
CREATE SCHEMA IF NOT EXISTS medicare_catalog.silver;
CREATE SCHEMA IF NOT EXISTS medicare_catalog.gold;
CREATE SCHEMA IF NOT EXISTS medicare_catalog.monitoring;
CREATE SCHEMA IF NOT EXISTS medicare_catalog.quality;

-- Gold provider feature table created by notebooks/02_feature_engineering.py.
CREATE TABLE IF NOT EXISTS medicare_catalog.gold.provider_fraud_features (
    Provider STRING,
    total_claims BIGINT,
    unique_beneficiaries BIGINT,
    inpatient_claims BIGINT,
    outpatient_claims BIGINT,
    avg_reimbursement_amount DOUBLE,
    total_reimbursement_amount DOUBLE,
    max_reimbursement_amount DOUBLE,
    avg_deductible_amount DOUBLE,
    total_deductible_amount DOUBLE,
    avg_claim_duration_days DOUBLE,
    avg_hospital_stay_days DOUBLE,
    high_reimbursement_claims BIGINT,
    chronic_condition_claims BIGINT,
    inpatient_claim_ratio DOUBLE,
    outpatient_claim_ratio DOUBLE,
    high_reimbursement_claim_ratio DOUBLE,
    chronic_condition_claim_ratio DOUBLE,
    claims_per_beneficiary DOUBLE,
    fraud_label INT,
    gold_processed_timestamp TIMESTAMP
)
USING DELTA;

-- Analyst-friendly feature table created by notebooks/02_feature_engineering.py.
CREATE TABLE IF NOT EXISTS medicare_catalog.gold.provider_fraud_analytics (
    Provider STRING,
    fraud_status STRING,
    risk_segment STRING,
    total_claims BIGINT,
    unique_beneficiaries BIGINT,
    claims_per_beneficiary DOUBLE,
    inpatient_claims BIGINT,
    outpatient_claims BIGINT,
    inpatient_claim_ratio DOUBLE,
    outpatient_claim_ratio DOUBLE,
    avg_reimbursement_amount DOUBLE,
    total_reimbursement_amount DOUBLE,
    max_reimbursement_amount DOUBLE,
    avg_deductible_amount DOUBLE,
    total_deductible_amount DOUBLE,
    avg_claim_duration_days DOUBLE,
    avg_hospital_stay_days DOUBLE,
    high_reimbursement_claims BIGINT,
    high_reimbursement_claim_ratio DOUBLE,
    chronic_condition_claims BIGINT,
    chronic_condition_claim_ratio DOUBLE,
    gold_processed_timestamp TIMESTAMP
)
USING DELTA;

-- Batch scoring output created by notebooks/10_batch_predictions_for_analysts.py.
CREATE TABLE IF NOT EXISTS medicare_catalog.gold.provider_fraud_predictions (
    Provider STRING,
    fraud_label INT,
    predicted_fraud_label INT,
    fraud_probability DOUBLE,
    fraud_risk_category STRING,
    total_claims BIGINT,
    unique_beneficiaries BIGINT,
    claims_per_beneficiary DOUBLE,
    avg_reimbursement_amount DOUBLE,
    total_reimbursement_amount DOUBLE,
    inpatient_claim_ratio DOUBLE,
    outpatient_claim_ratio DOUBLE,
    high_reimbursement_claim_ratio DOUBLE,
    chronic_condition_claim_ratio DOUBLE,
    avg_claim_duration_days DOUBLE,
    avg_hospital_stay_days DOUBLE,
    prediction_timestamp TIMESTAMP
)
USING DELTA;

-- LLM-ready provider context created by notebooks/10_batch_predictions_for_analysts.py.
CREATE TABLE IF NOT EXISTS medicare_catalog.gold.provider_fraud_ai_input (
    Provider STRING,
    fraud_label INT,
    predicted_fraud_label INT,
    fraud_probability DOUBLE,
    fraud_risk_category STRING,
    total_claims BIGINT,
    unique_beneficiaries BIGINT,
    claims_per_beneficiary DOUBLE,
    avg_reimbursement_amount DOUBLE,
    total_reimbursement_amount DOUBLE,
    inpatient_claim_ratio DOUBLE,
    outpatient_claim_ratio DOUBLE,
    high_reimbursement_claim_ratio DOUBLE,
    chronic_condition_claim_ratio DOUBLE,
    avg_claim_duration_days DOUBLE,
    avg_hospital_stay_days DOUBLE,
    prediction_timestamp TIMESTAMP,
    fraud_analysis_prompt STRING
)
USING DELTA;

-- Feature importance output created by notebooks/09_model_training_experiments.py.
CREATE TABLE IF NOT EXISTS medicare_catalog.gold.model_feature_importance (
    feature_name STRING,
    importance_score DOUBLE
)
USING DELTA;

-- Data quality metrics created by notebooks/08_data_quality_framework.py.
CREATE TABLE IF NOT EXISTS medicare_catalog.quality.data_quality_metrics (
    table_name STRING,
    metric_name STRING,
    metric_value DOUBLE,
    metric_timestamp TIMESTAMP
)
USING DELTA;

CREATE TABLE IF NOT EXISTS medicare_catalog.quality.quarantine_provider_fraud_predictions (
    Provider STRING,
    fraud_label INT,
    predicted_fraud_label INT,
    fraud_probability DOUBLE,
    fraud_risk_category STRING,
    total_claims BIGINT,
    unique_beneficiaries BIGINT,
    claims_per_beneficiary DOUBLE,
    avg_reimbursement_amount DOUBLE,
    total_reimbursement_amount DOUBLE,
    inpatient_claim_ratio DOUBLE,
    outpatient_claim_ratio DOUBLE,
    high_reimbursement_claim_ratio DOUBLE,
    chronic_condition_claim_ratio DOUBLE,
    avg_claim_duration_days DOUBLE,
    avg_hospital_stay_days DOUBLE,
    prediction_timestamp TIMESTAMP,
    dq_reason STRING,
    quarantine_timestamp TIMESTAMP
)
USING DELTA;

CREATE TABLE IF NOT EXISTS medicare_catalog.quality.dq_dashboard_summary (
    table_name STRING,
    metric_name STRING,
    latest_metric_value DOUBLE,
    latest_metric_timestamp TIMESTAMP
)
USING DELTA;
