# Databricks notebook source
# MAGIC %md
# MAGIC # 10 - Batch Predictions for Analysts and GenAI Input
# MAGIC
# MAGIC Loads the registered champion model, scores provider-level fraud risk, writes analyst-ready predictions, and prepares LLM-ready provider context.

# COMMAND ----------

import mlflow
import pandas as pd
from pyspark.sql.functions import *

# COMMAND ----------

gold_df = spark.table("medicare_catalog.gold.provider_fraud_features")

display(gold_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Select Model Features
feature_cols = [
    "total_claims",
    "unique_beneficiaries",
    "inpatient_claims",
    "outpatient_claims",
    "avg_reimbursement_amount",
    "total_reimbursement_amount",
    "max_reimbursement_amount",
    "avg_deductible_amount",
    "total_deductible_amount",
    "avg_claim_duration_days",
    "avg_hospital_stay_days",
    "high_reimbursement_claims",
    "chronic_condition_claims",
    "inpatient_claim_ratio",
    "outpatient_claim_ratio",
    "high_reimbursement_claim_ratio",
    "chronic_condition_claim_ratio",
    "claims_per_beneficiary"
]

# COMMAND ----------

model_uri = "models:/medicare_catalog.gold.medicare_provider_fraud_model@champion"

model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

scoring_pdf = (
    gold_df
    .select(["Provider"] + feature_cols)
    .fillna(0)
    .toPandas()
)

# COMMAND ----------

X_score = scoring_pdf[feature_cols]

predicted_label = model.predict(X_score)

scoring_pdf["predicted_fraud_label"] = predicted_label

# COMMAND ----------

sk_model = mlflow.sklearn.load_model(model_uri)

fraud_probability = sk_model.predict_proba(X_score)[:, 1]

scoring_pdf["fraud_probability"] = fraud_probability

# COMMAND ----------

def risk_category(prob):
    if prob >= 0.80:
        return "High Risk"
    elif prob >= 0.50:
        return "Medium Risk"
    else:
        return "Low Risk"

scoring_pdf["fraud_risk_category"] = scoring_pdf["fraud_probability"].apply(risk_category)

# COMMAND ----------

predictions_df = spark.createDataFrame(scoring_pdf)

# COMMAND ----------

analyst_predictions = (
    predictions_df
    .join(
        gold_df.select("Provider", "fraud_label"),
        on="Provider",
        how="left"
    )
    .withColumn("prediction_timestamp", current_timestamp())
    .select(
        "Provider",
        "fraud_label",
        "predicted_fraud_label",
        "fraud_probability",
        "fraud_risk_category",
        "total_claims",
        "unique_beneficiaries",
        "claims_per_beneficiary",
        "avg_reimbursement_amount",
        "total_reimbursement_amount",
        "inpatient_claim_ratio",
        "outpatient_claim_ratio",
        "high_reimbursement_claim_ratio",
        "chronic_condition_claim_ratio",
        "avg_claim_duration_days",
        "avg_hospital_stay_days",
        "prediction_timestamp"
    )
)

# COMMAND ----------

analyst_predictions.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.gold.provider_fraud_predictions")

# COMMAND ----------

display(
    spark.table(
        "medicare_catalog.gold.provider_fraud_predictions"
    ).limit(20)
)

spark.table(
    "medicare_catalog.gold.provider_fraud_predictions"
).groupBy(
    "fraud_risk_category"
).count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Analyst SQL Queries

# COMMAND ----------

# DBTITLE 1,High-Risk Providers
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM medicare_catalog.gold.provider_fraud_predictions
# MAGIC WHERE fraud_risk_category = 'High Risk'
# MAGIC ORDER BY fraud_probability DESC;

# COMMAND ----------

# DBTITLE 1,Top 20 Suspicious Providers
# MAGIC %sql
# MAGIC SELECT 
# MAGIC     Provider,
# MAGIC     fraud_probability,
# MAGIC     fraud_risk_category,
# MAGIC     total_claims,
# MAGIC     avg_reimbursement_amount,
# MAGIC     high_reimbursement_claim_ratio,
# MAGIC     claims_per_beneficiary
# MAGIC FROM medicare_catalog.gold.provider_fraud_predictions
# MAGIC ORDER BY fraud_probability DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,Compare Actual vs Predicted
# MAGIC %sql
# MAGIC SELECT 
# MAGIC     fraud_label,
# MAGIC     predicted_fraud_label,
# MAGIC     COUNT(*) AS provider_count
# MAGIC FROM medicare_catalog.gold.provider_fraud_predictions
# MAGIC GROUP BY fraud_label, predicted_fraud_label
# MAGIC ORDER BY fraud_label, predicted_fraud_label;

# COMMAND ----------

# Databricks notebook source
from pyspark.sql.functions import *

predictions = spark.table(
    "medicare_catalog.gold.provider_fraud_predictions"
)

llm_ready = predictions.withColumn(
    "fraud_analysis_prompt",
    concat(
        lit("Analyze this healthcare provider.\n\n"),
        lit("Provider: "), col("Provider"),
        lit("\nFraud Probability: "), col("fraud_probability"),
        lit("\nTotal Claims: "), col("total_claims"),
        lit("\nAverage Reimbursement: "), col("avg_reimbursement_amount"),
        lit("\nClaims Per Beneficiary: "), col("claims_per_beneficiary"),
        lit("\nRisk Category: "), col("fraud_risk_category"),
        lit("\n\nExplain why this provider may be suspicious.")
    )
)

llm_ready.write.mode("overwrite").saveAsTable(
    "medicare_catalog.gold.provider_fraud_ai_input"
)

display(llm_ready)

# COMMAND ----------
