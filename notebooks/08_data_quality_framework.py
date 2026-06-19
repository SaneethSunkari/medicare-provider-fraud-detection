# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Data Quality Framework
# MAGIC
# MAGIC Runs reusable quality checks across Gold feature and prediction tables, writes quality metrics, and quarantines invalid provider prediction records.

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

features_df = spark.table("medicare_catalog.gold.provider_fraud_features")
predictions_df = spark.table("medicare_catalog.gold.provider_fraud_predictions")

display(features_df.limit(5))
display(predictions_df.limit(5))

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS medicare_catalog.quality")

# COMMAND ----------

def run_dq_checks(df, table_name, primary_key, required_columns, numeric_non_negative_cols, ratio_cols):
    total_records = df.count()

    metrics = []

    # total records
    metrics.append((table_name, "total_records", float(total_records)))

    # duplicate primary key
    duplicate_count = (
        df.groupBy(primary_key)
        .count()
        .filter(col("count") > 1)
        .count()
    )
    metrics.append((table_name, "duplicate_primary_key_count", float(duplicate_count)))

    # missing primary key
    missing_pk_count = df.filter(col(primary_key).isNull()).count()
    metrics.append((table_name, "missing_primary_key_count", float(missing_pk_count)))

    # required column null checks
    for c in required_columns:
        null_count = df.filter(col(c).isNull()).count()
        null_percent = (null_count / total_records * 100) if total_records > 0 else 0
        metrics.append((table_name, f"null_count_{c}", float(null_count)))
        metrics.append((table_name, f"null_percent_{c}", float(null_percent)))

    # non-negative checks
    for c in numeric_non_negative_cols:
        invalid_count = df.filter(col(c) < 0).count()
        metrics.append((table_name, f"negative_value_count_{c}", float(invalid_count)))

    # ratio checks: must be between 0 and 1
    for c in ratio_cols:
        invalid_count = df.filter((col(c) < 0) | (col(c) > 1)).count()
        metrics.append((table_name, f"invalid_ratio_count_{c}", float(invalid_count)))

    return spark.createDataFrame(
        metrics,
        ["table_name", "metric_name", "metric_value"]
    ).withColumn("metric_timestamp", current_timestamp())

# COMMAND ----------

feature_required_columns = [
    "Provider",
    "total_claims",
    "unique_beneficiaries",
    "avg_reimbursement_amount",
    "total_reimbursement_amount",
    "claims_per_beneficiary",
    "fraud_label"
]

feature_non_negative_cols = [
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
    "claims_per_beneficiary"
]

feature_ratio_cols = [
    "inpatient_claim_ratio",
    "outpatient_claim_ratio",
    "high_reimbursement_claim_ratio",
    "chronic_condition_claim_ratio"
]

feature_dq_metrics = run_dq_checks(
    df=features_df,
    table_name="provider_fraud_features",
    primary_key="Provider",
    required_columns=feature_required_columns,
    numeric_non_negative_cols=feature_non_negative_cols,
    ratio_cols=feature_ratio_cols
)

display(feature_dq_metrics)

# COMMAND ----------

prediction_required_columns = [
    "Provider",
    "fraud_label",
    "predicted_fraud_label",
    "fraud_probability",
    "fraud_risk_category",
    "total_claims",
    "avg_reimbursement_amount",
    "claims_per_beneficiary"
]

prediction_non_negative_cols = [
    "fraud_probability",
    "total_claims",
    "unique_beneficiaries",
    "claims_per_beneficiary",
    "avg_reimbursement_amount",
    "total_reimbursement_amount"
]

prediction_ratio_cols = [
    "fraud_probability"
]

prediction_dq_metrics = run_dq_checks(
    df=predictions_df,
    table_name="provider_fraud_predictions",
    primary_key="Provider",
    required_columns=prediction_required_columns,
    numeric_non_negative_cols=prediction_non_negative_cols,
    ratio_cols=prediction_ratio_cols
)

display(prediction_dq_metrics)

# COMMAND ----------

all_dq_metrics = feature_dq_metrics.unionByName(prediction_dq_metrics)

all_dq_metrics.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("medicare_catalog.quality.data_quality_metrics")

display(spark.table("medicare_catalog.quality.data_quality_metrics"))

# COMMAND ----------

invalid_predictions = (
    predictions_df
    .withColumn(
        "dq_reason",
        concat_ws(
            ", ",
            when(col("Provider").isNull(), lit("Missing Provider")),
            when(col("fraud_probability").isNull(), lit("Missing Fraud Probability")),
            when((col("fraud_probability") < 0) | (col("fraud_probability") > 1), lit("Invalid Fraud Probability")),
            when(col("total_claims") < 0, lit("Negative Total Claims")),
            when(col("avg_reimbursement_amount") < 0, lit("Negative Avg Reimbursement")),
            when(col("claims_per_beneficiary") < 0, lit("Negative Claims Per Beneficiary"))
        )
    )
    .filter(col("dq_reason") != "")
    .withColumn("quarantine_timestamp", current_timestamp())
)

invalid_predictions.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("medicare_catalog.quality.quarantine_provider_fraud_predictions")

display(invalid_predictions)

# COMMAND ----------

dq_dashboard_summary = (
    spark.table("medicare_catalog.quality.data_quality_metrics")
    .groupBy("table_name", "metric_name")
    .agg(
        max("metric_timestamp").alias("latest_metric_timestamp"),
        last("metric_value").alias("latest_metric_value")
    )
)

dq_dashboard_summary.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("medicare_catalog.quality.dq_dashboard_summary")

display(dq_dashboard_summary)

# COMMAND ----------

print("DQ metrics count:", spark.table("medicare_catalog.quality.data_quality_metrics").count())
print("Quarantine records:", spark.table("medicare_catalog.quality.quarantine_provider_fraud_predictions").count())

display(spark.table("medicare_catalog.quality.dq_dashboard_summary"))

# COMMAND ----------

# Databricks notebook source
from pyspark.sql.functions import *

spark.sql("USE CATALOG medicare_catalog")
spark.sql("CREATE SCHEMA IF NOT EXISTS monitoring")

# COMMAND ----------

gold_df = spark.table("medicare_catalog.gold.provider_fraud_features")

display(gold_df.limit(5))
gold_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Basic Gold-Level Validation
total_records = gold_df.count()

missing_provider = gold_df.filter(col("Provider").isNull()).count()

missing_fraud_label = gold_df.filter(col("fraud_label").isNull()).count()

duplicate_provider = (
    gold_df.groupBy("Provider")
    .count()
    .filter(col("count") > 1)
    .count()
)

negative_total_claims = gold_df.filter(col("total_claims") < 0).count()

invalid_ratios = gold_df.filter(
    (col("inpatient_claim_ratio") < 0) |
    (col("inpatient_claim_ratio") > 1) |
    (col("outpatient_claim_ratio") < 0) |
    (col("outpatient_claim_ratio") > 1) |
    (col("high_reimbursement_claim_ratio") < 0) |
    (col("high_reimbursement_claim_ratio") > 1) |
    (col("chronic_condition_claim_ratio") < 0) |
    (col("chronic_condition_claim_ratio") > 1)
).count()

# COMMAND ----------

# DBTITLE 1,Create Gold DQ Metrics Table
gold_dq_metrics = spark.createDataFrame(
    [
        ("gold_provider_features", "total_records", total_records),
        ("gold_provider_features", "missing_provider", missing_provider),
        ("gold_provider_features", "missing_fraud_label", missing_fraud_label),
        ("gold_provider_features", "duplicate_provider", duplicate_provider),
        ("gold_provider_features", "negative_total_claims", negative_total_claims),
        ("gold_provider_features", "invalid_ratio_values", invalid_ratios)
    ],
    ["table_name", "metric_name", "metric_value"]
)

gold_dq_metrics = gold_dq_metrics.withColumn(
    "metric_timestamp",
    current_timestamp()
)

# COMMAND ----------

gold_dq_metrics.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("medicare_catalog.monitoring.data_quality_metrics")

# COMMAND ----------

# DBTITLE 1,Add Fraud Label Distribution Metrics
fraud_distribution = (
    gold_df
    .groupBy("fraud_label")
    .count()
    .withColumn("table_name", lit("gold_provider_features"))
    .withColumn("metric_name", concat(lit("fraud_label_"), col("fraud_label")))
    .withColumnRenamed("count", "metric_value")
    .select("table_name", "metric_name", "metric_value")
    .withColumn("metric_timestamp", current_timestamp())
)

fraud_distribution.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("medicare_catalog.monitoring.data_quality_metrics")

# COMMAND ----------

# DBTITLE 1,Add Null Percentage Metrics
from pyspark.sql import functions as F

metrics_df = spark.table("medicare_catalog.monitoring.data_quality_metrics")

latest_timestamp_df = (
    metrics_df
    .groupBy("table_name", "metric_name")
    .agg(
        F.max("metric_timestamp").alias("latest_metric_timestamp")
    )
)

dq_dashboard_summary = (
    metrics_df
    .join(
        latest_timestamp_df,
        on=["table_name", "metric_name"],
        how="inner"
    )
    .filter(
        F.col("metric_timestamp") == F.col("latest_metric_timestamp")
    )
    .select(
        "table_name",
        "metric_name",
        F.col("metric_value").alias("latest_metric_value"),
        "latest_metric_timestamp"
    )
)

dq_dashboard_summary.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.monitoring.dq_dashboard_summary")

# COMMAND ----------

display(
    spark.table(
        "medicare_catalog.monitoring.data_quality_metrics"
    )
)

display(
    spark.table(
        "medicare_catalog.monitoring.dq_dashboard_summary"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Dashboard SQL Queries

# COMMAND ----------

# DBTITLE 1,Latest DQ Summary
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM medicare_catalog.monitoring.dq_dashboard_summary
# MAGIC ORDER BY table_name, metric_name;

# COMMAND ----------

# DBTITLE 1,Quarantine Counts
# MAGIC %sql
# MAGIC SELECT table_name, metric_name, metric_value, metric_timestamp
# MAGIC FROM medicare_catalog.monitoring.data_quality_metrics
# MAGIC WHERE metric_name LIKE '%quarantine%'
# MAGIC ORDER BY metric_timestamp DESC;

# COMMAND ----------

# DBTITLE 1,Gold Null Percentages
# MAGIC %sql
# MAGIC SELECT table_name, metric_name, metric_value
# MAGIC FROM medicare_catalog.monitoring.data_quality_metrics
# MAGIC WHERE metric_name LIKE 'null_percent%'
# MAGIC ORDER BY metric_value DESC;

# COMMAND ----------

# DBTITLE 1,Fraud Label Distribution
# MAGIC %sql
# MAGIC SELECT metric_name, metric_value
# MAGIC FROM medicare_catalog.monitoring.data_quality_metrics
# MAGIC WHERE metric_name LIKE 'fraud_label_%';

# COMMAND ----------
