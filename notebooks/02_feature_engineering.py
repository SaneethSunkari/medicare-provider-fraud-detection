# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Feature Engineering
# MAGIC
# MAGIC Builds the Silver cleansed tables and Gold provider-level fraud features used by analytics, MLflow experiments, batch scoring, and GenAI workflows.

# COMMAND ----------

from pyspark.sql.functions import *


spark.sql("USE CATALOG medicare_catalog")

spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS monitoring")

# COMMAND ----------

bronze_beneficiary = spark.table("medicare_catalog.bronze.beneficiarydata")

display(bronze_beneficiary.limit(5))

# COMMAND ----------

bronze_beneficiary.printSchema()

# COMMAND ----------

silver_beneficiary = (
    bronze_beneficiary
    .dropDuplicates(["BeneID"])
    .withColumn("DOB", to_date(col("DOB"), "yyyy-MM-dd"))
    .withColumn(
        "DOD",
        when((col("DOD") == "NA") | (col("DOD").isNull()), None)
        .otherwise(to_date(col("DOD"), "yyyy-MM-dd"))
    )
    .withColumn(
        "age",
        floor(datediff(current_date(), col("DOB")) / 365.25)
    )
    .withColumn(
        "is_deceased",
        when(col("DOD").isNotNull(), lit(1)).otherwise(lit(0))
    )
    .withColumn(
        "gender_desc",
        when(col("Gender") == 1, "Male")
        .when(col("Gender") == 2, "Female")
        .otherwise("Unknown")
    )
    .withColumn(
        "race_desc",
        col("Race").cast("string")
    )
    .withColumn(
        "silver_processed_timestamp",
        current_timestamp()
    )
)

# COMMAND ----------

display(
    silver_beneficiary.select(
        "BeneID",
        "Gender",
        "gender_desc",
        "Race"
    ).limit(10)
)

# COMMAND ----------

# DBTITLE 1,Create Data Quality Flags
silver_beneficiary_dq = (
    silver_beneficiary
    .withColumn(
        "dq_missing_bene_id",
        when(col("BeneID").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_invalid_dob",
        when(col("DOB").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_invalid_age",
        when(
            (col("age") < 0) |
            (col("age") > 120),
            1
        ).otherwise(0)
    )
    .withColumn(
        "dq_is_valid",
        when(
            (col("dq_missing_bene_id") == 0) &
            (col("dq_invalid_dob") == 0) &
            (col("dq_invalid_age") == 0),
            1
        ).otherwise(0)
    )
)

# COMMAND ----------

# DBTITLE 1,Split Valid and Quarantine Records
valid_beneficiary = (
    silver_beneficiary_dq
    .filter(col("dq_is_valid") == 1)
)

quarantine_beneficiary = (
    silver_beneficiary_dq
    .filter(col("dq_is_valid") == 0)
    .withColumn(
        "quarantine_reason",
        concat_ws(
            ", ",
            when(col("dq_missing_bene_id") == 1, "Missing BeneID"),
            when(col("dq_invalid_dob") == 1, "Invalid DOB"),
            when(col("dq_invalid_age") == 1, "Invalid Age")
        )
    )
)

# COMMAND ----------

# DBTITLE 1,Save Valid Beneficiary
valid_beneficiary.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.silver.beneficiary")

# COMMAND ----------

# DBTITLE 1,Save Quarantine Beneficiary
quarantine_beneficiary.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.monitoring.quarantine_beneficiary")

# COMMAND ----------

display(
    spark.table(
        "medicare_catalog.silver.beneficiary"
    ).limit(10)
)

display(
    quarantine_beneficiary
)

print("Valid beneficiary records:", valid_beneficiary.count())
print("Quarantine beneficiary records:", quarantine_beneficiary.count())

# COMMAND ----------

# DBTITLE 1,Data Quality Metrics
beneficiary_dq_metrics = spark.createDataFrame(
    [
        (
            "beneficiary",
            "total_records",
            bronze_beneficiary.count()
        ),
        (
            "beneficiary",
            "valid_records",
            valid_beneficiary.count()
        ),
        (
            "beneficiary",
            "quarantine_records",
            quarantine_beneficiary.count()
        ),
        (
            "beneficiary",
            "duplicate_beneid_removed",
            bronze_beneficiary.count() - silver_beneficiary.count()
        )
    ],
    ["table_name", "metric_name", "metric_value"]
)

beneficiary_dq_metrics = (
    beneficiary_dq_metrics
    .withColumn(
        "metric_timestamp",
        current_timestamp()
    )
)

beneficiary_dq_metrics.write \
    .format("delta") \
    .mode("append") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.monitoring.data_quality_metrics")

# COMMAND ----------

# DBTITLE 1,Verify Metrics
display(
    spark.table(
        "medicare_catalog.monitoring.data_quality_metrics"
    )
)

# COMMAND ----------

# Databricks notebook source
from pyspark.sql.functions import *

spark.sql("USE CATALOG medicare_catalog")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS monitoring")

# COMMAND ----------

bronze_inpatient = spark.table("medicare_catalog.bronze.inpatientdata")

display(bronze_inpatient.limit(5))
bronze_inpatient.printSchema()

# COMMAND ----------

silver_inpatient = (
    bronze_inpatient
    .dropDuplicates(["ClaimID"])
    .withColumn("ClaimStartDt", to_date(col("ClaimStartDt"), "yyyy-MM-dd"))
    .withColumn("ClaimEndDt", to_date(col("ClaimEndDt"), "yyyy-MM-dd"))
    .withColumn("AdmissionDt", to_date(col("AdmissionDt"), "yyyy-MM-dd"))
    .withColumn("DischargeDt", to_date(col("DischargeDt"), "yyyy-MM-dd"))
    .withColumn("InscClaimAmtReimbursed", col("InscClaimAmtReimbursed").cast("double"))
    .withColumn("DeductibleAmtPaid", expr("try_cast(DeductibleAmtPaid as double)"))
    .withColumn(
        "claim_duration_days",
        datediff(col("ClaimEndDt"), col("ClaimStartDt")) + 1
    )
    .withColumn(
        "hospital_stay_days",
        datediff(col("DischargeDt"), col("AdmissionDt")) + 1
    )
    .withColumn("silver_processed_timestamp", current_timestamp())
)

# COMMAND ----------

# DBTITLE 1,Create Data Quality Flags
silver_inpatient_dq = (
    silver_inpatient
    .withColumn(
        "dq_missing_claim_id",
        when(col("ClaimID").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_missing_provider",
        when(col("Provider").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_missing_bene_id",
        when(col("BeneID").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_invalid_claim_dates",
        when(
            col("ClaimStartDt").isNull() |
            col("ClaimEndDt").isNull() |
            (col("ClaimEndDt") < col("ClaimStartDt")),
            1
        ).otherwise(0)
    )
    .withColumn(
        "dq_invalid_admission_dates",
        when(
            col("AdmissionDt").isNull() |
            col("DischargeDt").isNull() |
            (col("DischargeDt") < col("AdmissionDt")),
            1
        ).otherwise(0)
    )
    .withColumn(
        "dq_negative_reimbursement",
        when(col("InscClaimAmtReimbursed") < 0, 1).otherwise(0)
    )
    .withColumn(
        "dq_negative_deductible",
        when(col("DeductibleAmtPaid") < 0, 1).otherwise(0)
    )
    .withColumn(
        "dq_is_valid",
        when(
            (col("dq_missing_claim_id") == 0) &
            (col("dq_missing_provider") == 0) &
            (col("dq_missing_bene_id") == 0) &
            (col("dq_invalid_claim_dates") == 0) &
            (col("dq_invalid_admission_dates") == 0) &
            (col("dq_negative_reimbursement") == 0) &
            (col("dq_negative_deductible") == 0),
            1
        ).otherwise(0)
    )
)


# COMMAND ----------

# DBTITLE 1,Split Valid and Quarantine Records
valid_inpatient = (
    silver_inpatient_dq
    .filter(col("dq_is_valid") == 1)
)

quarantine_inpatient = (
    silver_inpatient_dq
    .filter(col("dq_is_valid") == 0)
    .withColumn(
        "quarantine_reason",
        concat_ws(
            ", ",
            when(col("dq_missing_claim_id") == 1, "Missing ClaimID"),
            when(col("dq_missing_provider") == 1, "Missing Provider"),
            when(col("dq_missing_bene_id") == 1, "Missing BeneID"),
            when(col("dq_invalid_claim_dates") == 1, "Invalid Claim Dates"),
            when(col("dq_invalid_admission_dates") == 1, "Invalid Admission/Discharge Dates"),
            when(col("dq_negative_reimbursement") == 1, "Negative Reimbursement"),
            when(col("dq_negative_deductible") == 1, "Negative Deductible")
        )
    )
)

# COMMAND ----------

valid_inpatient.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.silver.inpatient_claims")

# COMMAND ----------

quarantine_inpatient.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.monitoring.quarantine_inpatient_claims")

# COMMAND ----------

# DBTITLE 1,Add Data Quality Metrics
inpatient_dq_metrics = spark.createDataFrame(
    [
        (
            "inpatient_claims",
            "total_records",
            bronze_inpatient.count()
        ),
        (
            "inpatient_claims",
            "valid_records",
            valid_inpatient.count()
        ),
        (
            "inpatient_claims",
            "quarantine_records",
            quarantine_inpatient.count()
        ),
        (
            "inpatient_claims",
            "duplicate_claimid_removed",
            bronze_inpatient.count() - silver_inpatient.count()
        )
    ],
    ["table_name", "metric_name", "metric_value"]
)

inpatient_dq_metrics = inpatient_dq_metrics.withColumn(
    "metric_timestamp",
    current_timestamp()
)

inpatient_dq_metrics.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("medicare_catalog.monitoring.data_quality_metrics")

# COMMAND ----------

display(spark.table("medicare_catalog.silver.inpatient_claims").limit(10))

display(spark.table("medicare_catalog.monitoring.quarantine_inpatient_claims"))

display(spark.table("medicare_catalog.monitoring.data_quality_metrics"))

print("Total inpatient records:", bronze_inpatient.count())
print("Valid inpatient records:", valid_inpatient.count())
print("Quarantine inpatient records:", quarantine_inpatient.count())

# COMMAND ----------

# Databricks notebook source
from pyspark.sql.functions import *

spark.sql("USE CATALOG medicare_catalog")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS monitoring")

# COMMAND ----------

bronze_outpatient = spark.table("medicare_catalog.bronze.outpatientdata")

display(bronze_outpatient.limit(5))
bronze_outpatient.printSchema()

# COMMAND ----------

silver_outpatient = (
    bronze_outpatient
    .dropDuplicates(["ClaimID"])
    .withColumn("ClaimStartDt", to_date(col("ClaimStartDt"), "yyyy-MM-dd"))
    .withColumn("ClaimEndDt", to_date(col("ClaimEndDt"), "yyyy-MM-dd"))
    .withColumn("InscClaimAmtReimbursed", col("InscClaimAmtReimbursed").cast("double"))
    .withColumn("DeductibleAmtPaid", col("DeductibleAmtPaid").cast("double"))
    .withColumn(
        "claim_duration_days",
        datediff(col("ClaimEndDt"), col("ClaimStartDt")) + 1
    )
    .withColumn("silver_processed_timestamp", current_timestamp())
)

# COMMAND ----------

# DBTITLE 1,Create Data Quality Flags
silver_outpatient_dq = (
    silver_outpatient
    .withColumn(
        "dq_missing_claim_id",
        when(col("ClaimID").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_missing_provider",
        when(col("Provider").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_missing_bene_id",
        when(col("BeneID").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_invalid_claim_dates",
        when(
            col("ClaimStartDt").isNull() |
            col("ClaimEndDt").isNull() |
            (col("ClaimEndDt") < col("ClaimStartDt")),
            1
        ).otherwise(0)
    )
    .withColumn(
        "dq_negative_reimbursement",
        when(col("InscClaimAmtReimbursed") < 0, 1).otherwise(0)
    )
    .withColumn(
        "dq_negative_deductible",
        when(col("DeductibleAmtPaid") < 0, 1).otherwise(0)
    )
    .withColumn(
        "dq_is_valid",
        when(
            (col("dq_missing_claim_id") == 0) &
            (col("dq_missing_provider") == 0) &
            (col("dq_missing_bene_id") == 0) &
            (col("dq_invalid_claim_dates") == 0) &
            (col("dq_negative_reimbursement") == 0) &
            (col("dq_negative_deductible") == 0),
            1
        ).otherwise(0)
    )
)

# COMMAND ----------

# DBTITLE 1,Split Valid and Quarantine Records
valid_outpatient = (
    silver_outpatient_dq
    .filter(col("dq_is_valid") == 1)
)

quarantine_outpatient = (
    silver_outpatient_dq
    .filter(col("dq_is_valid") == 0)
    .withColumn(
        "quarantine_reason",
        concat_ws(
            ", ",
            when(col("dq_missing_claim_id") == 1, "Missing ClaimID"),
            when(col("dq_missing_provider") == 1, "Missing Provider"),
            when(col("dq_missing_bene_id") == 1, "Missing BeneID"),
            when(col("dq_invalid_claim_dates") == 1, "Invalid Claim Dates"),
            when(col("dq_negative_reimbursement") == 1, "Negative Reimbursement"),
            when(col("dq_negative_deductible") == 1, "Negative Deductible")
        )
    )
)

# COMMAND ----------

valid_outpatient.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.silver.outpatient_claims")

# COMMAND ----------

quarantine_outpatient.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.monitoring.quarantine_outpatient_claims")

# COMMAND ----------

# DBTITLE 1,Add Data Quality Metrics
outpatient_dq_metrics = spark.createDataFrame(
    [
        ("outpatient_claims", "total_records", bronze_outpatient.count()),
        ("outpatient_claims", "valid_records", valid_outpatient.count()),
        ("outpatient_claims", "quarantine_records", quarantine_outpatient.count()),
        ("outpatient_claims", "duplicate_claimid_removed", bronze_outpatient.count() - silver_outpatient.count())
    ],
    ["table_name", "metric_name", "metric_value"]
)

outpatient_dq_metrics = outpatient_dq_metrics.withColumn(
    "metric_timestamp",
    current_timestamp()
)

outpatient_dq_metrics.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("medicare_catalog.monitoring.data_quality_metrics")

# COMMAND ----------

display(spark.table("medicare_catalog.silver.outpatient_claims").limit(10))

display(spark.table("medicare_catalog.monitoring.quarantine_outpatient_claims"))

display(spark.table("medicare_catalog.monitoring.data_quality_metrics"))

print("Total outpatient records:", bronze_outpatient.count())
print("Valid outpatient records:", valid_outpatient.count())
print("Quarantine outpatient records:", quarantine_outpatient.count())

# COMMAND ----------

# Databricks notebook source
from pyspark.sql.functions import *

spark.sql("USE CATALOG medicare_catalog")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS monitoring")

# COMMAND ----------

inpatient = spark.table("medicare_catalog.silver.inpatient_claims")
outpatient = spark.table("medicare_catalog.silver.outpatient_claims")

display(inpatient.limit(5))
display(outpatient.limit(5))

inpatient.printSchema()
outpatient.printSchema()

# COMMAND ----------

# DBTITLE 1,Ensure Claim Type Columns Exist
inpatient = inpatient.withColumn("claim_type", lit("inpatient"))
outpatient = outpatient.withColumn("claim_type", lit("outpatient"))

# COMMAND ----------

# DBTITLE 1,Align Outpatient Columns
outpatient_aligned = (
    outpatient
    .withColumn("AdmissionDt", lit(None).cast("date"))
    .withColumn("DischargeDt", lit(None).cast("date"))
    .withColumn("hospital_stay_days", lit(None).cast("int"))
)

# COMMAND ----------

# DBTITLE 1,Select Common Columns
common_columns = [
    "ClaimID",
    "BeneID",
    "Provider",
    "ClaimStartDt",
    "ClaimEndDt",
    "AdmissionDt",
    "DischargeDt",
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
    "claim_duration_days",
    "hospital_stay_days",
    "claim_type"
]

inpatient_selected = inpatient.select(common_columns)
outpatient_selected = outpatient_aligned.select(common_columns)

# COMMAND ----------

# DBTITLE 1,Combine Inpatient + Outpatient
unified_claims = inpatient_selected.unionByName(outpatient_selected)

# COMMAND ----------

# DBTITLE 1,Add Unified-Level Data Quality Flags
unified_claims_dq = (
    unified_claims
    .withColumn(
        "dq_invalid_claim_type",
        when(~col("claim_type").isin("inpatient", "outpatient"), 1).otherwise(0)
    )
    .withColumn(
        "dq_invalid_claim_duration",
        when(col("claim_duration_days") <= 0, 1).otherwise(0)
    )
    .withColumn(
        "dq_is_valid_unified",
        when(
            (col("dq_invalid_claim_type") == 0) &
            (col("dq_invalid_claim_duration") == 0),
            1
        ).otherwise(0)
    )
)

# COMMAND ----------

# DBTITLE 1,Split Valid and Quarantine Records
valid_unified_claims = (
    unified_claims_dq
    .filter(col("dq_is_valid_unified") == 1)
)

quarantine_unified_claims = (
    unified_claims_dq
    .filter(col("dq_is_valid_unified") == 0)
    .withColumn(
        "quarantine_reason",
        concat_ws(
            ", ",
            when(col("dq_invalid_claim_type") == 1, "Invalid Claim Type"),
            when(col("dq_invalid_claim_duration") == 1, "Invalid Claim Duration")
        )
    )
)

# COMMAND ----------

valid_unified_claims.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.silver.unified_claims")

# COMMAND ----------

quarantine_unified_claims.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.monitoring.quarantine_unified_claims")

# COMMAND ----------

unified_dq_metrics = spark.createDataFrame(
    [
        ("unified_claims", "total_records", unified_claims.count()),
        ("unified_claims", "valid_records", valid_unified_claims.count()),
        ("unified_claims", "quarantine_records", quarantine_unified_claims.count())
    ],
    ["table_name", "metric_name", "metric_value"]
)

unified_dq_metrics = unified_dq_metrics.withColumn(
    "metric_timestamp",
    current_timestamp()
)

unified_dq_metrics.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("medicare_catalog.monitoring.data_quality_metrics")

# COMMAND ----------

display(spark.table("medicare_catalog.silver.unified_claims").limit(10))

display(spark.table("medicare_catalog.monitoring.quarantine_unified_claims"))

display(spark.table("medicare_catalog.monitoring.data_quality_metrics"))

print("Inpatient records:", inpatient.count())
print("Outpatient records:", outpatient.count())
print("Unified total:", unified_claims.count())
print("Valid unified records:", valid_unified_claims.count())
print("Quarantine unified records:", quarantine_unified_claims.count())

# COMMAND ----------

# Databricks notebook source
from pyspark.sql.functions import *

spark.sql("USE CATALOG medicare_catalog")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS monitoring")

# COMMAND ----------

bronze_provider = spark.table("medicare_catalog.bronze.provider")

display(bronze_provider.limit(5))
bronze_provider.printSchema()

# COMMAND ----------

silver_provider_labels = (
    bronze_provider
    .dropDuplicates(["Provider"])
    .withColumn(
        "fraud_label",
        when(col("PotentialFraud") == "Yes", 1)
        .when(col("PotentialFraud") == "No", 0)
        .otherwise(None)
    )
    .withColumn("silver_processed_timestamp", current_timestamp())
)

# COMMAND ----------

# DBTITLE 1,Create Data Quality Flags
silver_provider_labels_dq = (
    silver_provider_labels
    .withColumn(
        "dq_missing_provider",
        when(col("Provider").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_invalid_fraud_label",
        when(col("fraud_label").isNull(), 1).otherwise(0)
    )
    .withColumn(
        "dq_is_valid",
        when(
            (col("dq_missing_provider") == 0) &
            (col("dq_invalid_fraud_label") == 0),
            1
        ).otherwise(0)
    )
)

# COMMAND ----------

valid_provider_labels = (
    silver_provider_labels_dq
    .filter(col("dq_is_valid") == 1)
)

quarantine_provider_labels = (
    silver_provider_labels_dq
    .filter(col("dq_is_valid") == 0)
    .withColumn(
        "quarantine_reason",
        concat_ws(
            ", ",
            when(col("dq_missing_provider") == 1, "Missing Provider"),
            when(col("dq_invalid_fraud_label") == 1, "Invalid Fraud Label")
        )
    )
)

# COMMAND ----------

valid_provider_labels.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.silver.provider_labels")

# COMMAND ----------

quarantine_provider_labels.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.monitoring.quarantine_provider_labels")

# COMMAND ----------

provider_dq_metrics = spark.createDataFrame(
    [
        ("provider_labels", "total_records", bronze_provider.count()),
        ("provider_labels", "valid_records", valid_provider_labels.count()),
        ("provider_labels", "quarantine_records", quarantine_provider_labels.count()),
        ("provider_labels", "duplicate_provider_removed", bronze_provider.count() - silver_provider_labels.count())
    ],
    ["table_name", "metric_name", "metric_value"]
)

provider_dq_metrics = provider_dq_metrics.withColumn(
    "metric_timestamp",
    current_timestamp()
)

provider_dq_metrics.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("medicare_catalog.monitoring.data_quality_metrics")

# COMMAND ----------

display(spark.table("medicare_catalog.silver.provider_labels").limit(10))

display(spark.table("medicare_catalog.monitoring.quarantine_provider_labels"))

display(spark.table("medicare_catalog.monitoring.data_quality_metrics"))

print("Total provider records:", bronze_provider.count())
print("Valid provider records:", valid_provider_labels.count())
print("Quarantine provider records:", quarantine_provider_labels.count())

# COMMAND ----------

from pyspark.sql.functions import count, sum, col, current_timestamp, lit

# 1. Calculate all metrics in a single pass using aggregation
metrics_row = silver_provider_labels_dq.select(
    count("*").alias("total_records"),
    sum(col("dq_is_valid")).alias("valid_records"),
    sum(when(col("dq_is_valid") == 0, 1).otherwise(0)).alias("quarantine_records")
).collect()[0]

# Extract the calculated numbers from the row object
total = metrics_row["total_records"]
valid = metrics_row["valid_records"] or 0
quarantine = metrics_row["quarantine_records"] or 0

# Note: For 'duplicate_provider_removed', you still need a baseline count from the bronze table.
# If bronze isn't cached, you can calculate duplicates by comparing the total here 
# with your expected raw record baseline, or perform a single fast bronze.count() if needed.
bronze_total = bronze_provider.count() 
duplicates_removed = bronze_total - total

# 2. Reconstruct the standardized long-format DataFrame
provider_dq_metrics = spark.createDataFrame(
    [
        ("provider_labels", "total_records", int(total)),
        ("provider_labels", "valid_records", int(valid)),
        ("provider_labels", "quarantine_records", int(quarantine)),
        ("provider_labels", "duplicate_provider_removed", int(duplicates_removed))
    ], 
    ["table_name", "metric_name", "metric_value"]
)

# 3. Add timestamp and append to Delta
provider_dq_metrics = provider_dq_metrics.withColumn("metric_timestamp", current_timestamp())

provider_dq_metrics.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("medicare_catalog.monitoring.data_quality_metrics")

# COMMAND ----------

print("Quarantine provider records:", quarantine_provider_labels.count())

# COMMAND ----------

display(spark.table("medicare_catalog.monitoring.data_quality_metrics"))

# COMMAND ----------

# Databricks notebook source
from pyspark.sql.functions import *

spark.sql("USE CATALOG medicare_catalog")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")

# COMMAND ----------

claims = spark.table("medicare_catalog.silver.unified_claims")
beneficiary = spark.table("medicare_catalog.silver.beneficiary")
provider_labels = spark.table("medicare_catalog.silver.provider_labels")

display(claims.limit(5))
display(beneficiary.limit(5))
display(provider_labels.limit(5))

# COMMAND ----------

# DBTITLE 1,oin Claims with Beneficiary Data
claims_with_bene = (
    claims
    .join(
        beneficiary,
        on="BeneID",
        how="left"
    )
)

# COMMAND ----------

# DBTITLE 1,Create Claim-Level Helper Features
claims_features = (
    claims_with_bene
    .withColumn(
        "is_inpatient_claim",
        when(col("claim_type") == "inpatient", 1).otherwise(0)
    )
    .withColumn(
        "is_outpatient_claim",
        when(col("claim_type") == "outpatient", 1).otherwise(0)
    )
    .withColumn(
        "is_high_reimbursement_claim",
        when(col("InscClaimAmtReimbursed") >= 5000, 1).otherwise(0)
    )
    .withColumn(
        "has_chronic_condition",
        when(
            (col("ChronicCond_Alzheimer") == 1) |
            (col("ChronicCond_Heartfailure") == 1) |
            (col("ChronicCond_KidneyDisease") == 1) |
            (col("ChronicCond_Cancer") == 1) |
            (col("ChronicCond_ObstrPulmonary") == 1) |
            (col("ChronicCond_Depression") == 1) |
            (col("ChronicCond_Diabetes") == 1) |
            (col("ChronicCond_IschemicHeart") == 1) |
            (col("ChronicCond_Osteoporasis") == 1) |
            (col("ChronicCond_rheumatoidarthritis") == 1) |
            (col("ChronicCond_stroke") == 1),
            1
        ).otherwise(0)
    )
)

# COMMAND ----------

# DBTITLE 1,Aggregate Provider-Level Features
provider_features = (
    claims_features
    .groupBy("Provider")
    .agg(
        countDistinct("ClaimID").alias("total_claims"),
        countDistinct("BeneID").alias("unique_beneficiaries"),

        sum("is_inpatient_claim").alias("inpatient_claims"),
        sum("is_outpatient_claim").alias("outpatient_claims"),

        avg("InscClaimAmtReimbursed").alias("avg_reimbursement_amount"),
        sum("InscClaimAmtReimbursed").alias("total_reimbursement_amount"),
        max("InscClaimAmtReimbursed").alias("max_reimbursement_amount"),

        avg("DeductibleAmtPaid").alias("avg_deductible_amount"),
        sum("DeductibleAmtPaid").alias("total_deductible_amount"),

        avg("claim_duration_days").alias("avg_claim_duration_days"),
        avg("hospital_stay_days").alias("avg_hospital_stay_days"),

        sum("is_high_reimbursement_claim").alias("high_reimbursement_claims"),
        sum("has_chronic_condition").alias("chronic_condition_claims")
    )
)

# COMMAND ----------

# DBTITLE 1,Add Ratio Features
provider_features_enriched = (
    provider_features
    .withColumn(
        "inpatient_claim_ratio",
        col("inpatient_claims") / col("total_claims")
    )
    .withColumn(
        "outpatient_claim_ratio",
        col("outpatient_claims") / col("total_claims")
    )
    .withColumn(
        "high_reimbursement_claim_ratio",
        col("high_reimbursement_claims") / col("total_claims")
    )
    .withColumn(
        "chronic_condition_claim_ratio",
        col("chronic_condition_claims") / col("total_claims")
    )
    .withColumn(
        "claims_per_beneficiary",
        col("total_claims") / col("unique_beneficiaries")
    )
)

# COMMAND ----------

# DBTITLE 1,join Fraud Labels
gold_provider_features = (
    provider_features_enriched
    .join(
        provider_labels.select("Provider", "fraud_label"),
        on="Provider",
        how="left"
    )
    .withColumn("gold_processed_timestamp", current_timestamp())
)

# COMMAND ----------

# DBTITLE 1,Handle Missing Values
gold_provider_features_clean = (
    gold_provider_features
    .fillna(
        {
            "avg_hospital_stay_days": 0,
            "fraud_label": 0
        }
    )
)

# COMMAND ----------

gold_provider_features_clean.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.gold.provider_fraud_features")

# COMMAND ----------

# DBTITLE 1,Validate Gold Dataset
display(
    spark.table(
        "medicare_catalog.gold.provider_fraud_features"
    ).limit(10)
)

print(
    "Gold provider feature rows:",
    spark.table(
        "medicare_catalog.gold.provider_fraud_features"
    ).count()
)

spark.table(
    "medicare_catalog.gold.provider_fraud_features"
).groupBy("fraud_label").count().show()

# COMMAND ----------

# Databricks notebook source
from pyspark.sql.functions import *

provider_features = spark.table("medicare_catalog.gold.provider_fraud_features")

# COMMAND ----------

# DBTITLE 1,Create Analyst-Friendly Table
provider_fraud_analytics = (
    provider_features
    .withColumn(
        "fraud_status",
        when(col("fraud_label") == 1, "Fraud")
        .when(col("fraud_label") == 0, "Non-Fraud")
        .otherwise("Unknown")
    )
    .withColumn(
        "risk_segment",
        when(col("high_reimbursement_claim_ratio") >= 0.30, "High Risk")
        .when(col("high_reimbursement_claim_ratio") >= 0.10, "Medium Risk")
        .otherwise("Low Risk")
    )
    .select(
        "Provider",
        "fraud_status",
        "risk_segment",
        "total_claims",
        "unique_beneficiaries",
        "claims_per_beneficiary",
        "inpatient_claims",
        "outpatient_claims",
        "inpatient_claim_ratio",
        "outpatient_claim_ratio",
        "avg_reimbursement_amount",
        "total_reimbursement_amount",
        "max_reimbursement_amount",
        "avg_deductible_amount",
        "total_deductible_amount",
        "avg_claim_duration_days",
        "avg_hospital_stay_days",
        "high_reimbursement_claims",
        "high_reimbursement_claim_ratio",
        "chronic_condition_claims",
        "chronic_condition_claim_ratio",
        "gold_processed_timestamp"
    )
)

# COMMAND ----------

provider_fraud_analytics.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.gold.provider_fraud_analytics")

# COMMAND ----------

display(spark.table("medicare_catalog.gold.provider_fraud_analytics").limit(10))

spark.table("medicare_catalog.gold.provider_fraud_analytics") \
    .groupBy("fraud_status", "risk_segment") \
    .count() \
    .show()

# COMMAND ----------
