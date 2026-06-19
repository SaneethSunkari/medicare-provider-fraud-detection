# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Model Training Experiments
# MAGIC
# MAGIC Trains Random Forest provider fraud models, tracks experiments with MLflow, registers the best model in Databricks Model Registry, and assigns the `champion` alias.

# COMMAND ----------

from pyspark.sql.functions import *

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# COMMAND ----------

gold_df = spark.table("medicare_catalog.gold.provider_fraud_features")

display(gold_df.limit(5))
gold_df.printSchema()

# COMMAND ----------

# DBTITLE 1,Convert Spark DataFrame to Pandas
model_df = (
    gold_df
    .select(
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
        "claims_per_beneficiary",
        "fraud_label"
    )
    .fillna(0)
    .toPandas()
)

# COMMAND ----------

# DBTITLE 1,Split Features and Label
X = model_df.drop("fraud_label", axis=1)
y = model_df["fraud_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# COMMAND ----------

# DBTITLE 1,Set MLflow Experiment
mlflow.set_experiment("/Shared/medicare_provider_fraud_experiment")

# COMMAND ----------

# DBTITLE 1,Train Random Forest Model 1
from mlflow.models.signature import infer_signature

with mlflow.start_run(run_name="random_forest_provider_fraud"):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    signature = infer_signature(
        X_train,
        model.predict(X_train)
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5)
    )

    print("Model 1 Results")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", auc)

# COMMAND ----------

# DBTITLE 1,Train Random Forest Model 2
with mlflow.start_run(run_name="random_forest_deeper_model"):

    model_2 = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

    model_2.fit(X_train, y_train)

    y_pred_2 = model_2.predict(X_test)
    y_prob_2 = model_2.predict_proba(X_test)[:, 1]

    accuracy_2 = accuracy_score(y_test, y_pred_2)
    precision_2 = precision_score(y_test, y_pred_2, zero_division=0)
    recall_2 = recall_score(y_test, y_pred_2, zero_division=0)
    f1_2 = f1_score(y_test, y_pred_2, zero_division=0)
    auc_2 = roc_auc_score(y_test, y_prob_2)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 12)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy", accuracy_2)
    mlflow.log_metric("precision", precision_2)
    mlflow.log_metric("recall", recall_2)
    mlflow.log_metric("f1_score", f1_2)
    mlflow.log_metric("roc_auc", auc_2)

    signature_2 = infer_signature(X_train, model_2.predict(X_train))

    mlflow.sklearn.log_model(
    sk_model=model_2,
    artifact_path="model",
    signature=signature_2,
    input_example=X_train.head(5)
    )

    print("Model 2 Results")
    print("Accuracy:", accuracy_2)
    print("Precision:", precision_2)
    print("Recall:", recall_2)
    print("F1 Score:", f1_2)
    print("ROC AUC:", auc_2)

# COMMAND ----------

# DBTITLE 1,Check Class Balance
print("Training label distribution:")
print(y_train.value_counts(normalize=True))

print("Testing label distribution:")
print(y_test.value_counts(normalize=True))

# COMMAND ----------

# DBTITLE 1,Feature Importance
feature_importance_df = spark.createDataFrame(
    [(name, float(score)) for name, score in zip(X.columns, model_2.feature_importances_)],
    ["feature_name", "importance_score"]
).orderBy(col("importance_score").desc())

display(feature_importance_df)

# COMMAND ----------

feature_importance_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("medicare_catalog.gold.model_feature_importance")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM medicare_catalog.gold.model_feature_importance
# MAGIC ORDER BY importance_score DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Best Model

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# DBTITLE 1,Find Best Run by ROC AUC
experiment = mlflow.get_experiment_by_name(
    "/Shared/medicare_provider_fraud_experiment"
)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
    max_results=1
)

display(runs)

# COMMAND ----------

# DBTITLE 1,Get Best Run ID
best_run_id = runs.iloc[0]["run_id"]
best_auc = runs.iloc[0]["metrics.roc_auc"]

print("Best Run ID:", best_run_id)
print("Best ROC AUC:", best_auc)

# COMMAND ----------

# DBTITLE 1,Register Best Model
model_name = "medicare_catalog.gold.medicare_provider_fraud_model"

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

# COMMAND ----------

# DBTITLE 1,Confirm Registered Model
print("Registered model name:", registered_model.name)
print("Model version:", registered_model.version)

# COMMAND ----------

# DBTITLE 1,Add Description
client = MlflowClient()

client.update_model_version(
    name=model_name,
    version=registered_model.version,
    description="Random Forest model trained on provider-level Medicare fraud features from medicare_catalog.gold.provider_fraud_features."
)

# COMMAND ----------

# DBTITLE 1,Add Tags
client.set_model_version_tag(
    name=model_name,
    version=registered_model.version,
    key="project",
    value="medicare_provider_fraud_detection"
)

client.set_model_version_tag(
    name=model_name,
    version=registered_model.version,
    key="source_table",
    value="medicare_catalog.gold.provider_fraud_features"
)

client.set_model_version_tag(
    name=model_name,
    version=registered_model.version,
    key="metric_roc_auc",
    value=str(best_auc)
)

# COMMAND ----------

# DBTITLE 1,Set Champion Alias
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=registered_model.version
)

# COMMAND ----------

# DBTITLE 1,Validate
print("Model registered successfully")
print("Model name:", model_name)
print("Version:", registered_model.version)
print("Alias: champion")
print("ROC AUC:", best_auc)

# COMMAND ----------
