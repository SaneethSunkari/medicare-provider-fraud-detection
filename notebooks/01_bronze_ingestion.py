# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Bronze Ingestion and Catalog Setup
# MAGIC
# MAGIC Initializes the Databricks Unity Catalog schemas used by the Medicare provider fraud detection Lakehouse and validates access to raw Bronze source tables.

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# COMMAND ----------

spark.sql("USE CATALOG medicare_catalog")

spark.sql("CREATE SCHEMA IF NOT EXISTS bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS gold")
spark.sql("CREATE SCHEMA IF NOT EXISTS monitoring")

# COMMAND ----------

spark.sql("USE CATALOG medicare_catalog")
spark.sql("USE SCHEMA bronze")

display(spark.sql("SHOW TABLES"))



# COMMAND ----------

display(spark.table("medicare_catalog.bronze.beneficiarydata").limit(5))
display(spark.table("medicare_catalog.bronze.inpatientdata").limit(5))
display(spark.table("medicare_catalog.bronze.outpatientdata").limit(5))
display(spark.table("medicare_catalog.bronze.provider").limit(5))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM medicare_catalog.bronze.beneficiarydata;
# MAGIC SELECT COUNT(*) FROM medicare_catalog.bronze.inpatientdata;
# MAGIC SELECT COUNT(*) FROM medicare_catalog.bronze.outpatientdata;
# MAGIC SELECT COUNT(*) FROM medicare_catalog.bronze.provider;

# COMMAND ----------

# MAGIC %md
# MAGIC
