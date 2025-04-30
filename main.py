from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import lit
from functools import reduce

import gc
import sys

from models.cnn import CNN

import base64
import io
from PIL import Image
import numpy as np
import pandas as pd
import torch

# for duplicating the tables
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MODEL_PATH_PYTORCH = "models/fraud_cnn.pt"
MODEL_PATH_TENSORFLOW = "models/efficientNet.pt"

# Create Spark session
spark = SparkSession.builder.appName("FraudModelInference_Tables").getOrCreate()

# Loading CSV files into saprk DataFrames
#spark action
idimage_df = spark.read.option("header", True).csv("data/idimage_fixed.csv")
idlabel_df = spark.read.option("header", True).csv("data/idlabel.csv")
idmeta_df = spark.read.option("header", True).csv("data/idmeta.csv")  

#change isfraud column datatype to Bool
idlabel_df = idlabel_df.withColumn("isfraud", col("isfraud").cast(BooleanType()))   #Transformation

# Register all tables
idimage_df.createOrReplaceTempView("idimage")
idlabel_df.createOrReplaceTempView("idlabel")
idmeta_df.createOrReplaceTempView("idmeta")

def duplicate_df(df, n):
    # Duplicate the DataFrame n times, each with a unique batch id
    dfs = [df.withColumn("dup_batch", lit(i)) for i in range(n)]
    return reduce(lambda a, b: a.unionByName(b), dfs)

scales = [("original", 1), ("5x", 5), ("10x", 10),("20x",20),("50x",50) ,("100x",100)]   #("100x", 100), ("1000x", 1000)

idimage_tables = {}
idlabel_tables = {}
idmeta_tables = {}

for name, factor in scales:
    if factor == 1:
        idimage_tables[name] = idimage_df
        idlabel_tables[name] = idlabel_df
        idmeta_tables[name] = idmeta_df
    else:
        idimage_tables[name] = duplicate_df(idimage_df, factor)
        idlabel_tables[name] = duplicate_df(idlabel_df, factor)
        idmeta_tables[name] = duplicate_df(idmeta_df, factor)
    idimage_tables[name].createOrReplaceTempView(f"idimage_{name}")
    idlabel_tables[name].createOrReplaceTempView(f"idlabel_{name}")
    idmeta_tables[name].createOrReplaceTempView(f"idmeta_{name}")


try:
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH_PYTORCH, map_location=torch.device('cpu')))
    model.eval()
    broadcast_model = spark.sparkContext.broadcast(model)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 


def preprocess_image(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((128, 128))  
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dim
        image_array = image_array.transpose((0, 3, 1, 2)) 
        image_tensor = torch.from_numpy(image_array).float()
        
        return image_tensor

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return None

#=== Defining UDF ===

@pandas_udf(BooleanType())
def cnn_fraud_detector(image_col: pd.Series) -> pd.Series:
    mdl = broadcast_model.value
    results = []
    for base64_str in image_col:
        if mdl is None:
            results.append(False)
            continue

        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image = image.resize((128, 128))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array.transpose((0, 3, 1, 2))
            image_tensor = torch.from_numpy(image_array).float()

            with torch.no_grad():
                prediction = mdl(image_tensor)
            results.append(bool(prediction[0][0] > 0.5))
        except Exception as e:
            print(f"Prediction failed: {e}")
            results.append(False)

    return pd.Series(results)

queries = {
    "fraud_predicted": """
        SELECT 
            COUNT(*) AS total_ids,
            SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) AS fraud_predicted,
            (SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS fraud_rate_percentage
        FROM {table}
    """,
    "fraud_ground_truth": """
        SELECT 
            COUNT(*) AS total_ids,
            SUM(CASE WHEN isfraud THEN 1 ELSE 0 END) AS total_fraud,
            SUM(CASE WHEN NOT isfraud THEN 1 ELSE 0 END) AS total_nonfraud,
            (SUM(CASE WHEN isfraud THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS fraud_percentage
        FROM {table}
    """,
    "fraud_pattern": """
        SELECT 
            fraudpattern, 
            COUNT(*) AS pattern_count
        FROM {table}
        WHERE isfraud = TRUE
        GROUP BY fraudpattern
        ORDER BY pattern_count DESC
        LIMIT 10
    """
}


timings = {q: [] for q in queries}

for name, _ in scales:
    # Ensure predicted_fraud column exists (recompute if needed)
    idimage_pred = idimage_tables[name].withColumn(
        "predicted_fraud",
        cnn_fraud_detector(col("imageData"))
    )
    idimage_pred.createOrReplaceTempView(f"idimage_pred_{name}")
    
    # Time each query
    for qname, qsql in queries.items():
        if qname == "fraud_predicted":
            table = f"idimage_pred_{name}"
        elif qname == "fraud_ground_truth" or qname == "fraud_pattern":
            table = f"idlabel_{name}"
        elif qname == "fraud_rate_gender":
            qsql = qsql.format(meta=f"idmeta_{name}", label=f"idlabel_{name}")
        else:
            table = f"idlabel_{name}"
        if "{table}" in qsql:
            qsql = qsql.format(table=table)
        start = time.time()
        spark.sql(qsql).collect()
        elapsed = time.time() - start
        timings[qname].append(elapsed)

    print("==================",name," done ==================")
    spark.catalog.clearCache()
    gc.collect()
    if 'sc' in globals():
        sc._jvm.java.lang.System.gc()



# def cnn_fraud_detector(base64_str):
#     if model is None:
#         print("Model not loaded, cannot perform prediction.")
#         return False 

#     image_tensor = preprocess_image(base64_str)
#     if image_tensor is None:
#         return False
#     try:
#         with torch.no_grad():
#             prediction = model(image_tensor)
        
#         return bool(prediction[0][0] > 0.5)  
#     except Exception as e:
#         print(f"Prediction failed: {e}")
#         return False




# fraud_udf = udf(cnn_fraud_detector)

# spark.udf.register("cnn_fraud_udf", cnn_fraud_detector)

# idimage_schema = spark.sql("""
#     SELECT * FROM idimage LIMIT 0;
# """)

# idlabel_schema = spark.sql("""
#     SELECT * FROM idlabel LIMIT 0;
# """)

# idmeta_schema = spark.sql("""
#     SELECT * FROM idmeta LIMIT 0;
# """)

# print("Schema for `idimage` table:")
# idimage_schema.printSchema()
# print("\n" + "="*5 + "\n")

# print("Schema for `idlabel` table:")
# idlabel_schema.printSchema()
# print("\n" + "="*5 + "\n")

# print("Schema for `idmeta` table:")
# idmeta_schema.printSchema()
# print("\n" + "="*5 + "\n")


# ADDING PREDICTION COLUMN TO THE TABLE FOR FASTER ANSWER RETRiEVAL
# idimage_with_pred = idimage_df.withColumn(
#     "predicted_fraud",
#     cnn_fraud_detector(col("imageData"))
# )

# # REGISTERING THE NEW TEMPORARY VIEW
# idimage_with_pred.createOrReplaceTempView("idimage_pred")


for qname in queries:
    plt.plot([s[0] for s in scales], timings[qname], marker='o', label=qname)

# plt.xlabel('Data Scale')
# plt.ylabel('Query Execution Time (seconds)')
# plt.title('Spark SQL Query Time vs Data Scale')
# plt.legend()
# plt.grid(True)
# plt.savefig('time_comparision_spark_sql_100x.png')
# plt.show()

plt.xlabel('Data Scale')
plt.ylabel('Query Execution Time (seconds)')
plt.title('Spark SQL Query Time vs Data Scale')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Set y-axis to log scale
plt.yscale('log')

# Set custom y-ticks and labels
yticks = [0, 10, 20, 30, 40, 50, 100, 200, 300, 400]
# Remove 0, because log(0) is undefined
yticks_nozero = [y for y in yticks if y > 0]
plt.yticks(yticks_nozero, [str(y) for y in yticks_nozero])

# Optionally, minor ticks for more granularity
plt.gca().yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))

plt.savefig('time_comparision_spark_sql_final.png')
plt.show()



# Printing the comparision in tabular format
header = ["Query"] + [s[0] for s in scales]
rows = []
for qname in queries:
    row = [qname] + [f"{t:.2f}" for t in timings[qname]]
    rows.append(row)

print("\n=== Query Execution Times (seconds) ===")
print(tabulate(rows, headers=header, tablefmt="github"))

# SLOW BECAUSE OF UDF INSIDE QUERIES so it is calling ML model every time which is making the query slow.

# spark.sql("""
# SELECT 
#     COUNT(*) AS total_ids,
#     SUM(CASE WHEN cnn_fraud_udf(imageData) THEN 1 ELSE 0 END) AS fraud_predicted,
#     (SUM(CASE WHEN cnn_fraud_udf(imageData) THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS fraud_rate_percentage
# FROM idimage LIMIT 10
# """).show()


# print("""================ Total IDs and Predicted Fraud Percentage ================""")
# spark.sql("""
# SELECT 
#     COUNT(*) AS total_ids,
#     SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) AS fraud_predicted,
#     (SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS fraud_rate_percentage
# FROM idimage_pred
# """).show()


# print("""================ Total Fraudulent vs Non-Fraudulent IDs (Ground Truth) ================""")
# spark.sql("""
#     SELECT 
#         COUNT(*) AS total_ids,
#         SUM(CASE WHEN isfraud THEN 1 ELSE 0 END) AS total_fraud,
#         SUM(CASE WHEN NOT isfraud THEN 1 ELSE 0 END) AS total_nonfraud,
#         (SUM(CASE WHEN isfraud THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS fraud_percentage
#     FROM idlabel
# """).show()


# print("""================ Most Common Fraud Patterns ================""")
# spark.sql("""
#     SELECT 
#         fraudpattern, 
#         COUNT(*) AS pattern_count
#     FROM idlabel
#     WHERE isfraud = TRUE
#     GROUP BY fraudpattern
#     ORDER BY pattern_count DESC
#     LIMIT 10""").show()


# print("""================  Fraud Rate Gender Wise ================""")
# spark.sql("""
#     SELECT 
#         m.gender,
#         COUNT(*) AS total,
#         SUM(CASE WHEN l.isfraud THEN 1 ELSE 0 END) AS fraud_count,
#         (SUM(CASE WHEN l.isfraud THEN 1 ELSE 0 END) * 100.0) / COUNT(*) 
#     AS fraud_rate
#         FROM idmeta m
#         JOIN idlabel l ON m.id = l.id
#         GROUP BY m.gender
#         ORDER BY fraud_rate DESC
# """).show()



# print("""================  Ground Truth v/s Prediction ================""")
# spark.sql("""
#     SELECT 
#         m.id, m.name, l.isfraud, p.predicted_fraud
#     FROM idmeta m
#     JOIN idlabel l ON m.id = l.id
#     JOIN idimage_pred p ON m.name = p.name
#     WHERE l.isfraud <> p.predicted_fraud
# """).show()

