from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import pandas_udf

from models.cnn import CNN

import base64
import io
from PIL import Image
import numpy as np
import pandas as pd
import torch

MODEL_PATH_PYTORCH = "models/fraud_cnn.pt"
# MODEL_PATH_TENSORFLOW = "models/cnn.pt"

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




fraud_udf = udf(cnn_fraud_detector)

spark.udf.register("cnn_fraud_udf", cnn_fraud_detector)

idimage_schema = spark.sql("""
    SELECT * FROM idimage LIMIT 0;
""")

idlabel_schema = spark.sql("""
    SELECT * FROM idlabel LIMIT 0;
""")

idmeta_schema = spark.sql("""
    SELECT * FROM idmeta LIMIT 0;
""")

print("Schema for `idimage` table:")
idimage_schema.printSchema()
print("\n" + "="*5 + "\n")

print("Schema for `idlabel` table:")
idlabel_schema.printSchema()
print("\n" + "="*5 + "\n")

print("Schema for `idmeta` table:")
idmeta_schema.printSchema()
print("\n" + "="*5 + "\n")


# ADDING PREDICTION COLUMN TO THE TABLE FOR FASTER ANSWER RETRiEVAL
idimage_with_pred = idimage_df.withColumn(
    "predicted_fraud",
    cnn_fraud_detector(col("imageData"))
)

# REGISTERING THE NEW TEMPORARY VIEW
idimage_with_pred.createOrReplaceTempView("idimage_pred")


# SLOW BECAUSE OF UDF INSIDE QUERIES so it is calling ML model every time which is making the query slow.

# spark.sql("""
# SELECT 
#     COUNT(*) AS total_ids,
#     SUM(CASE WHEN cnn_fraud_udf(imageData) THEN 1 ELSE 0 END) AS fraud_predicted,
#     (SUM(CASE WHEN cnn_fraud_udf(imageData) THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS fraud_rate_percentage
# FROM idimage LIMIT 10
# """).show()


print("""================ Total IDs and Predicted Fraud Percentage ================""")
spark.sql("""
SELECT 
    COUNT(*) AS total_ids,
    SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) AS fraud_predicted,
    (SUM(CASE WHEN predicted_fraud THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS fraud_rate_percentage
FROM idimage_pred
""").show()


print("""================ Total Fraudulent vs Non-Fraudulent IDs (Ground Truth) ================""")
spark.sql("""
    SELECT 
        COUNT(*) AS total_ids,
        SUM(CASE WHEN isfraud THEN 1 ELSE 0 END) AS total_fraud,
        SUM(CASE WHEN NOT isfraud THEN 1 ELSE 0 END) AS total_nonfraud,
        (SUM(CASE WHEN isfraud THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS fraud_percentage
    FROM idlabel
""").show()


print("""================ Most Common Fraud Patterns ================""")
spark.sql("""
    SELECT 
        fraudpattern, 
        COUNT(*) AS pattern_count
    FROM idlabel
    WHERE isfraud = TRUE
    GROUP BY fraudpattern
    ORDER BY pattern_count DESC
    LIMIT 10""").show()


print("""================  Fraud Rate Gender Wise ================""")
spark.sql("""
    SELECT 
        m.gender,
        COUNT(*) AS total,
        SUM(CASE WHEN l.isfraud THEN 1 ELSE 0 END) AS fraud_count,
        (SUM(CASE WHEN l.isfraud THEN 1 ELSE 0 END) * 100.0) / COUNT(*) 
    AS fraud_rate
        FROM idmeta m
        JOIN idlabel l ON m.id = l.id
        GROUP BY m.gender
        ORDER BY fraud_rate DESC
""").show()







print("""================  Ground Truth v/s Prediction ================""")
spark.sql("""
    SELECT 
        m.id, m.name, l.isfraud, p.predicted_fraud
    FROM idmeta m
    JOIN idlabel l ON m.id = l.id
    JOIN idimage_pred p ON m.name = p.name
    WHERE l.isfraud <> p.predicted_fraud

""")