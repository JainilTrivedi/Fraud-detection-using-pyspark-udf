from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

from models.cnn import CNN

import base64
import io
from PIL import Image
import numpy as np
import pandas as pd
import torch

MODEL_PATH_PYTORCH = "models/cnn.pt"
MODEL_PATH_TENSORFLOW = "models/cnn.pt"

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

def cnn_fraud_detector(base64_str):
    if model is None:
        print("Model not loaded, cannot perform prediction.")
        return False 

    image_tensor = preprocess_image(base64_str)
    if image_tensor is None:
        return False
    try:
        with torch.no_grad():
            prediction = model(image_tensor)
        
        return bool(prediction[0][0] > 0.5)  
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False




fraud_udf = udf(cnn_fraud_detector, BooleanType())

spark.udf.register("cnn_fraud_udf", cnn_fraud_detector, BooleanType())

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


# spark.sql("""
#     SELECT name, cnn_fraud_udf(imageData) as is_fraud
#     FROM idimage
#     LIMIT 5
# """).show()

