from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col, trim, lit, substring_index 
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, BooleanType, FloatType 
from pyspark import SparkFiles
import os
import base64
import io
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import traceback


def debug_print(level, filename, functionname, message):
    print(f"[{level}] {filename}.{functionname} :: {message}\n\n")


# -------------------------------------------------------------------
# 1. Spark Session Setup
# -------------------------------------------------------------------
debug_print("INFO", "main.py", "<main>", "Starting Spark session setup.")
spark = (
    SparkSession.builder.appName("FraudModelInference_Final") 
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.local.dir", "/tmp/spark-temp")
    # .config("spark.driver.memory", "4g") # Increase if needed
    .getOrCreate()
)
debug_print("INFO", "main.py", "<main>", "Spark session created.")

os.makedirs("/tmp/spark-temp", exist_ok=True)
debug_print("INFO", "main.py", "<main>", "Temporary directory created.")

# -------------------------------------------------------------------
#  Distribute Model File
# -------------------------------------------------------------------
model_local_path = "models/cnn.pt"
model_filename = os.path.basename(model_local_path)
try:
    spark.sparkContext.addFile(model_local_path)
    debug_print("INFO", "main.py", "<main>", f"Added '{model_filename}' to SparkContext files.")
except Exception as e:
    debug_print("ERROR", "main.py", "<main>", f"Failed to add model file '{model_local_path}' to SparkContext: {e}")
    spark.stop()
    exit(1)

# -------------------------------------------------------------------
#  2. Load Relevant CSVs (idimage_fixed and idlabel)
# -------------------------------------------------------------------
debug_print("INFO", "main.py", "<main>", "Loading CSV files...")
try:

    idimage_csv_path = "data/idimage_fixed.csv" 
    idimage_df_raw = spark.read.option("header", True).csv(idimage_csv_path)
    debug_print("INFO", "main.py", "<main>", f"Loaded image data from: {idimage_csv_path}.")

    # --- Load label data ---
    idlabel_df = spark.read.option("header", True).csv("data/idlabel.csv")
    # Cast isfraud to Boolean right after loading
    idlabel_df = idlabel_df.withColumn("isfraud", col("isfraud").cast(BooleanType()))
    print(idlabel_df)
    debug_print("INFO", "main.py", "<main>", "idlabel.csv loaded and 'isfraud' cast to Boolean.")


except Exception as e:
    debug_print("ERROR", "main.py", "<main>", f"Error loading CSV data: {e}")
    spark.stop()
    exit(1)

# --- Filter idimage for valid data BEFORE joining ---
idimage_df = idimage_df_raw.filter(col("imageData").isNotNull() & (col("imageData") != "NULL") & (col("imageData") != ""))
idimage_valid_count = idimage_df.count()
debug_print("INFO", "main.py", "<main>", f"Filtered idimage_df. Kept {idimage_valid_count} rows with non-null/empty imageData.")

if idimage_valid_count == 0:
    debug_print("ERROR", "main.py", "<main>", f"No valid image data found in {idimage_csv_path} after filtering. Exiting.")
    spark.stop()
    exit(1)

# -------------------------------------------------------------------
#  3. Join idimage with idlabel (Corrected Join Key)
# -------------------------------------------------------------------
debug_print("INFO", "main.py", "<main>", "Joining idimage with idlabel...")

# Use aliases
image_alias = "i"
label_alias = "l"

# --- CORRECTED JOIN CONDITION ---
# Remove file extension from idimage.name before comparing
image_name_no_ext = substring_index(col(f"{image_alias}.name"), '.', 1)
# Join condition: image name (no ext, trimmed) == label id (trimmed)
join_condition = trim(image_name_no_ext) == trim(col(f"{label_alias}.id"))
# --- END CORRECTION ---

debug_print("INFO", "main.py", "<main>", f"Using join condition: trim(substring_index(i.name, '.', 1)) == trim(l.id)")

joined_df = idimage_df.alias(image_alias).join(
    idlabel_df.alias(label_alias),
    join_condition,
    how="left" # Keep all images, match labels where possible
)

# Select necessary columns: image name, image data, and the isfraud label (which will be NULL if no match)
final_df = joined_df.select(
    col(f"{image_alias}.name").alias("image_name"),
    col(f"{image_alias}.imageData"),
    col(f"{label_alias}.isfraud").alias("label_isfraud") # isfraud from idlabel, will be null if no match
)

final_df_count = final_df.count()
debug_print("INFO", "main.py", "<main>", f"Join resulted in {final_df_count} rows.")

# Check how many rows actually got a label matched
matched_label_count = final_df.filter(col("label_isfraud").isNotNull()).count()
debug_print("INFO", "main.py", "<main>", f"Number of images matched with a label in idlabel: {matched_label_count}")

if matched_label_count == 0:
     debug_print("WARN", "main.py", "<main>", "No images were matched with labels. Double-check filenames vs IDs if this is unexpected.")

# -------------------------------------------------------------------
#  4. Define Model Class (Needed inside UDF) - ARCHITECTURE CORRECTED
# -------------------------------------------------------------------
# Defined inside UDF below

# -------------------------------------------------------------------
#  5. Define Pandas UDF for Inference (Loads Model Internally) - ARCHITECTURE CORRECTED
# -------------------------------------------------------------------
# --- THIS UDF REMAINS THE SAME AS THE PREVIOUS WORKING VERSION ---
def infer_udf_base64(series: pd.Series) -> pd.Series:
    """
    Pandas UDF for performing inference on base64 encoded images.
    Loads the model and transform inside the UDF. Uses 128x128 Resize.
    """
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    import base64
    import io
    import os
    from pyspark import SparkFiles
    import traceback

    # --- Define Model Class INSIDE UDF ---
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            flattened_size = (128 // 8) ** 2 * 64 # = 16384
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(flattened_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
        def forward(self, x):
            return self.net(x)

    # --- Load Model and Transform INSIDE UDF ---
    model_name = "cnn.pt"
    model_path_on_executor = SparkFiles.get(model_name)
    device = torch.device("cpu")
    model = None # Initialize model to None

    try:
        model = CNN().to(device)
        model.load_state_dict(torch.load(model_path_on_executor, map_location=device), strict=True)
        model.eval()
    except Exception as load_error:
        error_msg = f"FATAL: Failed to load model '{model_name}' inside UDF: {load_error}\n{traceback.format_exc()}"
        print(error_msg)
        return pd.Series([f"ERROR: Model load failed"] * len(series))

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    # --- Process Series ---
    results = []
    for img_str in series:
        if img_str is None or len(img_str) < 100:
             results.append("ERROR: Input is None or too short")
             continue
        try:
            # Add padding for base64 decoding
            img_str_padded = img_str + '=' * (-len(img_str) % 4)
            img_bytes = base64.b64decode(img_str_padded)
            # Open image and ensure RGB
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Apply transforms
            tensor = transform(img).unsqueeze(0).to(device)
            # Get prediction
            with torch.no_grad():
                output = model(tensor)
                probability = output.item()
                results.append(str(probability)) # Return probability as string
        except Exception as e:
            # Handle specific known errors for better messages
            if isinstance(e, base64.binascii.Error): err_msg = f"ERROR: Base64 Decode Error - {e}"
            elif isinstance(e, Image.UnidentifiedImageError): err_msg = f"ERROR: PIL Image Error - {e}"
            elif isinstance(e, EOFError): err_msg = f"ERROR: EOFError (potentially truncated image data)"
            else: err_msg = f"ERROR: Processing Error - {e}"
            # Log error details (optional traceback for debugging)
            print(f"[ERROR] UDF Processing Error: {err_msg} for input starting with {str(img_str)[:50]}...")
            # print(f"{traceback.format_exc()}")
            results.append(err_msg) # Return error message in the result column

    return pd.Series(results)

# Register the Pandas UDF
predict_udf = pandas_udf(infer_udf_base64, returnType=StringType())
debug_print("INFO", "main.py", "<main>", "Pandas UDF defined.")

# -------------------------------------------------------------------
# 6. Run Inference on Sampled Data
# -------------------------------------------------------------------
num_samples = 50
print(f"\n Running model inference on up to {num_samples} samples from joined data...")
debug_print("INFO", "main.py", "<main>", f"Selecting up to {num_samples} samples for inference.")

# Take a sample from the final DataFrame
sample_df = final_df.limit(num_samples)
debug_print("INFO", "main.py", "<main>", "Sample DataFrame created.")

sample_count = sample_df.count()
if sample_count == 0:
     debug_print("WARN", "main.py", "<main>", "Sample DataFrame is empty after limit. Cannot run inference.")
else:
    debug_print("INFO", "main.py", "<main>", f"Applying inference UDF to {sample_count} rows.")
    # Apply UDF
    inferred_df = sample_df.withColumn("prediction_prob_str", predict_udf("imageData"))
    debug_print("INFO", "main.py", "<main>", "Inference UDF application started (lazy).")

    # --- Post-process UDF results ---
    debug_print("INFO", "main.py", "<main>", "Post-processing UDF results...")
    # Convert prediction string to float, handling potential errors
    inferred_df = inferred_df.withColumn(
        "prediction_prob",
        F.when(col("prediction_prob_str").startswith("ERROR"), None)
         .otherwise(col("prediction_prob_str").cast(FloatType())) # Use FloatType
    )
    # Create boolean prediction based on threshold
    prediction_threshold = 0.5 # Adjust threshold if needed
    # Logic: If prob < threshold -> Fraud (True), otherwise Not Fraud (False)
    # Ensure this matches how your model labels were defined (e.g., was 1 = fraud or 1 = not fraud?)
    # Assuming 1 = Not Fraud, 0 = Fraud based on previous results.
    inferred_df = inferred_df.withColumn(
        "predicted_isfraud",
         F.when(col("prediction_prob").isNull(), None) # Propagate nulls from errors
          .when(col("prediction_prob") < prediction_threshold, True) # Low prob -> Fraud
          .otherwise(False) # High prob or exactly threshold -> Not Fraud
    )
    debug_print("INFO", "main.py", "<main>", "Prediction probability converted to float and boolean prediction generated.")

    # -------------------------------------------------------------------
    # 7. Show Final Selected Output
    # -------------------------------------------------------------------
    debug_print("INFO", "main.py", "<main>", "Selecting final columns and showing output (triggers computation).")
    try:
        output_df = inferred_df.select(
            col("image_name"),
            col("label_isfraud"),     # Original label (Boolean/Null)
            col("prediction_prob"),   # Calculated probability (Float/Null)
            col("predicted_isfraud") # Final prediction (Boolean/Null)
        )
        # Show results, format probability for readability
        output_df.withColumn("prediction_prob", F.format_number(col("prediction_prob"), 6)).show(num_samples, truncate=False)
        debug_print("INFO", "main.py", "<main>", "Output displayed.")

        # Optional: Add a comparison check
        comparison_df = output_df.filter(col("label_isfraud").isNotNull())
        if comparison_df.count() > 0:
             print("\nComparison where label exists:")
             comparison_df.withColumn("match", col("label_isfraud") == col("predicted_isfraud")) \
                          .select("image_name", "label_isfraud", "predicted_isfraud", "match") \
                          .show(num_samples, truncate=False)


    except Exception as show_error:
        debug_print("ERROR", "main.py", "<main>", f"Error during .select() or .show() execution: {show_error}")
        if "Arrow" in str(show_error):
             print("\n HINT: Arrow-related error detected. Ensure PyArrow is installed ('pip install pyspark[sql]')")

# -------------------------------------------------------------------
# 8. Trying querires
# -------------------------------------------------------------------

debug_print("INFO", "main.py", "<main>", "Running Spark Queries.")


# Boolean UDF (returns True if fraud, False otherwise)
# @pandas_udf(BooleanType())
# def fraud_inference_udf_boolean(series: pd.Series) -> pd.Series:
#     results = []
#     for img_str in series:
#         # ... (same image decoding and model inference as before)
#         probability = output.item()  # Get probability from model
#         results.append(probability < 0.5)  # True = fraud, False = not fraud
#     return pd.Series(results)

# spark.udf.register("fraud_inference_udf", fraud_inference_udf_boolean)

# # Using DataFrame syntax
# result_df = inferred_df.filter(
#     (fraud_inference_udf_boolean("image_data")) &  # UDF returns True for fraud
#     (col("gender") == "M")
# ).select("age")

# # Using SQL syntax
# inferred_df.createOrReplaceTempView("table")

# result_df = spark.sql("""
#     SELECT age 
#     FROM table 
#     WHERE predict_udf(image_data) AND gender = 'M'
# """)

# print("Result DF 1\n")
# # print(result_df)
# print("Result DF 2\n")

# print("Result DF 3\n")

# print("Result DF 4\n")

# print("Result DF 5\n")


# -------------------------------------------------------------------
#  9. Shutdown Spark
# -------------------------------------------------------------------
debug_print("INFO", "main.py", "<main>", "Stopping Spark session.")
spark.stop()
debug_print("INFO", "main.py", "<main>", "Spark session stopped.")
print("\nScript finished.")
