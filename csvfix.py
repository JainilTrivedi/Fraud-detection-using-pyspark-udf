# preprocess_csv.py
import csv
import re

input_csv_path = "data/idimage.csv"
output_csv_path = "data/idimage_fixed.csv"
# Regex to capture the filename and the start of the base64 string
# It assumes the base64 part starts after the first comma and potential quote
start_pattern = re.compile(r'^([^,]+),"?(.*)')

print(f"Starting preprocessing of {input_csv_path}...")

try:
    with open(input_csv_path, "r", encoding="utf-8", errors="ignore") as infile, open(
        output_csv_path, "w", newline="", encoding="utf-8"
    ) as outfile:

        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
        # Write header
        writer.writerow(["name", "imageData"])

        current_name = None
        current_image_data_parts = []
        line_number = 0

        for line in infile:
            line_number += 1
            line = (
                line.strip()
            )  # Remove leading/trailing whitespace including newline chars

            if not line:  # Skip empty lines
                continue

            # Check if it's the start of a new record (contains filename and comma)
            match = start_pattern.match(line)
            # Heuristic: Assume a new record if it starts with non-comma chars followed by a comma
            is_new_record_line = (
                "," in line and not line.startswith('"') and not line.startswith("/")
            )

            # --- Logic Refined ---
            if line_number == 1:  # Skip header line in input
                continue

            if match and is_new_record_line:  # Likely start of a new record
                # If we have data from a previous record, write it out
                if current_name and current_image_data_parts:
                    full_image_data = "".join(current_image_data_parts).strip(
                        '"'
                    )  # Join parts, remove boundary quotes if any
                    if full_image_data:  # Write only if we have image data
                        writer.writerow([current_name, full_image_data])
                    else:
                        print(
                            f"Warning: No image data found for {current_name} before line {line_number}"
                        )

                # Start the new record
                current_name = match.group(1).strip()
                # Start collecting image data parts, including the first part from this line
                current_image_data_parts = [match.group(2).strip()]
                # print(f"Started new record: {current_name} at line {line_number}") # Debug

            elif current_name:  # This line is a continuation of the previous image data
                current_image_data_parts.append(line.strip())
            else:
                # This case handles lines before the first valid record start is found (e.g. header or garbage lines)
                if line_number > 1:  # Avoid printing for the header
                    print(
                        f"Warning: Skipping line {line_number} as it doesn't seem to belong to a record: {line[:100]}..."
                    )

        # Write the very last record after the loop finishes
        if current_name and current_image_data_parts:
            full_image_data = "".join(current_image_data_parts).strip('"')
            if full_image_data:
                writer.writerow([current_name, full_image_data])
            else:
                print(
                    f"Warning: No image data found for the last record {current_name}"
                )

    print(f"Preprocessing finished. Corrected data saved to {output_csv_path}")

except FileNotFoundError:
    print(f"Error: Input file not found at {input_csv_path}")
except Exception as e:
    print(f"An error occurred during preprocessing: {e}")
    import traceback

    traceback.print_exc()
