#!/bin/bash

# Script Name: concat.sh
# Description: Concatenates all .py files in the src directory into a single .txt file.
#              Each file's content is prefixed with its filename for easy identification.
# Usage: ./concat.sh

# Output file name
OUTPUT_FILE="combined_py_files.txt"

# Check if there are any .py files in the src directory
shopt -s nullglob
PY_FILES=(./src/*.py)
shopt -u nullglob

if [ ${#PY_FILES[@]} -eq 0 ]; then
    echo "No .py files found in the src directory."
    exit 1
fi

# Create or empty the output file
> "$OUTPUT_FILE"

# Iterate over each .py file and append its content to the output file
for FILE in "${PY_FILES[@]}"; do
    echo "===== $FILE =====" >> "$OUTPUT_FILE"      # Prefix with filename
    cat "$FILE" >> "$OUTPUT_FILE"                   # Append file content
    echo -e "\n\n" >> "$OUTPUT_FILE"                # Add spacing between files
done

echo "All .py files have been concatenated into '$OUTPUT_FILE'."