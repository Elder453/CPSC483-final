#!/bin/bash

# Set the output CSV file
output_file="summary.csv"

# Write the CSV header
echo "Dataset,gnn_type,loss_type,mse_acc_1,mse_pos_1,mse_pos_20,mse_acc_20,mse_pos_full,mse_acc_full,emd" > "$output_file"

# Use find to locate all *full.out files within the models directory
find models -type f -name "*full.out" | while read -r file; do
    # Extract the path components
    # Expected path: models/{Dataset}/{gnn_type}/{loss_type}/<filename>
    IFS='/' read -r _ Dataset gnn_type loss_type filename <<< "$file"
    
    # Initialize variables to store metric values
    mse_acc_1=""
    mse_pos_1=""
    mse_pos_20=""
    mse_acc_20=""
    mse_pos_full=""
    mse_acc_full=""
    emd=""
    
    # Extract metrics using grep and awk
    mse_acc_1=$(grep "mse_acc_1:" "$file" | awk '{print $2}')
    mse_pos_1=$(grep "mse_pos_1:" "$file" | awk '{print $2}')
    mse_pos_20=$(grep "mse_pos_20:" "$file" | awk '{print $2}')
    mse_acc_20=$(grep "mse_acc_20:" "$file" | awk '{print $2}')
    mse_pos_full=$(grep "mse_pos_full:" "$file" | awk '{print $2}')
    mse_acc_full=$(grep "mse_acc_full:" "$file" | awk '{print $2}')
    emd=$(grep "^emd:" "$file" | awk '{print $2}')
    
    # Handle cases where a metric might be missing
    mse_acc_1=${mse_acc_1:-N/A}
    mse_pos_1=${mse_pos_1:-N/A}
    mse_pos_20=${mse_pos_20:-N/A}
    mse_acc_20=${mse_acc_20:-N/A}
    mse_pos_full=${mse_pos_full:-N/A}
    mse_acc_full=${mse_acc_full:-N/A}
    emd=${emd:-N/A}
    
    # Append the extracted data to the CSV
    echo "$Dataset,$gnn_type,$loss_type,$mse_acc_1,$mse_pos_1,$mse_pos_20,$mse_acc_20,$mse_pos_full,$mse_acc_full,$emd" >> "$output_file"
done

echo "Parsing complete. Summary saved to $output_file."