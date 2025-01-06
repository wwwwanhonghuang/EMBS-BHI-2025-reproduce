#!/bin/bash

# Define the path and file names
DATASET_DIR="../data/dataset/epileptic_eeg_dataset"
ZIP_FILE="epileptic_eeg_dataset.zip"
DOWNLOAD_URL="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5pc2j46cbc-1.zip"



# Check if the zip file already exists
if [ ! -f "$ZIP_FILE" ]; then
    # Download the dataset if the zip file does not exist
    wget "$DOWNLOAD_URL" -O "$ZIP_FILE"
else
    echo "Zip file '$ZIP_FILE' already exists. Skipping download."
fi

# Unzip the file if it exists
# Decompress only the 'Raw_EDF_Files' folder from the zip archive if it exists
if [ -f "$ZIP_FILE" ]; then
    unzip "$ZIP_FILE" "Raw_EDF_Files/*" -d .  # Extract only the folder
fi


# Create the destination directory if it doesn't exist
mkdir -p "$DATASET_DIR"

# Move the extracted Raw_EDF_Files to the target directory
mv Raw_EDF_Files "$DATASET_DIR"

echo "Dataset preparation completed."
