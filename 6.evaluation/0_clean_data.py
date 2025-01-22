import pandas as pd
import re
from tqdm import tqdm
import os
import argparse

def span_length_k_keys(code, k):
    pattern = r'pre_(\d+)_end_(\d+)'
    match = re.match(pattern, code)
    if match:
        num1, num2 = map(int, match.groups())  # Extract and convert to integers
        return num2 == num1 + k - 1  # Check if second number is first + k
    return False

def is_span_length_less_equal_than_k_or_non_span(code, k):
    pattern = r'pre_(\d+)_end_(\d+)'
    match = re.match(pattern, code)
    if match:
        num1, num2 = map(int, match.groups())  # Extract and convert to integers
        return num2 - num1 + 1 <= k 
    return True


def process_file(file, MAX_K=10):
    record_file_path = os.path.join(record_path, file)
    data = pd.read_csv(record_file_path, sep=',')
    
    # Filter out spans with length greater than MAX_K initially
    filtered_df = data[data['Key'].apply(lambda x: is_span_length_less_equal_than_k_or_non_span(x, MAX_K))]
    filtered_df = filtered_df[~filtered_df['Key'].apply(lambda x: span_length_k_keys(x, 1))]

    print(f"Filtered df length (remain span <= {MAX_K}): {len(filtered_df)}")

    # Create a new DataFrame to store the results
    new_rows = []

    # Loop over k values to generate and insert average rows
    for k in range(2, MAX_K + 1):
        # Filter rows matching the `pre_xxx_end_xxx` pattern
        pre_end_rows = filtered_df[filtered_df['Key'].apply(lambda x: span_length_k_keys(x, k))]

        # Compute the average for each 'id' for the filtered rows
        averages = pre_end_rows.groupby('id')['Value'].mean().reset_index()
        averages.rename(columns={'Value': 'average_value'}, inplace=True)

        # Add a new row with the computed average and `span_k` as the key
        span_k_rows = averages.copy()
        span_k_rows['Key'] = f'span_{k}'
        span_k_rows['Value'] = span_k_rows['average_value']
        span_k_rows = span_k_rows[['id', 'Key', 'Value']]  # Keep necessary columns

        new_rows.append(span_k_rows)

        # Now, remove all rows that match the pattern `pre_xxx_end_xxx`
        filtered_df = filtered_df[~filtered_df['Key'].apply(lambda x: span_length_k_keys(x, k))]
        
        print(f"New df length after removing span_{k} rows: {len(filtered_df)}")

    # Concatenate all new rows into a single DataFrame and add them back to the original
    final_df = pd.concat([filtered_df] + new_rows, ignore_index=True)
    
    # Save the resulting DataFrame to a new CSV file
    final_df.to_csv(os.path.join(output_path, f'converted_{file}'), index=False)
    print(f"Processed and saved {file}.")
    feature_names = final_df.Key.unique()
    for feature_name in feature_names:
        print(f'Feature: {feature_name}')


parser = argparse.ArgumentParser()
parser.add_argument("--record_path", type=str, default="../data/feature_records/")
parser.add_argument("--output_path", type=str, default="../data/feature_records/0_clean_data")
args = parser.parse_args()
record_path = args.record_path
output_path = args.output_path

if not os.path.exists(output_path):
    os.makedirs(output_path)

args = parser.parse_args()
file_lists = [file_name for file_name in os.listdir(record_path) 
              if re.match(r'.*_data[a-z0-9\.]*csv', file_name) is not None] 

print(f'Files: {file_lists}')

for file in tqdm(file_lists):
    process_file(file)
