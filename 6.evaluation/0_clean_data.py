import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm
import os

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
        return num2 - num1 + 1 <= k  # Check if second number is first + k
    return True

base_path = "/data1/feature_records/"
file_lists = [file_name for file_name in os.listdir(base_path) if re.match(r'.*_data[a-z0-9\.]*csv', file_name) is not None] 
print(f'files: {file_lists}')



def process_file(file, base_path, MAX_K=3):
    record_file_path = os.path.join(base_path, file)
    data = pd.read_csv(record_file_path, sep=',')
    
    # Filter out spans with length greater than MAX_K initially
    filtered_df = data[data['Key'].apply(lambda x: is_span_length_less_equal_than_k_or_non_span(x, MAX_K))]
    print(f"Filtered df length (span <= {MAX_K}): {len(filtered_df)}")

    # Create a new DataFrame to store the results
    new_rows = []

    # Loop over k values to generate and insert average rows
    for k in range(2, MAX_K):
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
    final_df.to_csv(os.path.join(base_path, f'converted_{file}'), index=False)
    print(f"Processed and saved {file}.")
    feature_names = final_df.Key.unique()
    for feature_name in feature_names:
        print(f'feature: {feature_name}')

for file in tqdm(file_lists):
    process_file(file, base_path)
