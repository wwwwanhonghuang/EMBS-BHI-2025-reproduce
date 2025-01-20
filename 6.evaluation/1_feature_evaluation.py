import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import argparse
import itertools


def sample_and_compute_mutual_info(featre_records, feature_name, sample_size=1000, n_iterations=10, evaluation_items=['normal', 'seizures']):
    def select_and_sample_feature_data(category, feature_name):
        data = featre_records[category][featre_records[category].Key == feature_name]
        return data.sample(n=min(sample_size, len(data)))
    mi_values = []
    for _ in range(n_iterations):
        sampled_data = {
            category: select_and_sample_feature_data(category, feature_name)
            for category in evaluation_items
        }
        X = []
        y = []
        for index, evaluation_item in enumerate(evaluation_items):
            values = list(sampled_data[evaluation_item].Value)          
            X += values
            y += [index] * len(values)
        mi = compute_mutual_information_continuous_discrete(X, y)
        mi_values.append(np.mean(mi))
        
    return np.mean(mi_values)

def compute_mutual_information_continuous_discrete(X, y):
    # X: Continuous feature(s) (dataframe or array)
    # y: Discrete target variable (array)
    mi = mutual_info_classif(np.array(X).reshape(-1, 1), np.array(y))
    return mi


def calculate_kl_divergence(p, q):
    # Ensure both distributions have the same length (e.g., padding with zeros if necessary)
    # p and q are histograms or probability mass functions
    return entropy(p, q)

def sample_and_compute_kl(featre_records, feature_name, sample_size=1000, n_iterations=100, comparision_pair=['normal', 'seizures']):
    kl_values = []  # To store KL divergence values for each iteration

    for _ in tqdm(range(n_iterations)):
        # Sample the data
        sampled_data = load_and_sample(featre_records, feature_name, sample_size)

        p = sampled_data[comparision_pair[0]].Value
        q = sampled_data[comparision_pair[1]].Value
        
        # Compute the histograms (discrete distributions)
        p_hist, _ = np.histogram(p, bins=50, density=True)
        q_hist, _ = np.histogram(q, bins=50, density=True)
        
        # Compute KL divergence for the sampled distributions
        kl_value = calculate_kl_divergence(p_hist + 1e-10, q_hist + 1e-10)  # Adding small value to avoid zero division
        kl_values.append(kl_value)
    
    return kl_values

def read_feature_csv_files(feature_files):
    featre_records = {'normal':[], 'seizures':[], 'preepileptic':[]}
    for file in feature_files:
        print(f'read file "{file}')
        entry = None
        if file.find("normal") >= 0:
            entry = "normal"
        elif file.find("seizures") >= 0:
            entry = "seizures"
        elif file.find("preepileptic") >= 0:
            entry = "preepileptic"
        else:
            raise KeyError
        featre_records[entry].append(pd.read_csv(os.path.join(record_base_path, file)))
    for key in featre_records:
        featre_records[key] = pd.concat(featre_records[key])
    return featre_records

def load_and_sample(featre_records, feature_name, sample_size=1000):
    # Sample from the 'normal' category for a specific key value
    normal_data = featre_records['normal'][featre_records['normal']['Key'] == feature_name]
    seizure_data = featre_records['seizures'][featre_records['seizures']['Key'] == feature_name]
    preepileptic_data = featre_records['preepileptic'][featre_records['preepileptic']['Key'] == feature_name]
    sampled_data = {
        'normal': normal_data.sample(n=min(sample_size, len(normal_data))),
        'seizures': seizure_data.sample(n=min(sample_size, len(seizure_data))) if 'seizures' in featre_records else [],
        'preepileptic': preepileptic_data.sample(n=min(sample_size, len(preepileptic_data))) if 'preepileptic' in featre_records else []
    }

    return sampled_data

def plot_kl_divergence_bar(kl_values, comparision_pair):
    # Prepare the data for plotting
    keys = list(kl_values.keys())  # List of keys (features)
    means = [np.mean(kl_values[key]) for key in keys]  # Mean KL divergence for each key
    stds = [np.std(kl_values[key]) for key in keys]  # Standard deviation for each key
    print(keys)
    # Create the bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(keys, means, yerr=stds, color='blue', label="KL Divergence")
    
    # Labeling the plot
    plt.xlabel('Feature Name')
    plt.ylabel('KL Divergence')
    plt.title(f'{comparision_pair[0]}-{comparision_pair[1]} Feature KL Divergence Plot')
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary for better readability
    plt.legend()
    plt.tight_layout()  # To ensure the plot is properly adjusted
    

def plot_mi_divergence_bar(mutual_information_values):
    keys = list(mutual_information_values.keys())  # List of keys (features)
    means = [np.mean(mutual_information_values[key]) for key in keys]  # Mean KL divergence for each key
    stds = [np.std(mutual_information_values[key]) for key in keys]  # Standard deviation for each key

    # Create the bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(keys, means, yerr=stds, capsize=5, color='blue', label="Mutual Information")
    
    # Labeling the plot
    plt.xlabel('Feature Name')
    plt.ylabel('Mutual Information')
    plt.title('Feature-Prediction Mutual Information Plot')
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary for better readability
    plt.legend()
    plt.tight_layout()  # To ensure the plot is properly adjusted

parser = argparse.ArgumentParser()
parser.add_argument("--save_kl", type=bool, default=True)
parser.add_argument("--save_mi", type=bool, default=True)
parser.add_argument("--kl_comparision_list", type=str, nargs="+", default=["normal", "seizures", "preepileptic"])
parser.add_argument("--mi_evaluation_list", type=str, nargs="+", default=["normal", "seizures", "preepileptic"])
parser.add_argument("--output_folder", type=str, default="./out")
parser.add_argument("--record_base_path", type=str, default="../data/feature_records/0_clean_data")
parser.add_argument("--sample_size", type=int, default=1000)

args = parser.parse_args()
record_base_path = args.record_base_path
output_folder = args.output_folder
save_KL = args.save_kl
save_MI = args.save_mi
sample_size = args.sample_size

print(f"record_base_path = {record_base_path}")
print(f"output_folder = {output_folder}")

feature_files = [file_name for file_name in os.listdir(record_base_path) if 
                 file_name[-4:] == '.csv' and file_name[:9] == "converted"]

if not os.path.exists(record_base_path):
    raise FileNotFoundError(f"Base path '{record_base_path}' does not exist.")

print(f'1. Read feature records...')

featre_records = read_feature_csv_files(feature_files)
unique_feature_names = [featre_records[key]['Key'].unique() for key in featre_records]
common_feature_names = set(unique_feature_names[0]) 
for features in unique_feature_names[1:]:
    common_feature_names &= set(features)

print(f'\t - Evaluate features = {common_feature_names}.\n')

# for comparision_pair in itertools.combinations(args.kl_comparision_list, 2):
#     print(f'2. Evaluate KL-divergence between features. (compare = {comparision_pair})')
#     kl_values = {}
#     for feature_name in tqdm(common_feature_names):
#         print(f'feature_name = {feature_name}')
#         kl_values |= {feature_name: sample_and_compute_kl(featre_records, feature_name, sample_size=sample_size, n_iterations=10, comparision_pair=comparision_pair) for key in feature_name}

#     if save_KL:
#         np.save(os.path.join(output_folder, f"kl_values_{comparision_pair[0]}_{comparision_pair[1]}.npy"), kl_values, allow_pickle = True)

#     plot_kl_divergence_bar(kl_values, comparision_pair)
#     plt.savefig(os.path.join(output_folder, 
#         f"feature_kl-divergence_compare_{comparision_pair[0]}_{comparision_pair[1]}.svg"))
    

print(f'3. Evaluate KL-divergence between features.')
mutual_information_values = {}
for feature_name in tqdm(common_feature_names):
    print(f'feature_name = {feature_name}')
    mutual_information_values |= \
        {feature_name: sample_and_compute_mutual_info(featre_records, 
            feature_name, sample_size=sample_size, n_iterations=10, evaluation_items=args.mi_evaluation_list)}
            

if save_MI:
    np.save(os.path.join(output_folder, 
        f"mi_values.npy"), mutual_information_values, allow_pickle = True)

plot_mi_divergence_bar(mutual_information_values)
plt.savefig(os.path.join(output_folder, 
    f'feature_prediction_mi.svg'))