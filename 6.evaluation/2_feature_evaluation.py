import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import argparse
import itertools
from utils import load_yaml_config

def sample_and_compute_mutual_info(featre_records, feature_name, sample_size=1000, n_iterations=10, evaluation_items=['normal', 'seizures', "preepileptic"]):
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
    
    return mi_values

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

def read_feature_csv_files(feature_files, record_base_path):
    """
    Reads feature CSV files and organizes them into categories.

    Parameters:
        feature_files (dict): A dictionary with keys as categories (e.g., 'normal', 'seizures') 
                              and values as lists of filenames for each category.
        record_base_path (str): The base directory where the files are stored.

    Returns:
        dict: A dictionary containing concatenated dataframes for each category.
    """
    feature_records = {'normal': [], 'seizures': [], 'preepileptic': []}
    
    # Iterate through the categories
    for category in feature_records:
        files = feature_files.get(category, [])
        for file in files:
            print(f'Reading file "{file}" for category "{category}"...')
            file_path = os.path.join(record_base_path, file)
            
            # Check if file exists before reading
            if not os.path.exists(file_path):
                print(f"Warning: File '{file}' does not exist. Skipping...")
                continue

            # Append the dataframe to the list for the current category
            feature_records[category].append(pd.read_csv(file_path))
        
        # Concatenate all dataframes for the category (if any files were found)
        if feature_records[category]:
            feature_records[category] = pd.concat(feature_records[category], ignore_index=True)
        else:
            print(f"No files were loaded for category '{category}'.")
            feature_records[category] = pd.DataFrame() 
    
    return feature_records


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

def plot_kl_divergence_bar(all_kl_values, categories):
    kl_components = {category: np.array([all_kl_values[feature_name][category] for feature_name in all_kl_values.keys()]) for category in categories} 
    all_category_means = {category: np.mean(kl_components[category], axis=2) for category in categories}
    all_category_stds = {category: np.std(kl_components[category], axis=2) for category in categories}

    # Create the stacked bar plot with error bars
    _, ax = plt.subplots(figsize=(36, 32))
    cmap = plt.cm.viridis
    color_palette = cmap(np.linspace(0, 1, len(all_kl_values.keys())))# plt.cm.tab10.colors  
    offset = 0.36

    for category_index, category in enumerate(categories):
        # print(sorted(zip(means, keys, range(len(means)))))
       
        means = all_category_means[category]
        stds = all_category_stds[category]
        for index, (_, feature_name, feature_id) in enumerate(sorted(zip(means, all_kl_values.keys(), range(len(means))), reverse=True)):
            # Assign a unique color to each feature based on its index
            color = color_palette[feature_id % len(color_palette)]  # Ensure we don't run out of colors
            x_position = category_index * 11 - offset * index  # Shift each bar slightly to the right
            # Overlap bars with a slight transparency (alpha)
            ax.bar(x_position, means[feature_id], bottom=0,
                color=color, alpha=0.7, yerr=stds[feature_id], capsize=2)
            
            # Update the last_bar position for the next iteration
            if category_index == 0:
                ax.plot([x_position, x_position - 2], [means[feature_id], means[feature_id]], color=color, linewidth=2)
                ax.plot([x_position - 2, x_position - 2], [means[feature_id], means[feature_id] + 2], color=color, linewidth=2)
                ax.text(x_position - 2 - 0.02, means[feature_id] + 2, feature_name, ha='right', va='center', fontsize=10, color='black', rotation=90)
    
    # Labeling the plot
    ax.set_xticks([])
    x_ticks_labels = categories
    ax.set_xticks([11 * i + len(all_kl_values.keys()) * offset / 2 - 10 for i in range(len(categories))])
    ax.set_xticklabels(x_ticks_labels, rotation=0, fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=28)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_ylabel('KL Divergence', fontsize=36)
    ax.legend()

    plt.tight_layout()

def plot_mi_divergence_bar(mutual_information_values, categories):    
    mi_components = {category: np.array([mutual_information_values[feature_name][category] for feature_name in mutual_information_values.keys()]) for category in categories}  # Extract the MI components
    
    all_category_means = {category: np.mean(mi_components[category], axis=2) for category in categories}
    
    all_category_stds = {category: np.std(mi_components[category], axis=2) for category in categories}

    # Create the stacked bar plot with error bars
    fig, ax = plt.subplots(figsize=(36, 32))

    color_palette = plt.cm.tab10.colors  
    offset = 0.36
    for category_index, category in enumerate(categories):
        # print(sorted(zip(means, keys, range(len(means)))))
       
        means = all_category_means[category]
        stds = all_category_stds[category]
        for index, (feature_value, feature_name, feature_id) in enumerate(sorted(zip(means, mutual_information_values.keys(), range(len(means))), reverse=True)):
            # Assign a unique color to each feature based on its index
            color = color_palette[feature_id % len(color_palette)]  # Ensure we don't run out of colors
            x_position = category_index * 11 - offset * index  # Shift each bar slightly to the right
            # Overlap bars with a slight transparency (alpha)
            ax.bar(x_position, means[feature_id], bottom=0,
                color=color, alpha=0.7, yerr=stds[feature_id], capsize=2)
            
            # Update the last_bar position for the next iteration
            if category_index == 0:
                ax.plot([x_position, x_position - 2], [means[feature_id], means[feature_id]], color=color, linewidth=2)
                ax.plot([x_position - 2, x_position - 2], [means[feature_id], means[feature_id] + 0.1], color=color, linewidth=2)

                ax.text(x_position - 2 - 0.02, means[feature_id] + 0.1, feature_name, ha='right', va='center', fontsize=10, color='black', rotation=90)

    
    # Labeling the plot
    ax.set_xticks([])
    x_ticks_labels = categories
    ax.set_xticks([11 * i + len(mutual_information_values.keys()) * offset / 2 - 7 for i in range(len(categories))])
    ax.set_xticklabels(x_ticks_labels, rotation=0, fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=28)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_ylabel('Mutual Information', fontsize=36)
    ax.legend()

    plt.tight_layout()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_kl", type=bool, default=True)
    parser.add_argument("--save_mi", type=bool, default=True)
    parser.add_argument("--kl_comparision_list", type=str, nargs="+", default=["normal", "seizures", "preepileptic"])
    parser.add_argument("--mi_evaluation_list", type=str, nargs="+", default=["normal", "seizures", "preepileptic"])
    parser.add_argument("--output_folder", type=str, default="./out")
    parser.add_argument("--record_base_path", type=str, default="../data/feature_records/0_clean_data")
    parser.add_argument("--sample_size", type=int, default=1000)

    args = parser.parse_args()
    return args

def process_KL_values(feature_records, feature_names, kl_comparision_pairs, sample_size=1000, n_iterations=10):
    all_kl_values = {
        
    }

    for comparision_pair in kl_comparision_pairs:
        print(f'2. Evaluate KL-divergence between features. (compare = {comparision_pair})')
        category = f'{comparision_pair[0]}-{comparision_pair[1]}'
        for feature_name in tqdm(feature_names):
            print(f'feature_name = {feature_name}')
            if feature_name not in all_kl_values:
                all_kl_values[feature_name] = {}
            if category not in  all_kl_values[feature_name]:
                all_kl_values[feature_name][category] = []
                
            samples = sample_and_compute_kl(feature_records, feature_name, 
                sample_size=sample_size, n_iterations=n_iterations, comparision_pair=comparision_pair)
            all_kl_values[feature_name][category].append(samples)
    return all_kl_values

def _init_mutual_information_value_plot_category(mutual_information_values, feature_name, category_name):
    if category_name not in mutual_information_values[feature_name]:
        mutual_information_values[feature_name][category_name] = []


def process_mutual_information(feature_records, feature_names, 
                               mutual_information_values, sample_size=1000, n_iterations=10):
    for feature_name in tqdm(feature_names):
        print(f'feature_name = {feature_name}')
        if feature_name not in mutual_information_values:
            mutual_information_values[feature_name] = {}
        
        evaluation_list_map = {
            'All': ["normal", "seizures", "preepileptic"],
            'Normal-Seizures': ["normal", "seizures"], 
            'Normal-Pre_epileptic': ["normal", "preepileptic"], 
            'Seizures-Pre_epileptic': ["seizures", "preepileptic"]
        }
        for category in categories:
            _init_mutual_information_value_plot_category(mutual_information_values, feature_name, category)

        for category in categories:
            samples = sample_and_compute_mutual_info(feature_records, 
                feature_name, sample_size=sample_size, n_iterations=n_iterations, 
                evaluation_items=evaluation_list_map[category])
            mutual_information_values[feature_name][category].append(samples)
            
    return mutual_information_values

args = parse_args()
config = load_yaml_config("config.yaml")['feature_evaluation']
record_base_path = config['record_base_path']
output_folder = config['output_folder']
save_KL = config['save_KL']
save_MI = config['save_MI']
sample_size = config['sample_size']

kl_comparision_pairs = list(itertools.combinations(args.kl_comparision_list, 2))
categories = ['All', 'Normal-Seizures', 'Normal-Pre_epileptic', 'Seizures-Pre_epileptic']

print(f"record_base_path = {record_base_path}")
print(f"output_folder = {output_folder}")

feature_files = config["files"]
if not os.path.exists(record_base_path):
    raise FileNotFoundError(f"Base path '{record_base_path}' does not exist.")

print(f'1. Read feature records...')
print(feature_files)


feature_records = read_feature_csv_files(feature_files, record_base_path)
unique_feature_names = [
    set(feature_records[category]['Key'].unique()) 
    for category in feature_records
]
common_feature_names = unique_feature_names[0]
for features in unique_feature_names[1:]:
    common_feature_names &= features  # Intersection of sets

print(f'\t - Common features = {common_feature_names}.\n')


# sample_size is the sample size of dataset utilize in KL calculations. 
# `n_iterations`` equal the amount of samples of KL calculation.
all_kl_values = process_KL_values(feature_records=feature_records, feature_names=common_feature_names, 
                                  kl_comparision_pairs=kl_comparision_pairs, sample_size=sample_size, n_iterations=100)
if save_KL:
    np.save(os.path.join(output_folder, f"all_kl_values.npy"), all_kl_values, allow_pickle = True)

plot_kl_divergence_bar(all_kl_values, [f'{pair[0]}-{pair[1]}' for pair in kl_comparision_pairs])
plt.savefig(os.path.join(output_folder, f"feature_kl-divergence_comparision.pdf"))

print(f'3. Evaluate mutual information between features and classification labels.')

mutual_information_values = {}

mutual_information_values = process_mutual_information(feature_records=feature_records, feature_names=common_feature_names,
        mutual_information_values=mutual_information_values, sample_size=sample_size, n_iterations=100)
if save_MI:
    np.save(os.path.join(output_folder, f"mi_values.npy"), 
            mutual_information_values, allow_pickle = True)

plot_mi_divergence_bar(mutual_information_values, categories)
plt.savefig(os.path.join(output_folder, f'feature_prediction_mi.pdf'))