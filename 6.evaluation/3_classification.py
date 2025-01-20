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
import yaml
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef, log_loss, cohen_kappa_score,
    average_precision_score, hamming_loss, brier_score_loss, fbeta_score,
    precision_recall_curve, auc, fowlkes_mallows_score
)
import numpy as np
import pandas as pd



def load_yaml_config(yaml_file):
    """
    Load a YAML configuration file.

    Args:
        yaml_file (str): Path to the YAML file.
    
    Returns:
        dict: Parsed YAML data as a Python dictionary.
    """
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def load_datasets_from_yaml(yaml_config, label_map = {'normal': 0, 'pre_epileptic': 1, 'seizure': 2}):
    """
    Load datasets based on YAML configuration.

    Args:
        yaml_config (dict): Dictionary from parsed YAML configuration.
    
    Returns:
        pd.DataFrame: Combined DataFrame with a 'category' column.
    """

    dataset_config = yaml_config.get("dataset_for_classification", {})
    dataset = {}
    for category, file_paths in dataset_config.items():
        combined_data = []

        for file_path in file_paths:
            try:
                # Load CSV file
                data = pd.read_csv(file_path)
                # Add a column to indicate the category
                data['category'] = category
                # Append to the combined dataset
                combined_data.append(data)
            except FileNotFoundError:
                print(f"Warning: File not found - {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        concatenated_data = pd.concat(combined_data)
        dataset[category] = {"X":concatenated_data,  'y': np.array([label_map[category]] * len(concatenated_data))}
    return dataset

def feature_filtering(dataset, select_features):
    for category in dataset:
        dataset[category]['X'] = dataset[category]['X'][select_features]
    return dataset

def integrate_dataset(dataset, selected_features):
    # Initialize an empty list to hold the processed rows
    integrated_data = []

    # Iterate over each category in the dataset
    for category in dataset:
        # Group the data by instance_id
        grouped = dataset[category]['X'].groupby('id')

        for instance_id, group in grouped:
            # Initialize a dictionary to store feature values for this instance
            instance_data = {'id': instance_id}

            # Create a list of feature values for the selected features
            feature_values = []
            for feature in selected_features:
                # Find the value for the selected feature, or NaN if not present
                feature_value = group[group['Key'] == feature]['Value']
                if not feature_value.empty:
                    feature_values.append(feature_value.iloc[0])  # Take the first match (there should be only one per instance)
                else:
                    feature_values.append(np.nan)  # Fill with NaN if feature is missing
            
            # Assign the sorted features list and the prediction
            instance_data['features'] = feature_values
            instance_data['pred'] = dataset[category]['y'][group.index[0]]  # Use the first index for prediction

            # Append to the integrated data list
            integrated_data.append(instance_data)

    # Convert the integrated data into a DataFrame
    return pd.DataFrame(integrated_data)

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, default="./out")
parser.add_argument("--record_base_path", type=str, default="/data1")
parser.add_argument("--configuration_file", type=str, default="./config.yaml")

args = parser.parse_args()
record_base_path = args.record_base_path
output_folder = args.output_folder
configuration_file_path = args.configuration_file

yaml_config = load_yaml_config(configuration_file_path)

if not yaml_config['use_prepared_dataset']['enabled']:
    # load pareto optimal feature names.
    pareto_optimal_feature_names = np.load(os.path.join(output_folder, "pareto_optimal_features.npy", allow_pickle=True))  
    print(f'Features Names = {pareto_optimal_feature_names}.')

    # load dataset
    dataset = load_datasets_from_yaml(yaml_config)
    feature_filtering(dataset, pareto_optimal_feature_names)

    # integrate the dataset
    dataset = integrate_dataset(dataset, pareto_optimal_feature_names)
else:
    dataset = pd.read_csv(yaml_config['use_prepared_dataset']['dataset_path'])

# Training step 1: Prepare features and labels
X = dataset.drop(columns=['id', 'pred'])
y = dataset['pred']

# K-fold cross-validation setup
kf = StratifiedKFold(n_splits=yaml_config['n_splits'], shuffle=True)

# Initialize result storage
metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'roc_auc': [],
    'mcc': [],
    'log_loss': [],
    'cohen_kappa': [],
    'average_precision': [],
    'hamming_loss': [],
    'brier_score': [],
    'f2_score': [],
    'pr_auc': [],
    'fmi': []
}

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] 

    # Calculate all metrics
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
    metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
    metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
    metrics['roc_auc'].append(roc_auc_score(y_test, y_pred_prob))
    metrics['mcc'].append(matthews_corrcoef(y_test, y_pred))
    metrics['log_loss'].append(log_loss(y_test, y_pred_prob))
    metrics['cohen_kappa'].append(cohen_kappa_score(y_test, y_pred))
    metrics['average_precision'].append(average_precision_score(y_test, y_pred_prob))
    metrics['hamming_loss'].append(hamming_loss(y_test, y_pred))
    metrics['brier_score'].append(brier_score_loss(y_test, y_pred_prob))
    metrics['f2_score'].append(fbeta_score(y_test, y_pred, beta=2, average='weighted'))
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    metrics['pr_auc'].append(pr_auc)
    
    # Fowlkes-Mallows Index
    metrics['fmi'].append(fowlkes_mallows_score(y_test, y_pred))

# Convert the results into a pandas DataFrame for easy analysis
metrics_df = pd.DataFrame(metrics)

# Display the mean and std of all metrics
print(metrics_df.mean())
print(metrics_df.std())
if yaml_config['save_evaluation_metrics_data']:
    metrics_df.to_csv(os.path.join(output_folder, "metrics_df.csv"))