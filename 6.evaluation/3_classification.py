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

# 1. load pareto optimal feature names.
pareto_optimal_feature_names = np.load(os.path.join(output_folder, "pareto_optimal_features.npy", allow_pickle=True))  
print(f'Features Names = {pareto_optimal_feature_names}.')

# 2. load dataset
yaml_config = load_yaml_config(configuration_file_path)
dataset = load_datasets_from_yaml(yaml_config)
feature_filtering(dataset, pareto_optimal_feature_names)

# 3. Integrate the dataset
dataset = integrate_dataset(dataset, pareto_optimal_feature_names)


from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score, matthews_corrcoef,
                             log_loss, brier_score_loss, hamming_loss)

# Step 1: Prepare features and labels
X = dataset.drop(columns=['instance_id', 'pred'])
y = dataset['pred']

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Step 4: Predictions
y_pred = classifier.predict(X_test)

# Step 5: Metrics Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mcc = matthews_corrcoef(y_test, y_pred)

print(f'Precision (Weighted): {precision}')
print(f'Recall (Weighted): {recall}')
print(f'F1 Score (Weighted): {f1}')
print(f'MCC: {mcc}')

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc}')

# Precision-Recall Curve and AUC
precision, recall, _ = precision_recall_curve(y_test, classifier.predict_proba(X_test)[:, 1])
pr_auc = average_precision_score(y_test, classifier.predict_proba(X_test)[:, 1])
print(f'Precision-Recall AUC: {pr_auc}')

# Log Loss
logloss = log_loss(y_test, classifier.predict_proba(X_test))
print(f'Log Loss: {logloss}')

# Brier Score
brier_score = brier_score_loss(y_test, classifier.predict_proba(X_test)[:, 1])
print(f'Brier Score: {brier_score}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')


