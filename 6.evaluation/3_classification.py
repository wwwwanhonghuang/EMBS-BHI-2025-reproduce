import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import yaml
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, matthews_corrcoef, log_loss, cohen_kappa_score,
    average_precision_score, hamming_loss, brier_score_loss, fbeta_score,
    precision_recall_curve, auc, fowlkes_mallows_score
)




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

    dataset_config = yaml_config.get("dataset_for_classification", {}).get("dataset", {})
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
        # Ensure the 'Key' column exists and is being used for filtering
        if 'Key' in dataset[category]['X'].columns:
            # Filter based on 'Key' column matching the selected features
            dataset[category]['X'] = dataset[category]['X'][dataset[category]['X']['Key'].isin(select_features)]
        else:
            print(f"Warning: 'Key' column not found in {category} category.")
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

# Define classifiers
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, default="./out")
parser.add_argument("--record_base_path", type=str, 
    default="../data/feature_records/0_clean_data")
parser.add_argument("--configuration_file", type=str, default="./config.yaml")

args = parser.parse_args()
record_base_path = args.record_base_path
output_folder = args.output_folder
configuration_file_path = args.configuration_file

yaml_config = load_yaml_config(configuration_file_path)

if not yaml_config['dataset_for_classification']['use_prepared_dataset']['enabled']:
    # load pareto optimal feature names.
    pareto_optimal_feature_names = np.load(os.path.join(output_folder, "pareto_optimal_features.npy"), allow_pickle=True)  
    print(f'Features Names = {pareto_optimal_feature_names}.')

    # load dataset
    dataset = load_datasets_from_yaml(yaml_config)
    feature_filtering(dataset, pareto_optimal_feature_names)

    # integrate the dataset
    dataset = integrate_dataset(dataset, pareto_optimal_feature_names)
else:
    dataset = pd.read_csv(yaml_config['dataset_for_classification']['use_prepared_dataset']['dataset_path'])

# Training step 1: Prepare features and labels
X = np.array(list(dataset.drop(columns=['id', 'pred'])['features']))
y = np.array(dataset['pred'])

# Find rows without NaN in X
non_nan_indices = ~np.isnan(X).any(axis=1)

# Keep only valid rows
X = X[non_nan_indices]
y = y[non_nan_indices]
# K-fold cross-validation setup
kf = StratifiedKFold(n_splits=yaml_config['dataset_for_classification']['n_splits'], 
    shuffle=True)

# Initialize result storage
metrics = {clf_name: {
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
} for clf_name in classifiers.keys()}

y[y == 2] = 1

for clf_name, clf in classifiers.items():
    print(f"Training classifier: {clf_name}")
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else np.zeros_like(y_pred)
        
        # Compute metrics
        metrics[clf_name]['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics[clf_name]['precision'].append(precision_score(y_test, y_pred, average='weighted'))
        metrics[clf_name]['recall'].append(recall_score(y_test, y_pred, average='weighted'))
        metrics[clf_name]['f1'].append(f1_score(y_test, y_pred, average='weighted'))
        if hasattr(clf, "predict_proba"):
            metrics[clf_name]['roc_auc'].append(roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
            metrics[clf_name]['log_loss'].append(log_loss(y_test, y_pred_prob))
            metrics[clf_name]['average_precision'].append(average_precision_score(y_test, y_pred_prob))
            metrics[clf_name]['brier_score'].append(brier_score_loss(y_test, y_pred_prob))
        metrics[clf_name]['mcc'].append(matthews_corrcoef(y_test, y_pred))
        metrics[clf_name]['cohen_kappa'].append(cohen_kappa_score(y_test, y_pred))
        metrics[clf_name]['hamming_loss'].append(hamming_loss(y_test, y_pred))
        metrics[clf_name]['f2_score'].append(fbeta_score(y_test, y_pred, beta=2, average='weighted'))
        
        # Precision-Recall AUC
        if hasattr(clf, "predict_proba"):
            precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
            pr_auc = auc(recall, precision)
            metrics[clf_name]['pr_auc'].append(pr_auc)
        
        # Fowlkes-Mallows Index
        metrics[clf_name]['fmi'].append(fowlkes_mallows_score(y_test, y_pred))

# Combine results into a DataFrame for each metric
metrics_summary = {}
for metric_name in list(metrics[list(classifiers.keys())[0]].keys()):
    metric_data = {clf_name: np.mean(metrics[clf_name][metric_name]) for clf_name in classifiers.keys()}
    metrics_summary[metric_name] = metric_data

# Convert to DataFrame for easier plotting
metrics_df = pd.DataFrame(metrics_summary).T

# Save metrics
np.save("metrics_summary.npy    ", metrics_summary, allow_pickle=True)
metrics_df.to_csv(os.path.join(output_folder, "classification_metrics_comparison.csv"))
print(metrics_df)

# Plot metrics
fig, ax = plt.subplots(figsize=(12, 8))
metrics_df.plot(kind='bar', ax=ax, edgecolor='black', linewidth=2)
ax.set_title("Metrics Comparison Across Classifiers")
ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.legend(title="Classifier", loc='upper right')
plt.ylim([0, 1])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "classification_metrics_comparison.pdf"))
plt.show()
