import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from utils import load_yaml_config


parser = argparse.ArgumentParser()
parser.add_argument("--output_folder", type=str, default="./out")
parser.add_argument("--output_file", type=str, default="feature_filtered.npy")

parser.add_argument("--record_base_path", type=str, 
    default="/data1/feature_records/0_clean_data")
parser.add_argument("--configuration_file", type=str, default="./config.yaml")

args = parser.parse_args()
record_base_path = args.record_base_path
output_folder = args.output_folder
configuration_file_path = args.configuration_file

yaml_config = load_yaml_config(configuration_file_path)



def filtering(features):
    import re
    features = filter(lambda feature_name: re.match(r'word_delay_.*', feature_name) is None, features)
    features = filter(lambda feature_name: re.match(r'symbol_entropy.*', feature_name) is None, features)

    return list(features)

if not yaml_config['dataset_for_classification']['use_prepared_dataset']['enabled']:
    # load pareto optimal feature names.
    pareto_optimal_feature_names = np.load(os.path.join(output_folder, "pareto_optimal_features.npy"), allow_pickle=True)  
    filtered_feature_names = filtering(pareto_optimal_feature_names)
    print(f'Features Names = {filtered_feature_names}. Len = {len(filtered_feature_names)}')
    np.save(os.path.join(args.output_folder, args.output_file), filtered_feature_names, allow_pickle = True)