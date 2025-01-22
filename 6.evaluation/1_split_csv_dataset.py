import pandas as pd
import os
import yaml
from utils import load_yaml_config
import numpy as np

config = load_yaml_config("config.yaml")['data_set_splitting']
ratio = config['retrain_ratio']

record_base_path = config['record_base_path']
output_folder = config['output_folder']
files = config["files"]
os.makedirs(output_folder, exist_ok=True)

for file in files:
    file_path = os.path.join(record_base_path, file)
    dataset = pd.read_csv(file_path)
    dataset['id'].astype(int)
    cnt_sentences = len(dataset.id.unique())
    cnt_select_sentences = int(ratio * cnt_sentences) 
    select_sentences = list(range(cnt_sentences))
    np.random.shuffle(select_sentences)
    select_sentences = select_sentences[:cnt_select_sentences]
    retain_data = dataset[dataset.id.isin(select_sentences)]
    data_use_in_feature_investigation = dataset[~dataset.id.isin(select_sentences)]
    retain_data.to_csv(os.path.join(output_folder, "retain_" + file))
    data_use_in_feature_investigation.to_csv(os.path.join(output_folder, "use_in_feature_evaluation_" + file))
    

    