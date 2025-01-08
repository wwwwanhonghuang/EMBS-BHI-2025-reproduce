
# 1. Dataset Preparation
The Epileptic EEG Dataset [1] can be download from https://data.mendeley.com/datasets/5pc2j46cbc/1
Or follow command may help:
```
$ cd <path-to-1.dataset_preparation>
$ wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5pc2j46cbc-1.zip -O epileptic_eeg_dataset.zip
$ unzip epileptic_eeg_dataset.zip
$ rm -rf 
$ mkdir -p ../data/dataset/epileptic_eeg_dataset
$ mv Raw_EDF_Files ../data/dataset/epileptic_eeg_dataset
```

Processes above can also be done by running the script in this folder:
```
$ bash prepare_dataset.sh
```

## Citation
Nasreddine, Wassim (2021), “Epileptic EEG Dataset”, Mendeley Data, V1, doi: 10.17632/5pc2j46cbc.1