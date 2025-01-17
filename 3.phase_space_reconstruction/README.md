## 3 Phase Space Reconstruction
This step objectives to segment the EEG signal in normal, pre-epileptic and seizure area, by identify the recurrency trajectories in
reconstructed EEG microstate's phase space.


### 3.1 Prepare Configuration file
The default configuration file should work.
``` json
{
    "dataset_name": "epileptic_eeg_dataset",
    "sids": [10, 11, 12, 13, 14, 15, "all"],
    "merged_record_ids": [[[10, 1], [10, 2]], 
                          [[11, 1], [11, 2], [11, 3], [11, 4]], 
                          [[12, 1], [12, 2]], 
                          [[13, 1], [13, 2], [13, 3], [13, 4]], 
                          [[14, 1]],
                          [[15, 1], [15, 2], [15, 3], [15, 4]], 
                          [[10, 1], [10, 2], [11, 1], [11, 2], [11, 3], [12, 1], [12, 2], [13, 1], [13, 2], [13, 3], [13, 4], [14, 1], [15, 1], [15, 2], [15, 3], [15, 4]]],
    "microstate_filename_form": "\\[seg\\-\\[prep\\-asr\\]\\]person_#{sid}_states4_gev_.*",
    "delay": 2,
    "n_states": 4,
    "corpus_storage_base_path": "../data/recurrence_sentence/epileptic_eeg_dataset/",
    "microstate_storage_base_path": "../data/microstates/epileptic_eeg_dataset/",
    "cut": [2, 800],
    "dataset_base_path": "../data/dataset"
}

```
+ `dataset_name` specifies the dataset's name, which should correspondent to a dataset folder name under <repository-root>/data/dataset.
+ `sids` specifies a set of data ID, which defined in the 'indexes' entries of configuration file (default =<repository-root>\2.microstate_training\configs\config-all-person-microstate.json) of microstate training.
+ `merged_record_ids` specifies a order-3 list, where merged_record_ids[sid_index] is a person record list. Each person record is a list of 2-element list, where the first element is the person id, and the second element is a record id for that person. For [x, y], it corresponds to the record file `p{x}_Record{y}.edf`.
+ `microstate_filename_form` entry defines a regex specifies the name pattern of microstate file.
+ `delay` specify the embeded dimension used in phase space reconstruction. (the `delay` name should be changed in the future)
+ `cut` specify the minimal and maximal recurrence sentence length.

Please kindly ensure microstate segmentation files are already generated from `2.microstate_training`, and exist in the folder specified by `microstate_storage_base_path` ^_^.

### 3.2 Run the script
``` bash
$ cd <path-to-3.phase_space_reconstruction>
$ python dataset_splitting_and_segmentation.py --configuration-file ./configs/epilepsy_all_person_intergrated.json
```

Command above will generate the sentences from 
all person's integrated EEG data.

We can specify the script run with a specific configuration file by append `--configuration-file <path-to-configuration-file>` in the 
running parameter. i.e., `python dataset_splitting_and_segmentation.py --configuration-file <path-to-configuration-file>`.

We can also generate sentence for each person  and the integrated 'person' respectively.
In this case, follow command should work
``` bash
$ cd <path-to-3.phase_space_reconstruction>
$ python dataset_splitting_and_segmentation.py --configuration-file ./configs/epilepsy_all_person_intergrated.json
```