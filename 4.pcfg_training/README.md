## 4. PCFG Training
### 4.1  Build the project
```
$ cd <repository_root/lib/pcfg-cky-inside-outside>
$ cmake .
$ make -j
```

### 4.2  Prepare the configuration
In `<repository-root>/lib\pcfg-cky-inside-outside`, there is a `config.yaml` configuration file.
The content should contains the `main` section seems like:

``` yaml
main:
  grammar_file: "data/grammar.pcfg"
  input: "data/sentences_converted.txt"
  log_intervals: 10000
  log_path: "data/logs/"
  log_f:
    enabled: true
    intervals: 10000
  log_warning_in_training: false
  batch_size_for_parameter_update: -1
  split_data:
    enabled: true
    val_dataset_path: "data/validate_sentences.txt" 
    train_dataset_path: "data/train_sentences.txt"
    train_fraction: 0.8
  n_epochs: 5
  limit_n_sentences: -1
```
This section setting the configuration for PCFG training.
`grammar_file` specifies a grammar file path.
`input` specifies a file path of sentences.
`log_intervals` specifies the logging intervals (numbers of sentences)
`batch_size_for_parameter_update` specifies the batch size used to update the PCFG parameters. Set to `-1` to make it only update at the end of a epoch. 
`n_epochs` specifies the numbers of epoches.
`split_data` specifies the dataset splitting  parameters.

### 4.3 Convert npy files to txt Files
``` bash
$ python sentence_plain_text_encoder.py --file_path ../data/recurrence_sentence/epileptic_eeg_dataset/seizure_integrated_all_d2_s4.npy --output_file_path ../data/recurrence_sentence/epileptic_eeg_dataset/seizure_integrated_all_d2_s4.txt
$ python sentence_plain_text_encoder.py --file_path ../data/recurrence_sentence/epileptic_eeg_dataset/normal_integrated_all_d2_s4.npy --output_file_path ../data/recurrence_sentence/epileptic_eeg_dataset/normal_integrated_all_d2_s4.txt
$ python sentence_plain_text_encoder.py --file_path ../data/recurrence_sentence/epileptic_eeg_dataset/pre-epileptic_integrated_all_d2_s4.npy --output_file_path ../data/recurrence_sentence/epileptic_eeg_dataset/pre-epileptic_integrated_all_d2_s4.txt
```

### 4.4 Phase Space Reconstruction
Ensure setting up the `phase_convert` section in `4.pcfg_training\pcfg-cky-inside-outside\config.yaml`. The default one may work.
```
$ cd <repository-root>/lib\pcfg-cky-inside-outside>
$ ./bin/phase_convert
```

### 4.5 Train PCFG
```
$ cd <repository-root>/lib\pcfg-cky-inside-outside>
$ ./bin/main_executable
```