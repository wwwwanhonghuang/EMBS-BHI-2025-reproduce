## 4. PCFG Training
### 4.1  Build the project
``` bash
$ cd <repository_root/lib/pcfg-cky-inside-outside>
$ git checkout master
$ cmake .
$ make train_pcfg -j
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

### 4.3 Convert npy form of microstate sequences to plain text form
``` bash
$ python sentence_plain_text_encoder.py --file_path ../data/recurrence_sentences/epileptic_eeg_dataset/seizure_integrated_all_d2_s4.npy --output_file_path ../data/recurrence_sentences/epileptic_eeg_dataset/seizure_integrated_all_d2_s4.txt
$ python sentence_plain_text_encoder.py --file_path ../data/recurrence_sentences/epileptic_eeg_dataset/normal_integrated_all_d2_s4.npy --output_file_path ../data/recurrence_sentences/epileptic_eeg_dataset/normal_integrated_all_d2_s4.txt
$ python sentence_plain_text_encoder.py --file_path ../data/recurrence_sentences/epileptic_eeg_dataset/pre-epileptic_integrated_all_d2_s4.npy --output_file_path ../data/recurrence_sentences/epileptic_eeg_dataset/pre-epileptic_integrated_all_d2_s4.txt
```
Or run the `convert_npy_to_plain_text.sh` script with command `sh convert_npy_to_plain_text.sh`.


### 4.4 Convert microstate sequences to word sequences
[4.3](./README.md#43-convert-npy-files-to-txt-files) generate the sentence, i.e., microstate sequences.

Before training, we need convert the microstate sequence into correct word id, according a PCFG file.

To do this, firstly, ensuring setting up the `phase_convert` section in configuration file `4.pcfg_training\pcfg-cky-inside-outside\config.yaml`. 
The default configuration file should work^.

``` bash
$ sh make_phase_converter.sh
$ sh phase_space_reconstruction.sh
```

### 4.5 Train PCFG
Still we need prepare a configuration file.
The default configuration file `4.pcfg_training/config_train.yaml` should work.

``` bash
$ cd <repository-root>/lib/pcfg-cky-inside-outside
$ ./bin/train_pcfg ../../4.pcfg_training/config_train.yaml
```
