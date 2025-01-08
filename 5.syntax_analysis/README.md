## 5 Syntax Analysis
### 5.1 Build the project
> [!NOTE]
> This step can be skipped if `4.1` ([4.pcfg_training](../4.pcfg_training/README.md#41-build-the-project)) has been completed

``` bash
$ cd <repository_root/lib/pcfg-cky-inside-outside>
$ cmake .
$ make syntax_analysis -j
```

### 5.2 Configure
The three default configurations files in this folder should work.
Optionally edit these configurations if necessary.

Example:
``` json
syntax_analysis:
  grammar_file: "../../data/pcfg/grammar-trained.pcfg"
  input: "../../data/pre-epileptic_integrated_all_d2_s4_converted.txt"
  log_intervals: 100000
  log_path: "../../data/logs/pre-epileptic/"
  report_path: "../../data/reports/pre-epileptic/"
  serialize_to_files: false
  tree_serialization_path: "../../data/serialized_tree/seizure/"
```
+ `grammar_file` specify a pcfg grammar file.
+ `input` specify a plain text file, each line is a sentence, with space is used for separating words. This input file can be generated according steps from [3.phase space reconstruction](../3.phase_space_reconstruction/README.md) to 4.pcfg_training ([Section 4.3](../4.pcfg_training/README.md#43-convert-npy-files-to-txt-files))
+ `serialize_to_files` specify whether the program should also save the serialized parse trees. This was set to `false`, as it can take more than `400` GB disk storage.

### 5.3 Run the binary
``` bash
$ cd <repository_root/lib/pcfg-cky-inside-outside>
$ ./bin/syntax_analysis 5.syntax_analysis/config_seizure.yaml
$ ./bin/syntax_analysis 5.syntax_analysis/config_preepileptic.yaml
$ ./bin/syntax_analysis 5.syntax_analysis/config_normal.yaml
```