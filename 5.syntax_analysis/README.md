## 5 Syntax Analysis
### 5.1 Build the project
> [!NOTE]
> This step can be skipped if `4.1` ([4.pcfg_training](../4.pcfg_training/README.md#41-build-the-project)) has been completed

``` bash
$ cd <repository_root/lib/pcfg-cky-inside-outside>
$ cmake .
$ make -j
```

### 5.2 Configure
Edit the `syntax_analysis` section of `<repository_root/lib/pcfg-cky-inside-outside>/config.yaml`.

Example:
``` json
syntax_analysis:
  grammar_file: "../../data/pcfg/grammar-trained.pcfg"
  input: "../../data/pre-epileptic_integrated_all_d2_s4_converted.txt"
  log_intervals: 1
  log_path: "../../data/logs/pre-epileptic/"
  report_path: "../../data/reports/pre-epileptic/"
  serialize_to_files: false
  tree_serialization_path: "../../data/serialized_tree/seizure/"
```