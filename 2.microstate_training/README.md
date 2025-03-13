## Step 2: Train Microstate Maps from Epileptic Dataset

### 2.0 Prerequisites
[Library Version in Development]
1. mne == 1.7.1
2. numpy == 1.26.4
3. pandas == 2.2.2
4. hmmlearn == 0.3.2
5. pyprep == 0.4.3
> [!CAUTION]
> We modified the definition pyprep's `fit()` and `reference()` function to make the parameters for training can be passed explicitly.
> The scripts cannot work under original `pyprep` and will report exception like passed parameters more than require to the prep functions.
> **Please kindly run `sh patch_prep.sh` after install `pyprep` to apply this patch^.**
> The modification history can be seen from the history of https://github.com/wwwwanhonghuang/pyprep, which is forked from https://github.com/sappelhoff/pyprep
6. asrpy == 0.0.3



> [!CAUTION]
> **`sh patch_prep.sh` is required after `pyprep` installed.**



### 2.1 Prepare Configuration
The **default training configuration** in `./configs` folder in current folder should work.
For further development, this section provide the details of the configuration file
A configuration file may seems like:
``` json
{
    "indexes":{
	"all": [[11, 1], [11, 2], [11, 3], [11, 4], [12, 1], [12, 2], [13, 1], [13, 2], [13, 3], [13, 4], [14, 1], [15, 1], [15, 2], [15, 3], [15,4 ]]
    },
    "save_prefix":"[prep-asr]",
    "preprocessings":{
        "pipeline":[
            ["drop_channels", {"ch_names": ["ECG EKG", "Manual"], "on_missing": "warn"}], 
            ["prep", {
            "montage": "standard_1020",
            "prep_params":{
                "ref_chs": "eeg",
                "reref_chs": "eeg",
                "line_freqs":[]
            },
            "reference_args": {
                "correlation_secs": 1.0, 
                "correlation_threshold": 0.4, 
                "frac_bad": 0.01
            }
            }],
        ["asr", {"cutoff": 30}], ["average_reference", {}], ["min_max_nor", {}]],
        "post_merge_pipeline": [["average_reference", {}], ["min_max_nor", {}]]
    },
    "extraction_process":{
        "dataset_base_path": "../data",
        "database_name": "epileptic_eeg_dataset",
        "number-microstate-least": 4,
        "number-microstate-most": 4,
        "kmeans-iterations": 200,
        "stop-threshold": 0.025,
        "store-microstates-n4": true,
        "store-preprocessed": true,
        "store-segmentation": true,
        "store-path": "../data/microstates/epileptic_eeg_dataset/",
        "preprocessed_file_prefix": "[preprocessed_prep_asr]"
    }
}
```
+ `indexes` block define a set of `string: list[List[int]]` items. In which, the `string` define the name of a training task, and the list value indicates which records should be utilize in training. Each record is identify by a 2-element list, where the first element is the person id, and the second element is a record id of that person.  For `[x, y]`, it correspond to the `p{x}_Record{y}.edf` record file.
+ `save_prefix` specify what prefix should be attached to the output file names.
+ `preprocessings` specify a list of preprocessing methods. Currently, only serveral preprocessing methods are supported: `drop_channels`, `prep` and `asr`.
+ `extraction_process` specify the detailed training setting. It will respectively train from `number-microstate-least` microstates to `number-microstate-most` microstates. And stop at a certain $k$, $number-microstate-least\le  k \le number-microstate-most$, if the evaluation metrics (i.e., global explain variation) change less than `stop-threshold`. And one that exhibit the best GEV will be saved.
When `store-microstates-n4` is enabled, even though the `4`-component microstates is not exhibit the best GEV, it will still be stored.



## 2.2 Run the script

```bash
$ cd <path-to-2.microstate_training>
$ python microstate_extraction.py
or
$ python microstate_extraction.py --database_index_configuration ./configs/config-all-person-microstate.json
```
The script utilize default configuration file `./configs/config-all-person-microstate.json`
It can use `python microstate_extraction.py --database_index_configuration <path-to-config-file>`  to run with a specific configuration file.

