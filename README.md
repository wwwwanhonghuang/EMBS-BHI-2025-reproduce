[!Warn] Repository Under Developing.

## 1. Environment Preparation

It may install a virtual environment for better manage dependencies.

``` bash
$ conda create -n embc2025 python=3.9
$ cd <path-to-repository-root>
$ conda activate embc2025 requirement.txt
```

> Important, The experiments were under CPU version of pytorch. You may replace the pytorch version you like according to the instructions in https://pytorch.org/get-started/locally/.



## 2. Results Reproduction

### 2.1 <span style="color: red">**Step 1:** </span>**Data Preparation**

+ **Link**: [1.dataset_preparation](./1.dataset_preparation)

+ **Contract**
  + **Input**: nothing
  + **Output:** The raw EEG dataset 



### 2.2 <span style="color: red">**Step 2:** </span>**Microstate Training**

+ **Link**: [2.microstate_training](./2.microstate_training)

+ **Contract**

  + **Input**: The raw EEG dataset 

  + **Output**
    + Microstate Topomaps
    + Microstate Sequences



### 2.3 <span style="color: red">**Step 3:** </span>**Phase Space Reconstruction** 

+ **Link**: [3.phase_space_reconstruction](./3.phase_space_reconstruction)

+ **Contract**

  + **Input**
    + Microstate Topomaps
    + Microstate Sequences

  + **Output**
    + Time-delay method reconstructed brain state sequences



### 2.4 <span style="color: red">**Step 4:** </span>**PCFG Training**

+ **Link**: [4.pcfg_training](./4.pcfg_training)

+ **Input**
  + Possibility Context-Free Grammar File
  + Time-delay method reconstructed brain state sequences

+ **Output**
  + Possibility Context-Free Grammar File



### 2.5 <span style="color: red">**Step 5:** </span>**Syntax Analysis**

+ **Link**: [5.syntax_analysis](./5.syntax_analysis)

+ **Contract**

  + **Input**
    + Possibility Context-Free Grammar File
    + Time-delay method reconstructed brain state sequences

  + **Output**
    + Syntax trees
    + Statistical features of syntax trees



### 2.6 <span style="color: red">**Step 6** </span>**Informatics Features and Tree-structural Features Evaluation**

+ **Link**: [6.evaluation](./6.evaluation)

+ **Contract**

  + **Input**
    + Possibility Context-Free Grammar File
    + Time-delay method reconstructed brain state sequences

  + **Output**
    + Syntax trees
    + Statistical features of syntax trees



### 2.7 <span style="color: red">**Step 7** </span>**Seizure Prediction**

+ **Link**: [7.seizure_prediction](./7.seizure_prediction)

+ **Contract**

  + **Input**
    + Syntax trees

  + **Output**
    + Seizure Prediction Results

