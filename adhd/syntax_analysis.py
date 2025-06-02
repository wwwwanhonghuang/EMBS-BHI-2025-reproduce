import argparse
import os
import re
import numpy as np
import subprocess  # Added for running external commands

parser = argparse.ArgumentParser()

parser.add_argument("--l", type=int)
parser.add_argument("--md5", type=str)
parser.add_argument("--dataset_base_path", type=str, default="./data/adhd_microstate_dataset_samples")
parser.add_argument("--syntax_analysis_binary", type=str, default="../lib/pcfg-cky-inside-outside/bin/syntax_analysis")  # Fixed typo: add_argument
args = parser.parse_args()

l = args.l
md5 = args.md5
dataset_base_path = args.dataset_base_path
trained_grammar_base_path = os.path.join(dataset_base_path, "..", "trained_grammars")

trained_outcome_base_path = os.path.join(trained_grammar_base_path, str(l), md5)

# Directory to search
# Regex pattern (e.g., files starting with 'data_' and ending in '.txt')
pattern = re.compile(r"^epoch_.*\.likelihood$")

syntax_analysis_binary = args.syntax_analysis_binary

# List matching files
matching_files = [f for f in os.listdir(trained_outcome_base_path) if pattern.match(f)]
best_epoch = -1
best_likelihood = -np.inf
for matching_file in matching_files:
    match = re.match(r'epoch_([0-9]+)\.likelihood', matching_file)
    epoch_id = int(match.group(1))
    
    with open(os.path.join(trained_outcome_base_path, matching_file)) as f:
        likelihood = float(f.read())
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_epoch = epoch_id

print(f'Best likelihood = {best_likelihood}, in epoch {best_epoch}')

best_grammar_file = os.path.join(trained_outcome_base_path, f"log_epoch_id_{best_epoch}.pcfg")

print(f"Best grammar file = {best_grammar_file}")

with open(best_grammar_file, 'r') as f:
    grammar_ctx = f.read()
    grammar_ctx = re.sub(r"\[[0-9]+\]\s*", "", grammar_ctx)
    grammar_ctx = re.sub(r"\s*->\s*", "->", grammar_ctx)
    
    out_best_grammar_file = os.path.join(trained_outcome_base_path, "grammar_best.pcfg")
    with open(out_best_grammar_file, 'w') as out_best_grammar_file_io:
        out_best_grammar_file_io.write(grammar_ctx)
        
print(f"PCFG file generated.")
origin_dataset_files = [f for f in os.listdir(os.path.join(dataset_base_path, str(l))) if re.compile(f"^{md5}_gev_[01]\.[0-9]+.*\.npz$").match(f)]
assert len(origin_dataset_files) == 1, f"Expect exactly 1 origin_dataset_file, found {origin_dataset_files}" 
origin_dataset_file_path = os.path.join(dataset_base_path, str(l), origin_dataset_files[0])
data = np.load(origin_dataset_file_path)
full_ids = set(range(0, 61))
sample_ids = set([int(sid) for sid in str(data['sample_id']).split("_")])
print(f"sample_ids = {sample_ids}, len = {len(sample_ids)}")
eval_sample_ids = full_ids - sample_ids
print(f"samples used for evaluation = {eval_sample_ids}, len = {len(eval_sample_ids)}")

syntax_analysis_data_path = os.path.join(dataset_base_path, "..", "syntax_analysis_data", md5)
print(f'syntax_analysis_data_path = {syntax_analysis_data_path}')
os.makedirs(syntax_analysis_data_path, exist_ok=True)

for eval_sample_id in eval_sample_ids:
    os.makedirs(os.path.join(syntax_analysis_data_path, str(eval_sample_id)), exist_ok=True)

sentence_converted_path = os.path.join(dataset_base_path, "..", "sentence_converted", str(l))

segmentation_files = [f for f in os.listdir(sentence_converted_path) if re.compile(f"^{md5}_gev_[01]\.[0-9]+_seg_text\.txt$").match(f)]
assert len(segmentation_files) == 1, f"Expect exactly 1 segmentation_file, found {segmentation_files}" 
segmentation_file = os.path.join(sentence_converted_path, segmentation_files[0])  # Added full path

repetition_files = [f for f in os.listdir(sentence_converted_path) if re.compile(f"^{md5}_gev_[01]\.[0-9]+_repetitions\.txt$").match(f)]
assert len(repetition_files) == 1, f"Expect exactly 1 repetition_files, found {repetition_files}" 
repetition_file = os.path.join(sentence_converted_path, repetition_files[0])  # Added full path

for eval_sample_id in eval_sample_ids:
    print(f"Processing sample id = {eval_sample_id}.")
    tree_serialization_path = os.path.join(syntax_analysis_data_path, "controls", str(eval_sample_id))  # Fixed os.join to os.path.join
    os.makedirs(tree_serialization_path, exist_ok=True)
    syntax_analysis_configuration_yaml = f"""
syntax_analysis:
    grammar_file: "{out_best_grammar_file}"  # Changed to use the generated grammar file
    input: "{segmentation_file}"
    repetition: "{repetition_file}"
    log_intervals: 1000000
    log_path: "./data/logs"
    report_path: "./data/reports"
    serialize_to_files: true
    tree_serialization_path: "{tree_serialization_path}"
    report_statistics: false
    sentence_from: -1
    sentence_to: -1
"""
    configuration_path = os.path.join(tree_serialization_path, 'configuration.tmp.yaml')
    with open(configuration_path, 'w') as configuration_file_io:
        configuration_file_io.write(syntax_analysis_configuration_yaml)
    
    # Run the syntax analysis command
    subprocess.run([syntax_analysis_binary, configuration_path], check=True)
    

for eval_sample_id in range(0, 60):
    print(f"Processing sample id = {eval_sample_id}.")
    tree_serialization_path = os.path.join(syntax_analysis_data_path, "controls", str(eval_sample_id))  # Fixed os.join to os.path.join
    os.makedirs(tree_serialization_path, exist_ok=True)
    syntax_analysis_configuration_yaml = f"""
syntax_analysis:
    grammar_file: "{out_best_grammar_file}"  # Changed to use the generated grammar file
    input: "{segmentation_file}"
    repetition: "{repetition_file}"
    log_intervals: 1000000
    log_path: "./data/logs"
    report_path: "./data/reports"
    serialize_to_files: true
    tree_serialization_path: "{tree_serialization_path}"
    report_statistics: false
    sentence_from: -1
    sentence_to: -1
"""
    configuration_path = os.path.join(tree_serialization_path, 'configuration.tmp.yaml')
    with open(configuration_path, 'w') as configuration_file_io:
        configuration_file_io.write(syntax_analysis_configuration_yaml)
    
    # Run the syntax analysis command
    subprocess.run([syntax_analysis_binary, configuration_path], check=True)