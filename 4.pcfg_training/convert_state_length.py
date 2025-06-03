import argparse
import numpy as np
from functools import reduce
from tqdm import tqdm
import os 

parser = argparse.ArgumentParser()
recurrence_sentence_base_path = '../data/recurrence_sentences/epileptic_eeg_dataset/'

parser.add_argument("--file_path", default=os.path.join(recurrence_sentence_base_path, "./seizure_integrated_all_d2_s4_repetition.npy"))
parser.add_argument("--output_file_path", default=os.path.join(recurrence_sentence_base_path, "./seizure_integrated_all_d2_s4_repetition_converted.txt"))

args = parser.parse_args()
data = np.load(args.file_path, allow_pickle=True)
f = open(args.output_file_path, "w")

for item in tqdm(data):
    item = np.array(item)
    repetitions = []
    for i in range(1, len(item)):
        repetitions.append(item[i] + item[i - 1] + 2)
    line = reduce(lambda x, y: f"{x} {y}", repetitions)
    f.write(line)
    f.write("\n")

f.close()
