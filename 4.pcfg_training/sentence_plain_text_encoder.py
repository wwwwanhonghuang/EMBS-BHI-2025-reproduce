import argparse
import numpy as np
from functools import reduce
from tqdm import tqdm
import os
import ast
def parse_list(arg):
    print(f"Received arg: {arg}")  # debug line

    try:
        # Safely evaluate the string as a Python literal
        return ast.literal_eval(arg)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid list: {arg}")


parser = argparse.ArgumentParser()
recurrence_sentence_base_path = '../data/recurrence_sentences/epileptic_eeg_dataset/'

parser.add_argument("--file_path", default=os.path.join(recurrence_sentence_base_path, "./seizure_integrated_all_d2_s4.npy"))
parser.add_argument("--output_file_path", default=os.path.join(recurrence_sentence_base_path, "./seizure_integrated_all_d2_s4.txt"))
parser.add_argument('--remap', type=parse_list, help='List of integers', default=None)

args = parser.parse_args()
remap = args.remap
data = np.load(args.file_path, allow_pickle=True)
f = open(args.output_file_path, "w")

for item in tqdm(data):
    item = np.vectorize(lambda x: x if remap is None else remap[x])(np.array(item)) + 1
    line = reduce(lambda x, y: f"{x} {y}", item)
    f.write(line)
    f.write("\n")

f.close()
