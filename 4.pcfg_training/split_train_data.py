import random
import argparse
from pathlib import Path

def split_dataset(input_file, train_percent, seed=None):
    # Read all lines
    with open(input_file, 'r') as f:
        lines = f.readlines()[:-1]

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(lines)

    # Compute split index
    split_index = int(len(lines) * train_percent / 100)

    # Split
    train_lines = lines[:split_index]
    retained_lines = lines[split_index:]

    # Create output file paths
    
    input_path = Path(input_file)
    output_dir = input_path.parent
    prefix = input_path.stem  # "xxx" part of xxx.txt
    train_file = output_dir / f"{prefix}_train.txt"
    retained_file = output_dir / f"{prefix}_retained.txt"
    
    # Write to files
    with open(train_file, 'w') as f:
        f.writelines(train_lines)

    with open(retained_file, 'w') as f:
        f.writelines(retained_lines)

    print(f"Wrote {len(train_lines)} lines to {train_file}")
    print(f"Wrote {len(retained_lines)} lines to {retained_file}")

# Run from CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a dataset txt file into train and retained parts.")
    parser.add_argument("input_file", help="Path to the input txt file")
    parser.add_argument("--percent", type=float, default=80, help="Percentage of data to use for training (default: 80)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    args = parser.parse_args()

    split_dataset(args.input_file, args.percent, args.seed)
