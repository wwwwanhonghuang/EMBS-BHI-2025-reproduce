import random
import argparse
from sklearn.model_selection import train_test_split
import os

def read_data(input_file_path):
    with open(input_file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
def out_data(data, output_file_path):
    with open(output_file_path, 'w') as f:
        f.write("\n".join(data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets.")
    parser.add_argument("--normal_sentence_file")
    parser.add_argument("--seizure_sentence_file")
    parser.add_argument("--preepileptic_sentence_file")
    parser.add_argument("--preepileptic_repetition_file", required=True)
    parser.add_argument("--seizure_repetition_file", required=True)
    parser.add_argument("--normal_repetition_file", required=True)
    parser.add_argument("--output_base_path", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--percent", type=int, default=80)

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
    
    # Read all data
    dataset = {
        'normal': read_data(args.normal_sentence_file),
        'seizure': read_data(args.seizure_sentence_file),
        'preepileptic': read_data(args.preepileptic_sentence_file),
        'normal_repetition': read_data(args.normal_repetition_file),
        'seizure_repetition': read_data(args.seizure_repetition_file),
        'preepileptic_repetition': read_data(args.preepileptic_repetition_file)
    }

    def split_indexes(data, seed=None):
        # First split into train+val (80%) and test (20%)
        train_indexes, tmp_indexes = train_test_split(
            range(len(data)), 
            test_size=0.2,
            random_state=seed
        )
        # Then split train+val into train (7/8) and val (1/8)
        validation_indexes, test_indexes = train_test_split(
            tmp_indexes,
            test_size=0.5,  # 10% of original / 80% remaining = 12.5%
            random_state=seed
        )
        return train_indexes, validation_indexes, test_indexes
    
    # Split each category
    splits = {}
    for category in ['normal', 'seizure', 'preepileptic']:
        train_idx, val_idx, test_idx = split_indexes(dataset[category], args.seed)
        splits[category] = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }

    # Prepare data splits
    train_data = {
        'sentences': (
            [dataset['normal'][i] for i in splits['normal']['train_idx']] +
            [dataset['seizure'][i] for i in splits['seizure']['train_idx']] +
            [dataset['preepileptic'][i] for i in splits['preepileptic']['train_idx']]
        ),
        'repetitions': (
            [dataset['normal_repetition'][i] for i in splits['normal']['train_idx']] +
            [dataset['seizure_repetition'][i] for i in splits['seizure']['train_idx']] +
            [dataset['preepileptic_repetition'][i] for i in splits['preepileptic']['train_idx']]
        )
    }

    val_data = {
        'sentences': (
            [dataset['normal'][i] for i in splits['normal']['val_idx']] +
            [dataset['seizure'][i] for i in splits['seizure']['val_idx']] +
            [dataset['preepileptic'][i] for i in splits['preepileptic']['val_idx']]
        ),
        'repetitions': (
            [dataset['normal_repetition'][i] for i in splits['normal']['val_idx']] +
            [dataset['seizure_repetition'][i] for i in splits['seizure']['val_idx']] +
            [dataset['preepileptic_repetition'][i] for i in splits['preepileptic']['val_idx']]
        )
    }

    test_data = {
        'sentences': {
            'normal': [dataset['normal'][i] for i in splits['normal']['test_idx']],
            'seizure': [dataset['seizure'][i] for i in splits['seizure']['test_idx']],
            'preepileptic': [dataset['preepileptic'][i] for i in splits['preepileptic']['test_idx']]
        },
        'repetitions': {
            'normal': [dataset['normal_repetition'][i] for i in splits['normal']['test_idx']],
            'seizure': [dataset['seizure_repetition'][i] for i in splits['seizure']['test_idx']],
            'preepileptic': [dataset['preepileptic_repetition'][i] for i in splits['preepileptic']['test_idx']]
        }
    }

    # Create output directory
    os.makedirs(args.output_base_path, exist_ok=True)

    # Write output files
    out_data(train_data['sentences'], os.path.join(args.output_base_path, "train_sentences.txt"))
    out_data(train_data['repetitions'], os.path.join(args.output_base_path, "train_repetitions.txt"))
    
    out_data(val_data['sentences'], os.path.join(args.output_base_path, "val_sentences.txt"))
    out_data(val_data['repetitions'], os.path.join(args.output_base_path, "val_repetitions.txt"))

    for category in ['normal', 'seizure', 'preepileptic']:
        out_data(
            test_data['sentences'][category],
            os.path.join(args.output_base_path, f"{category}_test_sentences.txt")
        )
        out_data(
            test_data['repetitions'][category],
            os.path.join(args.output_base_path, f"{category}_test_repetitions.txt")
        )