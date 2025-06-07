import argparse
import os
import re
import logging
import shutil
from typing import List, Set, Tuple, Optional
import numpy as np
from tqdm import tqdm
import subprocess
import sys
# Add third-party lib path (if needed)
sys.path.append("../third_parts/microstate_lib/code")
from data_utils import match_reorder_topomaps

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> List[List[int]]:
    """Loads a text file containing space-separated integers into a list of lists."""
    try:
        with open(file_path, "r") as f:
            return [[int(x) for x in line.strip().split()] for line in f]
    except (IOError, ValueError) as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def remap_microstates(
    source_microstates: np.ndarray, 
    target_microstates: np.ndarray
) -> List[int]:
    """Remaps source microstates to match target topomaps."""
    permutation = list(match_reorder_topomaps(source_microstates, target_microstates, return_attribution_only=True))
    return permutation


def apply_time_delay(sequence: List[int], delay: int = 2) -> List[int]:
    """Applies time-delay embedding to a sequence of microstates."""
    time_delayed = []
    for i in range(delay - 1, len(sequence)):
        encoded = 0
        for j in range(delay):
            encoded *= 4
            encoded += int(sequence[i - j] - 1)
        time_delayed.append(encoded + 1)
    return time_delayed


def main():
    parser = argparse.ArgumentParser(description="Process EEG microstate data for syntax analysis.")
    parser.add_argument("--l", type=int, required=True, help="Parameter L (e.g., number of states).")
    parser.add_argument("--md5", type=str, required=True, help="MD5 hash identifier for the dataset.")
    parser.add_argument("--dataset_base_path", type=str, default="./data/adhd_microstate_dataset_samples", help="Base path for input data.")
    parser.add_argument("--syntax_analysis_binary", type=str, default="../lib/pcfg-cky-inside-outside/bin/syntax_analysis", help="Path to syntax analysis binary.")
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.dataset_base_path):
        raise FileNotFoundError(f"Dataset path {args.dataset_base_path} does not exist.")

    # Load origin dataset
    origin_dataset_dir = os.path.join(args.dataset_base_path, str(args.l))
    origin_files = [
        f for f in os.listdir(origin_dataset_dir) 
        if re.match(f"^{args.md5}_gev_[01]\.[0-9]+.*\.npz$", f)
    ]
    if len(origin_files) != 1:
        raise ValueError(f"Expected 1 matching file, found {len(origin_files)}: {origin_files}")

    origin_file_path = os.path.join(origin_dataset_dir, origin_files[0])
    try:
        data = np.load(origin_file_path)
    except Exception as e:
        logger.error(f"Failed to load {origin_file_path}: {e}")
        raise

    microstates = data["microstates"]
    sample_ids = set(int(sid) for sid in str(data["sample_id"]).split("_"))
    logger.info(f"Sample IDs: {sample_ids}, Count: {len(sample_ids)}")
    eval_sample_ids = set(range(0, 60)) - sample_ids
    
    # Prepare syntax analysis directories
    syntax_analysis_path = os.path.join(args.dataset_base_path, "..", "syntax_analysis_data", str(args.l), args.md5)
    os.makedirs(syntax_analysis_path, exist_ok=True)

    # Copy control files (segmentation & repetitions)
    sentence_converted_path = os.path.join(args.dataset_base_path, "..", "sentence_converted", str(args.l))
    
    controls_dir = os.path.join(syntax_analysis_path, "controls")
    os.makedirs(controls_dir, exist_ok=True)
    
    for eval_id in eval_sample_ids:
        segmentation_file = next(
        (f for f in os.listdir(sentence_converted_path) 
            if re.match(f"^{args.md5}_gev_[01]\.[0-9]+_seg_text\.txt$", f)),
            None
        )
        repetition_file = next(
            (f for f in os.listdir(sentence_converted_path) 
            if re.match(f"^{args.md5}_gev_[01]\.[0-9]+_repetitions\.txt$", f)),
            None
        )

        if not segmentation_file or not repetition_file:
            raise FileNotFoundError("Missing segmentation or repetition file.")

        shutil.copy(
            os.path.join(sentence_converted_path, segmentation_file),
            os.path.join(controls_dir, f"C{eval_id}_seg.txt")
        )
        shutil.copy(
            os.path.join(sentence_converted_path, repetition_file),
            os.path.join(controls_dir, f"C{eval_id}_repetitions.txt")
        )

    # Process ADHD samples
    adhd_dataset_path = os.path.join(args.dataset_base_path, "..", "syntax_analysis_data", "adhd")
    adhd_samples = [f for f in os.listdir(adhd_dataset_path) if f.endswith(".npz")]
    logger.info(f"Found {len(adhd_samples)} ADHD samples.")

    for sample_file in tqdm(adhd_samples, desc="Processing ADHD samples"):
        sample_id = sample_file.split(".")[0].lstrip("A")
        adhd_data_path = os.path.join(adhd_dataset_path, sample_file)
        
        try:
            adhd_data = np.load(adhd_data_path)
            adhd_microstates = adhd_data["microstates"]
        except Exception as e:
            logger.error(f"Failed to load {adhd_data_path}: {e}")
            continue

        # Remap microstates
        permutation = remap_microstates(adhd_microstates, microstates)
        
        # Apply time delay
        adhd_seg_data = load_data(os.path.join(adhd_dataset_path, "..", "plain_text_segments", f"A{sample_id}_seg.txt"))
        remapped_data = [
            [permutation[x - 1] + 1 for x in seg_line]
            for seg_line in adhd_seg_data
        ]
        time_delayed_data = [apply_time_delay(seg_line) for seg_line in remapped_data]

        # Save remapped data
        os.makedirs(os.path.join(syntax_analysis_path, "adhd"), exist_ok=True)
        output_path = os.path.join(syntax_analysis_path, "adhd", f"A{sample_id}_seg.txt")

        with open(output_path, "w") as f:
            for line in time_delayed_data:
                f.write(" ".join(map(str, line)) + "\n")

        repetition_file = os.path.join(adhd_dataset_path, "..", "sentence_converted", f'A{sample_id}_repetitions.txt')
        if not os.path.exists(repetition_file):
            raise FileNotFoundError(f"Missing repetition file {repetition_file}.")
        shutil.copy(
            repetition_file,
            os.path.join(syntax_analysis_path, "adhd", f"A{sample_id}_repetitions.txt")
        )
        
    
    logger.info("Processing complete.")
    


if __name__ == "__main__":
    main()