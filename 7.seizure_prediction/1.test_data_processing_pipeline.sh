
PRE_EPILEPTIC_LENGTH=120 # pre-epileptic area length, in secs.
# from retained record to a microstate sequence npy file.
DATASET_SPLITTING_AND_SEGMENTATION_PY_SCRIPT=../3.phase_space_reconstruction/dataset_splitting_and_segmentation.py
python $DATASET_SPLITTING_AND_SEGMENTATION_PY_SCRIPT -cf ../3.phase_space_reconstruction/configs/epilepsy_all_person_intergrated-test.json -pz $PRE_EPILEPTIC_LENGTH

# npy to plain text
bash convert_npy_to_plain_text_for_test.sh

# convert state length and write to plain text file.
bash convert_state_length.sh

# phase space reconstruction
bash phase_space_reconstruction.sh
