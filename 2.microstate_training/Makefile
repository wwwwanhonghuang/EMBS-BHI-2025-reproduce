all: train_microstate_dev

# Default values (can be overridden by command-line)
FORCE_REPREPROCESSING ?= 0
PREEPILEPTIC_LENGTH_SEC ?= 120
CONFIGURATION_FILE ?= ./configs/config-all-person-microstate-dev.json

train-microstate-dev:
	@echo "Running microstate training (Dev dataset) with:"
	@echo "  FORCE_REPREPROCESSING=$(FORCE_REPREPROCESSING)"
	@echo "  PREEPILEPTIC_LENGTH_SEC=$(PREEPILEPTIC_LENGTH_SEC)"
	@echo "  CONFIGURATION_FILE=$(CONFIGURATION_FILE)"
	@FLAGS="--pre_epileptic_zone $(PREEPILEPTIC_LENGTH_SEC) --database_index_configuration $(CONFIGURATION_FILE) --use_gfp"; \
	if [ "$(FORCE_REPREPROCESSING)" -eq 1 ]; then \
		FLAGS="$$FLAGS --force_repreprocessing"; \
	fi; \
	python microstate_extraction.py $$FLAGS --no_normal_only

dev-signal-to-microstate:
	@MICROSTATE_FOLDER=../data/microstates/epileptic_eeg_dataset/; \
	echo "Using MICROSTATE_FOLDER=$$MICROSTATE_FOLDER"; \
	RECORD_FILE_PATH=$$(find $$MICROSTATE_FOLDER -name '[preprocessed_prep_asr]pall.edf' | head -n 1); \
	SEQUENCE_SAVE_PATH=./data/sequence_exp1_dev.npy; \
	echo "Using RECORD_FILE_PATH=$$RECORD_FILE_PATH"; \
	microstate_file=$$(find $$MICROSTATE_FOLDER -name '[test]person_all_gev_*.npy' | head -n 1); \
	if [ -z "$$microstate_file" ]; then \
		echo "❌ No matching microstate file found in $$MICROSTATE_FOLDER"; exit 1; fi; \
	if [ -z "$$RECORD_FILE_PATH" ]; then \
		echo "❌ No record file found in $$MICROSTATE_FOLDER"; exit 1; fi; \
	echo "Using microstate file: $$microstate_file"; \
	python signal_to_microstate_sequence.py \
		--microstate_file_path $$microstate_file \
		--record_file_path $$RECORD_FILE_PATH \
		--save_path $$SEQUENCE_SAVE_PATH \
           
