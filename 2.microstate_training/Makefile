all: train_microstate_dev

# Default values (can be overridden by command-line)
FORCE_REPREPROCESSING ?= 0
PREEPILEPTIC_LENGTH_SEC ?= 120

train-microstate-dev-gfp-normal-only:
	@echo "Running microstate training (Dev dataset) with:"
	@echo "  FORCE_REPREPROCESSING=$(FORCE_REPREPROCESSING)"
	@echo "  PREEPILEPTIC_LENGTH_SEC=$(PREEPILEPTIC_LENGTH_SEC)"
	@echo "  CONFIGURATION_FILE=$(CONFIGURATION_FILE)"
	@FLAGS="--pre_epileptic_zone $(PREEPILEPTIC_LENGTH_SEC) --database_index_configuration $(CONFIGURATION_FILE) --use_gfp"; \
	if [ "$(FORCE_REPREPROCESSING)" -eq 1 ]; then \
		FLAGS="$$FLAGS --force_repreprocessing"; \
	fi; \
	python microstate_extraction.py $$FLAGS

train-microstate-dev-nogfp-normal-only:
	CONFIGURATION_FILE=./configs/config-all-person-microstate-dev.json	;\
	FLAGS="--pre_epileptic_zone $(PREEPILEPTIC_LENGTH_SEC) --database_index_configuration $$CONFIGURATION_FILE"; \
	echo "Running microstate training (Dev dataset, non-gfp, normal-only) with:" ;\
	echo "  FORCE_REPREPROCESSING=$(FORCE_REPREPROCESSING)" ;\
	echo "  PREEPILEPTIC_LENGTH_SEC=$(PREEPILEPTIC_LENGTH_SEC)" ;\
	echo "  CONFIGURATION_FILE=$$CONFIGURATION_FILE" ;\
	if [ "$(FORCE_REPREPROCESSING)" -eq 1 ]; then \
		FLAGS="$$FLAGS --force_repreprocessing"; \
	fi; \
	python microstate_extraction.py $$FLAGS

dev-signal-to-microstate:		
	@MICROSTATE_FOLDER=../data/microstates/epileptic_eeg_dataset/; \
	echo "Using MICROSTATE_FOLDER=$$MICROSTATE_FOLDER"; \
	RECORD_FILE_PATH=$$(find $$MICROSTATE_FOLDER -name '\[preprocessed_prep_asr\]pall.edf' | head -n 1); \
	SEQUENCE_SAVE_PATH=../data/microstates/sepileptic_eeg_dataset/equence_exp1_dev_normal_only_no_use_gfp_preson_all; \
	echo "Using RECORD_FILE_PATH=$$RECORD_FILE_PATH"; \
	microstate_file=$$(find $$MICROSTATE_FOLDER -name '\[\[normal-only-prep-asr\]\]person_all_states4_gev*.npy' | head -n 1); \
	if [ -z "$$microstate_file" ]; then \
		echo "❌ No matching microstate file found in $$MICROSTATE_FOLDER"; exit 1; fi; \
	if [ -z "$$RECORD_FILE_PATH" ]; then \
		echo "❌ No record file found in $$MICROSTATE_FOLDER"; exit 1; fi; \
	echo "Using microstate file: $$microstate_file"; \
	python signal_to_microstate_sequence.py \
		--microstate_file_path $$microstate_file \
		--record_file_path $$RECORD_FILE_PATH \
		--sequence_file_save_path $$SEQUENCE_SAVE_PATH.npy \
		| tee output.txt; \
	gev_val=$$(grep "gev =" output.txt | sed -E 's/.*gev = ([0-9.]+),.*/\1/'); \
	mv $${SEQUENCE_SAVE_PATH}.npy $${SEQUENCE_SAVE_PATH}_gev_$${gev_val}.npy
	rm output.txt
