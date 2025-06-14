exp1-npy-to-plain-text:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp1-distinguishability ./convert_npy_to_plain_text.sh

exp1-no-normal-only-nogfp-npy-to-plain-text:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp1-distinguishability-nogfp-no-normal-only ./convert_npy_to_plain_text.sh

exp2-npy-to-plain-text:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp2-nogfp-no-normal-only-general-ms-pcfg ; \
	REMAP="--remap [0,3,1,2]" ; \
	if [ -z "$BASE_PATH" ]; then \
		echo "Error: BASE_PATH is not set." ; \
		exit 1 ; \
	fi ; \
	python sentence_plain_text_encoder.py --file_path "$$BASE_PATH/seizure_integrated_all_d2_s4.npy" --output_file_path "$$BASE_PATH/seizure_integrated_all_d2_s4.txt" $$REMAP ; \
	python sentence_plain_text_encoder.py --file_path "$$BASE_PATH/normal_integrated_all_d2_s4.npy" --output_file_path "$$BASE_PATH/normal_integrated_all_d2_s4.txt" $$REMAP ; \
	python sentence_plain_text_encoder.py --file_path "$$BASE_PATH/pre-epileptic_integrated_all_d2_s4.npy" --output_file_path "$$BASE_PATH/pre-epileptic_integrated_all_d2_s4.txt" $$REMAP ; \
	python sentence_plain_text_encoder.py --file_path "$$BASE_PATH/seizure_integrated_all_d2_s4_repetition.npy" --output_file_path "$$BASE_PATH/seizure_integrated_all_d2_s4_repetition.txt" ; \
	python sentence_plain_text_encoder.py --file_path "$$BASE_PATH/normal_integrated_all_d2_s4_repetition.npy" --output_file_path "$$BASE_PATH/normal_integrated_all_d2_s4_repetition.txt" ; \
	python sentence_plain_text_encoder.py --file_path "$$BASE_PATH/pre-epileptic_integrated_all_d2_s4_repetition.npy" --output_file_path "$$BASE_PATH/pre-epileptic_integrated_all_d2_s4_repetition.txt" ; \


exp3-npy-to-plain-text:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp3-nogfp-no-normal-only-general-ms-pcfg ./convert_npy_to_plain_text.sh

exp3-phase-space-reconstruction:
	BINARY_PATH="../lib/pcfg-cky-inside-outside/bin/phase_convert"; \
        CONFIGURATION_BASE_PATH="./phase_space_reconstruction_configs/exp3-nogfp-no-normal-only"; \
        $$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_normal.yaml; \
        $$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_preseizure.yaml; \
        $$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_seizure.yaml; \

exp2-phase-space-reconstruction:
	BINARY_PATH="../lib/pcfg-cky-inside-outside/bin/phase_convert"; \
        CONFIGURATION_BASE_PATH="./phase_space_reconstruction_configs/exp2-nogfp-no-normal-only"; \
        $$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_normal.yaml; \
        $$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_preseizure.yaml; \
        $$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_seizure.yaml; \

exp1-phase-space-reconstruction:
	BINARY_PATH="../lib/pcfg-cky-inside-outside/bin/phase_convert"; \
	CONFIGURATION_BASE_PATH="./phase_space_reconstruction_configs/exp1-distinguishability"; \
	$$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_normal.yaml; \
	$$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_preseizure.yaml; \
	$$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_seizure.yaml; \

exp1-phase-space-reconstruction-no-normal-only-nogfp:
	BINARY_PATH="../lib/pcfg-cky-inside-outside/bin/phase_convert"; \
        CONFIGURATION_BASE_PATH="./phase_space_reconstruction_configs/exp1-distinguishability-no-normal-only-nogfp"; \
        $$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_normal.yaml; \
        $$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_preseizure.yaml; \
        $$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_seizure.yaml; \


exp1-convert-length:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp1-distinguishability ./convert_state_length.sh

exp1-convert-length-no-normal-only-nogfp:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp1-distinguishability-nogfp-no-normal-only  ./convert_state_length.sh

exp2-convert-length:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp2-nogfp-no-normal-only-general-ms-pcfg ./convert_state_length.sh


exp3-convert-length:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp3-nogfp-no-normal-only-general-ms-pcfg ./convert_state_length.sh


exp1-split-data-all-zones:
	SENTENCE_FOLDER_NAME=exp1-distinguishability-nogfp-no-normal-only ; \
	sh 3_split_dataset_all_area.sh ; \



evaluate_grammars:
	GRAMMAR_FOLDER=./logs ; \
        SENTENCE_PATH=