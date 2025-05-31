exp1-npy-to-plain-text:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp1-distinguishability ./convert_npy_to_plain_text.sh

exp1-phase-space-reconstruction:
	BINARY_PATH="../lib/pcfg-cky-inside-outside/bin/phase_convert"; \
	CONFIGURATION_BASE_PATH="./phase_space_reconstruction_configs/exp1-distinguishability"; \
	$$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_normal.yaml; \
	$$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_preseizure.yaml; \
	$$BINARY_PATH $$CONFIGURATION_BASE_PATH/config_sentence_encoding_seizure.yaml; \

exp1-convert-length:
	BASE_PATH=../data/recurrence_sentences/epileptic_eeg_dataset/exp1-distinguishability ./convert_state_length.sh
