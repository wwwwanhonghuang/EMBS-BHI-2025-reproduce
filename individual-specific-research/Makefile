# Makefile for individual project setup

# Common Configuration

# JSON template
define MICROSTATE_JSON_TEMPLATE
{
    "indexes":{
        "all": $(INDEXES)
    },
    "sampling_freq": 500,
    "save_prefix":"$(SAVE_PREFIX)",
    "preprocessings":{
        "pipeline":[
            ["drop_channels", {"ch_names": ["ECG EKG", "Manual"], "on_missing": "warn"}], 
            ["prep", {
            "montage": "standard_1020",
            "prep_params":{
                "ref_chs": "eeg",
                "reref_chs": "eeg",
                "line_freqs":[]
            },
            "reference_args": {
                "correlation_secs": 1.0, 
                "correlation_threshold": 0.4, 
                "frac_bad": 0.01
            }
            }],
        ["asr", {"cutoff": 30}], ["average_reference", {}], ["min_max_nor", {}]],
        "post_merge_pipeline": [["average_reference", {}], ["min_max_nor", {}]]
    },
    "extraction_process":{
        "load_preprocessed": true,
        "dataset_base_path": "$(DATASET_BASE_PATH)",
        "database_name": "epileptic_eeg_dataset",
        "number-microstate-least": 4,
        "number-microstate-most": 4,
        "kmeans-iterations": 50,
        "stop-threshold": 0.025,
        "store-microstates-n4": true,
        "store-preprocessed": true,
        "store-segmentation": true,
        "store-path": "$(STORE_PATH)",
        "preprocessed_file_prefix": "$(PREPROCESSED_FILE_PREFIX)"
    }
}
endef

export MICROSTATE_JSON_TEMPLATE

.PHONY: generate_microstate_dev generate_microstate_eval

NORMAL_ONLY_DESC=


MICROSTATE_DEV_SAVE_PREFIX = [dev$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))]
MICROSTATE_DEV_DATASET_BASE_PATH = ../data/dataset
MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX = [preprocessed]

MICROSTATE_EVAL_SAVE_PREFIX = [eval$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))]
MICROSTATE_EVAL_DATASET_BASE_PATH = ../data/dataset
MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX = [eval_preprocessed]

make_individual_project_11:
	@DEV_INDEXES="[[11, 1], [11, 2], [11, 3]]" ; \
	EVAL_INDEXES="[[11, 4]]" ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=11 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

make_individual_project_12:
	@DEV_INDEXES="[[12, 1]]" ; \
	EVAL_INDEXES="[[12, 2]]" ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=12 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

make_individual_project_13:
	@DEV_INDEXES="[[13, 1], [13, 2], [13, 3]]" ; \
	EVAL_INDEXES="[[13, 4]]" ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=13 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

make_individual_project_14:
	@DEV_INDEXES="[[14, 1]]" ; \
	EVAL_INDEXES="[[14, 2]]" ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=14 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

make_individual_project_15:
	@DEV_INDEXES="[[15, 1], [15, 2], [15, 3]]" ; \
	EVAL_INDEXES="[[15, 4]]" ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=15 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

_make_individual_project:
	echo "Creating project structure for person $(PERSON_ID)" ; \
	mkdir -p p$(PERSON_ID)/microstates/configurations ; \
	touch p$(PERSON_ID)/Makefile ; \
	echo "Project setup complete for person $(PERSON_ID)" ; \
	$(MAKE) generate_microstate_dev \
		INDEXES="$(DEV_INDEXES)" \
		PERSON_ID=$(PERSON_ID) \
		MICROSTATE_DEV_STORE_PATH="./p$(PERSON_ID)/microstates/" \
		NORMAL_ONLY_DESC=;
	$(MAKE) generate_microstate_eval \
		INDEXES="$(EVAL_INDEXES)" \
		PERSON_ID=$(PERSON_ID) \
		MICROSTATE_EVAL_STORE_PATH="./p$(PERSON_ID)/microstates/" \
		NORMAL_ONLY_DESC=;
	$(MAKE) generate_microstate_dev \
		INDEXES="$(DEV_INDEXES)" \
		PERSON_ID=$(PERSON_ID) \
		MICROSTATE_DEV_STORE_PATH="./p$(PERSON_ID)/microstates/" \
		NORMAL_ONLY_DESC=normal_only;
	$(MAKE) generate_microstate_eval \
		INDEXES="$(EVAL_INDEXES)" \
		PERSON_ID=$(PERSON_ID) \
		MICROSTATE_EVAL_STORE_PATH="./p$(PERSON_ID)/microstates/" \
		NORMAL_ONLY_DESC=normal_only;
		
generate_microstate_dev:
	@$(MAKE) _generate_microstate_training_configuration_json \
		TYPE=dev \
		INDEXES="$(INDEXES)" \
		SAVE_PREFIX="$(MICROSTATE_DEV_SAVE_PREFIX)" \
		DATASET_BASE_PATH="$(MICROSTATE_DEV_DATASET_BASE_PATH)" \
		STORE_PATH="$(MICROSTATE_DEV_STORE_PATH)" \
		PREPROCESSED_FILE_PREFIX="$(MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX)" \
		OUTPUT_FILE="p$(PERSON_ID)/microstates/configurations/microstate_dev$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC)).json"

generate_microstate_eval:
	@$(MAKE) _generate_microstate_training_configuration_json \
		TYPE=eval \
		INDEXES="$(INDEXES)" \
		SAVE_PREFIX="$(MICROSTATE_EVAL_SAVE_PREFIX)" \
		DATASET_BASE_PATH="$(MICROSTATE_EVAL_DATASET_BASE_PATH)" \
		STORE_PATH="$(MICROSTATE_EVAL_STORE_PATH)" \
		PREPROCESSED_FILE_PREFIX="$(MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX)" \
		OUTPUT_FILE="p$(PERSON_ID)/microstates/configurations/microstate_eval$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC)).json"

_generate_microstate_training_configuration_json:
	@echo "$$MICROSTATE_JSON_TEMPLATE" > $(OUTPUT_FILE)
	@echo "Generated $(TYPE) configuration at $(OUTPUT_FILE)"