
PREEPILEPTIC_LENGTH_SEC?=120
FORCE_REPREPROCESSING?=0
define MICROSTATE_MAKEFILE_TRAINING_ITEM_TEMPLATE
train-microstate-$(EXPERIMENT_TYPE)-$(GFP_PEEK_DESC)-$(NORMAL_ONLY_DESC):
	CONFIGURATION_FILE=$$(CWD)/microstates/configurations/microstate_$(EXPERIMENT_TYPE)_$(NORMAL_ONLY_DESC)_$(GFP_PEEK_DESC).json; \\
	FLAGS="--pre_epileptic_zone $(PREEPILEPTIC_LENGTH_SEC) --database_index_configuration $$$$CONFIGURATION_FILE"; \\
	echo "Running microstate training ($(EXPERIMENT_TYPE), $(GFP_PEEK_DESC), $(NORMAL_ONLY_DESC)) with:" ;\\
	echo "  FORCE_REPREPROCESSING=$(FORCE_REPREPROCESSING)" ; \\
	echo "  PREEPILEPTIC_LENGTH_SEC=$(PREEPILEPTIC_LENGTH_SEC)" ; \\
	echo "  CONFIGURATION_FILE=$$$$CONFIGURATION_FILE" ; \\
	if [ "$$(FORCE_REPREPROCESSING)" -eq 1 ]; then \\
		FLAGS="$$$$FLAGS --force_repreprocessing"; \\
	fi; \\
	cd ../../2.microstate_training ; \
python microstate_extraction.py $$$$FLAGS \
  $(if $(filter all,$(NORMAL_ONLY_DESC)),--no_normal_only) \
  $(if $(filter gfp,$(GFP_PEEK_DESC)),--use_gfp)
  
endef

export MICROSTATE_MAKEFILE_TRAINING_ITEM_TEMPLATE

.PHONY: generate_microstate_dev generate_microstate_eval

NORMAL_ONLY_DESC=
GFP_PEEK_DESC=

MICROSTATE_DEV_SAVE_PREFIX = [dev$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC))]
MICROSTATE_DEV_DATASET_BASE_PATH = ../data/dataset
MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX = [dev_preprocessed$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC))]

MICROSTATE_EVAL_SAVE_PREFIX = [eval$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC))]
MICROSTATE_EVAL_DATASET_BASE_PATH = ../data/dataset
MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX = [eval_preprocessed$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC))]

all: make_individual_project_11 make_individual_project_12 make_individual_project_13 make_individual_project_14 make_individual_project_15

clean-all-projects:
	rm -rf p*

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
	@DEV_INDEXES="[[14, 1], [14, 2]]" ; \
	EVAL_INDEXES="[[14, 3]]" ; \
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
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			$(MAKE) generate_microstate_dev \
				INDEXES="$(DEV_INDEXES)" \
				PERSON_ID=$(PERSON_ID) \
				MICROSTATE_DEV_STORE_PATH="./p$(PERSON_ID)/microstates/" \
				NORMAL_ONLY_DESC=$$_NORMAL_ONLY_DESC \
				GFP_PEEK_DESC=$$_GFP_PEEK_DESC; \
		done; \
	done;
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			$(MAKE) generate_microstate_eval \
				INDEXES="$(EVAL_INDEXES)" \
				PERSON_ID=$(PERSON_ID) \
				MICROSTATE_EVAL_STORE_PATH="./p$(PERSON_ID)/microstates/" \
				NORMAL_ONLY_DESC=$$_NORMAL_ONLY_DESC \
				GFP_PEEK_DESC=$$_GFP_PEEK_DESC; \
		done; \
	done;
	echo 'CWD := $$(shell pwd)' > p$(PERSON_ID)/Makefile
	echo 'FORCE_REPREPROCESSING = $(FORCE_REPREPROCESSING)' >> p$(PERSON_ID)/Makefile
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			for _EXPERIMENT_TYPE in dev eval; do \
				$(MAKE) _generate_microstate_training_makefile_items \
					NORMAL_ONLY_DESC=$$_NORMAL_ONLY_DESC \
					GFP_PEEK_DESC=$$_GFP_PEEK_DESC \
					EXPERIMENT_TYPE=$$_EXPERIMENT_TYPE; \
			done; \
		done; \
	done;
	echo 'microstate-train-all:' >> p$(PERSON_ID)/Makefile
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			for _EXPERIMENT_TYPE in dev eval; do \
				echo "	\$$(MAKE) train-microstate-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}" >> p$(PERSON_ID)/Makefile; \
			done; \
		done; \
	done;


generate_microstate_dev:
	@$(MAKE) _generate_microstate_training_configuration_json \
		TYPE=dev \
		INDEXES="$(INDEXES)" \
		SAVE_PREFIX="$(MICROSTATE_DEV_SAVE_PREFIX)" \
		DATASET_BASE_PATH="$(MICROSTATE_DEV_DATASET_BASE_PATH)" \
		STORE_PATH="$(MICROSTATE_DEV_STORE_PATH)" \
		PREPROCESSED_FILE_PREFIX="$(MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX)" \
		OUTPUT_FILE="p$(PERSON_ID)/microstates/configurations/microstate_dev$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC)).json"

generate_microstate_eval:
	@$(MAKE) _generate_microstate_training_configuration_json \
		TYPE=eval \
		INDEXES="$(INDEXES)" \
		SAVE_PREFIX="$(MICROSTATE_EVAL_SAVE_PREFIX)" \
		DATASET_BASE_PATH="$(MICROSTATE_EVAL_DATASET_BASE_PATH)" \
		STORE_PATH="$(MICROSTATE_EVAL_STORE_PATH)" \
		PREPROCESSED_FILE_PREFIX="$(MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX)" \
		OUTPUT_FILE="p$(PERSON_ID)/microstates/configurations/microstate_eval$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC)).json"

_generate_microstate_training_configuration_json:
	envsubst < microstate_training_configuration_template.json.in > $(OUTPUT_FILE)
	@echo "Generated $(TYPE) configuration at $(OUTPUT_FILE)"

_generate_microstate_training_makefile_items:
	@echo "$$MICROSTATE_MAKEFILE_TRAINING_ITEM_TEMPLATE" >> p$(PERSON_ID)/Makefile
	