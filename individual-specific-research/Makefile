
PREEPILEPTIC_LENGTH_SEC?=120
FORCE_REPREPROCESSING?=0
define MICROSTATE_MAKEFILE_TRAINING_MAKEFILE_ITEM_TEMPLATE
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

define SIGNAL_TO_SEQUENCE_MAKEFILE_ITEM_TEMPLATE
signal-to-sequence-$(EXPERIMENT_TYPE)-$(GFP_PEEK_DESC)-$(NORMAL_ONLY_DESC):
	MICROSTATE_FILE=$$(CWD)/microstates/[$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)]all.npy; \\
	RECORD_FILE=$$(CWD)/microstates/[$(EXPERIMENT_TYPE)_preprocessed]all.edf; \\
	FLAGS="--record_file_path $$$$RECORD_FILE --microstate_file_path $$$$MICROSTATE_FILE --sequence_file_save_path sequences/sequence_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC).npy"; \\
	cd ../../2.microstate_training ; \\
	python signal_to_microstate_sequence.py $$$$FLAGS 2>&1 | \\
	tee /dev/tty | \\
	grep -Po 'gev\s*=\s*\K[0-9.]+' > $$(CWD)/microstates/$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC).gev

endef

define SEQUENCE_SPILTTING_MAKEFILE_ITEM_TEMPLATE
sequence_spiltting-$(EXPERIMENT_TYPE)-$(GFP_PEEK_DESC)-$(NORMAL_ONLY_DESC):
	CONFIGURATION_FILE=$$(CWD)/microstates/configurations/microstate_$(EXPERIMENT_TYPE)_$(NORMAL_ONLY_DESC)_$(GFP_PEEK_DESC).json; \\
	RECORD_FILE=$$(CWD)/microstates/[$(EXPERIMENT_TYPE)_preprocessed]all.edf; \\
	FLAGS="-cf $$$$CONFIGURATION_FILE -pz $(PREEPILEPTIC_LENGTH_SEC)"; \\
	cd ../../3.phase_space_reconstruction \\
	python dataset_splitting_and_segmentation.py $$$$FLAGS
endef

export MICROSTATE_MAKEFILE_TRAINING_MAKEFILE_ITEM_TEMPLATE
export SIGNAL_TO_SEQUENCE_MAKEFILE_ITEM_TEMPLATE
export SEQUENCE_SPILTTING_MAKEFILE_ITEM_TEMPLATE

.PHONY: generate_microstate_dev generate_microstate_eval

NORMAL_ONLY_DESC=
GFP_PEEK_DESC=

MICROSTATE_DEV_SAVE_PREFIX = [dev$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC))$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))]
MICROSTATE_DEV_DATASET_BASE_PATH = ../data/dataset
MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX = [dev_preprocessed]

MICROSTATE_EVAL_SAVE_PREFIX = [eval$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC))$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC))]
MICROSTATE_EVAL_DATASET_BASE_PATH = ../data/dataset
MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX = [eval_preprocessed]

DEV_PERSONS := p11_dev p12_dev p13_dev p14_dev p15_dev
EVAL_PERSONS := p11_eval p12_eval p13_eval p14_eval p15_eval
PERSON_IDS := 11 12 13 14 15
EXPERIMENT_TYPES := dev eval
GFP_PEEK_DESCS := nogfp gfp
NORMAL_ONLY_DESCS := normal_only all
p11_dev := "11,1" "11,2" "11,3"
p12_dev := "12,1"
p13_dev := "13,1" "13,2" "13,3"
p14_dev := "14,1" "14,2"
p15_dev := "15,1" "15,2" "15,3"
p11_eval := "11,4"
p12_eval := "12,2"
p13_eval := "13,4"
p14_eval := "14,3"
p15_eval := "15,4"

define get_indexes
$$(echo $(p$(1)_$(2)) | sed -E 's/([0-9]+,[0-9]+)/[\1]/g' | paste -sd, - | sed 's/^/[/; s/$$/]/')
endef

all: make_individual_project_11 make_individual_project_12 make_individual_project_13 make_individual_project_14 make_individual_project_15

clean-all-projects:
	rm -rf p*

make_individual_project_11:
	@DEV_INDEXES="[$(foreach pair, $(p11_dev), [$(pair)], )]" ; \
	EVAL_INDEXES="[$(foreach pair, $(p11_eval), [$(pair)], )]" ; \
	DEV_INDEXES=$$(echo $$DEV_INDEXES | sed 's/, ]/]/g') ; \
	EVAL_INDEXES=$$(echo $$EVAL_INDEXES | sed 's/, ]/]/g') ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=11 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

make_individual_project_12:
	@DEV_INDEXES="[$(foreach pair, $(p12_dev), [$(pair)], )]" ; \
	EVAL_INDEXES="[$(foreach pair, $(p12_eval), [$(pair)], )]" ; \
	DEV_INDEXES=$$(echo $$DEV_INDEXES | sed 's/, ]/]/g') ; \
	EVAL_INDEXES=$$(echo $$EVAL_INDEXES | sed 's/, ]/]/g') ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=12 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

make_individual_project_13:
	@DEV_INDEXES="[$(foreach pair, $(p13_dev), [$(pair)], )]" ; \
	EVAL_INDEXES="[$(foreach pair, $(p13_eval), [$(pair)], )]" ; \
	DEV_INDEXES=$$(echo $$DEV_INDEXES | sed 's/, ]/]/g') ; \
	EVAL_INDEXES=$$(echo $$EVAL_INDEXES | sed 's/, ]/]/g') ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=13 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

make_individual_project_14:
	@DEV_INDEXES="[$(foreach pair, $(p14_dev), [$(pair)], )]" ; \
	EVAL_INDEXES="[$(foreach pair, $(p14_eval), [$(pair)], )]" ; \
	DEV_INDEXES=$$(echo $$DEV_INDEXES | sed 's/, ]/]/g') ; \
	EVAL_INDEXES=$$(echo $$EVAL_INDEXES | sed 's/, ]/]/g') ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=14 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

make_individual_project_15:
	@DEV_INDEXES="[$(foreach pair, $(p15_dev), [$(pair)], )]" ; \
	EVAL_INDEXES="[$(foreach pair, $(p15_eval), [$(pair)], )]" ; \
	DEV_INDEXES=$$(echo $$DEV_INDEXES | sed 's/, ]/]/g') ; \
	EVAL_INDEXES=$$(echo $$EVAL_INDEXES | sed 's/, ]/]/g') ; \
	$(MAKE) _make_individual_project \
		PERSON_ID=15 \
		DEV_INDEXES="$$DEV_INDEXES" \
		EVAL_INDEXES="$$EVAL_INDEXES"

_make_individual_project:
	echo "Creating project structure for person $(PERSON_ID)" ; \
	mkdir -p p$(PERSON_ID)/microstates/configurations ; \
	mkdir -p p$(PERSON_ID)/sequences ; \
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
	echo '' >> p$(PERSON_ID)/Makefile
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			for _EXPERIMENT_TYPE in dev eval; do \
				$(MAKE) _generate_signal_to_sequence_items \
					NORMAL_ONLY_DESC=$$_NORMAL_ONLY_DESC \
					GFP_PEEK_DESC=$$_GFP_PEEK_DESC \
					EXPERIMENT_TYPE=$$_EXPERIMENT_TYPE; \
			done; \
		done; \
	done;
	echo 'signal-to-sequence-all:' >> p$(PERSON_ID)/Makefile
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			for _EXPERIMENT_TYPE in dev eval; do \
				echo "	\$$(MAKE) signal-to-sequence-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}" >> p$(PERSON_ID)/Makefile; \
			done; \
		done; \
	done;
	mkdir -p p$(PERSON_ID)/sequences/configurations	
	$(foreach EXPERIMENT_TYPE,$(EXPERIMENT_TYPES),\
	$(foreach GFP_PEEK_DESC,$(GFP_PEEK_DESCS),\
	$(foreach NORMAL_ONLY_DESC,$(NORMAL_ONLY_DESCS),\
		$(MAKE) _generate_signal_segmentation_configuration_json \
		TYPE=$(EXPERIMENT_TYPE) \
		INDEXES="$(if $(filter dev,$(EXPERIMENT_TYPE)),$(DEV_INDEXES),$(EVAL_INDEXES))" \
		SAVE_PREFIX="$(MICROSTATE_EVAL_SAVE_PREFIX)" \
		DATASET_BASE_PATH="$(MICROSTATE_EVAL_DATASET_BASE_PATH)" \
		STORE_PATH="$(MICROSTATE_EVAL_STORE_PATH)" \
		PREPROCESSED_FILE_PREFIX="$(MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX)" \
		OUTPUT_FILE="p$(PERSON_ID)/sequences/configurations/microstate_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC).json"; \
	)))

generate_microstate_dev:
	@$(MAKE) _generate_microstate_training_configuration_json \
		TYPE=dev \
		INDEXES="$(INDEXES)" \
		SAVE_PREFIX="$(MICROSTATE_DEV_SAVE_PREFIX)" \
		DATASET_BASE_PATH="$(MICROSTATE_DEV_DATASET_BASE_PATH)" \
		STORE_PATH="$(MICROSTATE_DEV_STORE_PATH)" \
		PREPROCESSED_FILE_PREFIX="$(MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX)" \
		OUTPUT_FILE="p$(PERSON_ID)/microstates/configurations/microstate_dev$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC))$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC)).json"

generate_microstate_eval:
	@$(MAKE) _generate_microstate_training_configuration_json \
		TYPE=eval \
		INDEXES="$(INDEXES)" \
		SAVE_PREFIX="$(MICROSTATE_EVAL_SAVE_PREFIX)" \
		DATASET_BASE_PATH="$(MICROSTATE_EVAL_DATASET_BASE_PATH)" \
		STORE_PATH="$(MICROSTATE_EVAL_STORE_PATH)" \
		PREPROCESSED_FILE_PREFIX="$(MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX)" \
		OUTPUT_FILE="p$(PERSON_ID)/microstates/configurations/microstate_eval$(if $(GFP_PEEK_DESC),_$(GFP_PEEK_DESC))$(if $(NORMAL_ONLY_DESC),_$(NORMAL_ONLY_DESC)).json"

_generate_microstate_training_configuration_json:
	envsubst < microstate_training_configuration_template.json.in > $(OUTPUT_FILE)
	@echo "Generated $(TYPE) configuration at $(OUTPUT_FILE)"

_generate_signal_segmentation_configuration_json:
	@INDEXES='$(INDEXES)' \
	SAVE_PREFIX='$(SAVE_PREFIX)' \
	STORE_PATH='$(STORE_PATH)' \
	DATASET_BASE_PATH='$(DATASET_BASE_PATH)' \
	envsubst < splitting_configuration.json.in > $(OUTPUT_FILE)
	@echo "Generated $(TYPE) configuration at $(OUTPUT_FILE)"
	
_generate_microstate_training_makefile_items:
	@echo "$$MICROSTATE_MAKEFILE_TRAINING_MAKEFILE_ITEM_TEMPLATE" >> p$(PERSON_ID)/Makefile
	
_generate_signal_to_sequence_items:
	@echo "$$SIGNAL_TO_SEQUENCE_MAKEFILE_ITEM_TEMPLATE" >> p$(PERSON_ID)/Makefile

signal-to-sequence-dev-nogfp-normal_only:
	MICROSTATE_FILE=$(CWD)/microstates/[dev_all_nogfp]all.npy; \
	RECORD_FILE=$(CWD)/microstates/[dev_preprocessed]all.edf; \
	FLAGS="--record_file_path $$RECORD_FILE --microstate_file_path $$MICROSTATE_FILE --sequence_file_save_path sequences/sequence_dev_nogfp_normal_only.npy"; \
	cd ../../2.microstate_training ; python signal_to_microstate_sequence.py $$FLAGS  

clean-all-projects-generations-only:
	find p*/ -type f \( -name '*.json' -o -name 'Makefile' -o -name '*.yaml' \) -delete
