
PREEPILEPTIC_LENGTH_SEC?=120
FORCE_REPREPROCESSING?=0
INITIAL_PCFG_PATH?=$(PWD)/../data/pcfg/grammar.pcfg
PSR_BINARY_PATH?=$(PWD)/../lib/pcfg-cky-inside-outside/bin/phase_convert
PCFG_ESTIMATOR_BINARY_PATH?=$(PWD)/../lib/pcfg-cky-inside-outside/bin/train_pcfg
SYNTAX_ANALYSIS_BINARY_PATH?=$(PWD)/../lib/pcfg-cky-inside-outside/bin/syntax_analysis
TOPO_MAP_REMAPPING_SCRIPT_PATH?=$(PWD)/../3.phase_space_reconstruction/topomap_remapping.py
EXP_ID ?= 3

define MICROSTATE_MAKEFILE_TRAINING_MAKEFILE_ITEM_TEMPLATE
train-microstate-$(EXPERIMENT_TYPE)-$(GFP_PEEK_DESC)-$(NORMAL_ONLY_DESC):
	CONFIGURATION_FILE=$$(CWD)/microstates/configurations/microstate_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC).json; \\
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
	MICROSTATE_FILE=$$$$(find "$$(CWD)/microstates" -type f -regextype posix-extended -regex ".*/\[\[${EXPERIMENT_TYPE}_${GFP_PEEK_DESC}_${NORMAL_ONLY_DESC}\]\]person_all_states4_gev_0\.[0-9]+\.npy" | sort | head -n 1) ; \\
	echo "Find microstate file $$$$MICROSTATE_FILE" ; \\
	RECORD_FILE=$$(CWD)/microstates/[$(EXPERIMENT_TYPE)_preprocessed]pall.edf; \\
	FLAGS="--record_file_path $$$$RECORD_FILE --microstate_file_path $$$$MICROSTATE_FILE --sequence_file_save_path $$(CWD)/sequences/sequence_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC).npy"; \\
	cd ../../2.microstate_training ; \\
	python signal_to_microstate_sequence.py $$$$FLAGS 2>&1 | \\
	tee /dev/tty | \\
	grep -Po 'gev\s*=\s*\K[0-9.]+' > $$(CWD)/microstates/$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC).gev; \\
	if [ "$(EXPERIMENT_TYPE)" = "eval" ]; then \\
		DEV_MICROSTATE_FILE="$$$$(find "$$(CWD)/microstates" -type f -regextype posix-extended -regex ".*/\[\[dev_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)\]\]person_all_states4_gev_0\.[0-9]+\.npy" | sort | head -n 1)" ; \\
		FLAGS="--record_file_path $$$$RECORD_FILE --microstate_file_path $$$$DEV_MICROSTATE_FILE --sequence_file_save_path $$(CWD)/sequences/sequence_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)_devms_on_eval.npy"; \\
		python signal_to_microstate_sequence.py $$$$FLAGS 2>&1 | \\
		tee /dev/tty | \\
		grep -Po 'gev\s*=\s*\K[0-9.]+' > $$(CWD)/microstates/$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)_devms_on_eval.gev; \\
	fi;

endef

export MICROSTATE_MAKEFILE_TRAINING_MAKEFILE_ITEM_TEMPLATE
export SIGNAL_TO_SEQUENCE_MAKEFILE_ITEM_TEMPLATE

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


test-remapping:
	for PERSON_ID in 11; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				MICROSTATE_BASE_PATH=$(PWD)/p$${PERSON_ID}/microstates ; \
				EVAL_MICROSTATE_FILE="$$(find "$${MICROSTATE_BASE_PATH}" -type f -regextype posix-extended -regex ".*/\\[\\[eval_$${g}_$${n}\\]\\]person_all_states4_gev_0\.[0-9]+\\.npy" | sort | head -n 1)" ; \
				DEV_MICROSTATE_FILE="$$(find "$${MICROSTATE_BASE_PATH}" -type f -regextype posix-extended -regex ".*/\\[\\[dev_$${g}_$${n}\\]\\]person_all_states4_gev_0\.[0-9]+\\.npy" | sort | head -n 1)" ; \
				REMAP=$$(python $(TOPO_MAP_REMAPPING_SCRIPT_PATH) --input_microstate_file $${EVAL_MICROSTATE_FILE} --dst_microstate_file $${DEV_MICROSTATE_FILE} | sed -E 's/^\((.*)\)$$/[\1]/' | tr -d ' ') ; \
				echo "REMAP=$$REMAP" ; \
			done ; \
		done ; \
	done;

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
				MICROSTATE_DEV_STORE_PATH="$(PWD)/p$(PERSON_ID)/microstates/" \
				NORMAL_ONLY_DESC=$$_NORMAL_ONLY_DESC \
				GFP_PEEK_DESC=$$_GFP_PEEK_DESC; \
		done; \
	done;
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			$(MAKE) generate_microstate_eval \
				INDEXES="$(EVAL_INDEXES)" \
				PERSON_ID=$(PERSON_ID) \
				MICROSTATE_EVAL_STORE_PATH="$(PWD)/p$(PERSON_ID)/microstates/" \
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
# SAVE_PREFIX="$(subst ],\\\\], $(subst [,\\\\[, [seg-$(if $(filter dev,$(EXPERIMENT_TYPE)),$(MICROSTATE_DEV_SAVE_PREFIX),$(MICROSTATE_EVAL_SAVE_PREFIX))]))" \

	$(foreach EXPERIMENT_TYPE,$(EXPERIMENT_TYPES),\
	$(foreach GFP_PEEK_DESC,$(GFP_PEEK_DESCS),\
	$(foreach NORMAL_ONLY_DESC,$(NORMAL_ONLY_DESCS),\
	    $(shell mkdir -p p$(PERSON_ID)/sequences/corpus_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)) \
		$(MAKE) _generate_signal_segmentation_configuration_json \
		TYPE=$(EXPERIMENT_TYPE) \
		INDEXES="[$(if $(filter dev,$(EXPERIMENT_TYPE)),$(DEV_INDEXES),$(EVAL_INDEXES))]" \
		SAVE_PREFIX="sequence_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)" \
		DATASET_BASE_PATH="$(MICROSTATE_EVAL_DATASET_BASE_PATH)" \
		CORPUS_STORE_PATH="$(PWD)/p$(PERSON_ID)/sequences/corpus_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)" \
		STORE_PATH="$(PWD)/p$(PERSON_ID)/sequences/" \
		PREPROCESSED_FILE_PREFIX="$(if $(filter dev,$(EXPERIMENT_TYPE)),$(MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX),$(MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX))" \
		OUTPUT_FILE="$(PWD)/p$(PERSON_ID)/sequences/configurations/microstate_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC).json"; \
	)))
	$(foreach EXPERIMENT_TYPE,eval,\
	$(foreach GFP_PEEK_DESC,$(GFP_PEEK_DESCS),\
	$(foreach NORMAL_ONLY_DESC,$(NORMAL_ONLY_DESCS),\
	    $(shell mkdir -p p$(PERSON_ID)/sequences/corpus_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)_devms_on_eval) \
		$(MAKE) _generate_signal_segmentation_configuration_json \
		TYPE=$(EXPERIMENT_TYPE) \
		INDEXES="[$(if $(filter dev,$(EXPERIMENT_TYPE)),$(DEV_INDEXES),$(EVAL_INDEXES))]" \
		SAVE_PREFIX="sequence_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)_devms_on_eval" \
		DATASET_BASE_PATH="$(MICROSTATE_EVAL_DATASET_BASE_PATH)" \
		CORPUS_STORE_PATH="$(PWD)/p$(PERSON_ID)/sequences/corpus_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)_devms_on_eval" \
		STORE_PATH="$(PWD)/p$(PERSON_ID)/sequences/" \
		PREPROCESSED_FILE_PREFIX="$(if $(filter dev,$(EXPERIMENT_TYPE)),$(MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX),$(MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX))" \
		OUTPUT_FILE="$(PWD)/p$(PERSON_ID)/sequences/configurations/microstate_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)_devms_on_eval.json"; \
	)))
	@for g in nogfp gfp; do \
		for n in normal_only all; do \
			for e in dev eval; do \
				if [ "$$e" = "dev" ]; then \
					prefix=$(MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX); \
				else \
					prefix=$(MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX); \
				fi; \
				echo "sequence_spiltting-$$e-$$g-$$n:" >> p$(PERSON_ID)/Makefile; \
				echo "	@echo \"Running with $$e $$n $$g\"" >> p$(PERSON_ID)/Makefile; \
				echo "	CONFIGURATION_FILE=\$$(CWD)/sequences/configurations/microstate_$${e}_$${g}_$${n}.json; \\" >> p$(PERSON_ID)/Makefile; \
				echo "	RECORD_FILE=\$$(CWD)/microstates/$${prefix}all.edf; \\" >> p$(PERSON_ID)/Makefile; \
				echo "	FLAGS=\"-cf \$$\$${CONFIGURATION_FILE} -pz 120\"; \\" >> p$(PERSON_ID)/Makefile; \
				echo "	cd ../../3.phase_space_reconstruction; \\" >> p$(PERSON_ID)/Makefile; \
				echo "	python dataset_splitting_and_segmentation.py \$$\$${FLAGS}" >> p$(PERSON_ID)/Makefile; \
				echo "" >> p$(PERSON_ID)/Makefile; \
			done; \
		done; \
	done
	@for g in nogfp gfp; do \
		for n in normal_only all; do \
			for e in eval; do \
				if [ "$$e" = "dev" ]; then \
					prefix=$(MICROSTATE_DEV_PREPROCESSED_FILE_PREFIX); \
				else \
					prefix=$(MICROSTATE_EVAL_PREPROCESSED_FILE_PREFIX); \
				fi; \
				echo "sequence_spiltting-$$e-$$g-$$n-devms-on-eval:" >> p$(PERSON_ID)/Makefile; \
				echo "	@echo \"Running with $$e $$n $$g\"" >> p$(PERSON_ID)/Makefile; \
				echo "	CONFIGURATION_FILE=\$$(CWD)/sequences/configurations/microstate_$${e}_$${g}_$${n}_devms_on_eval.json; \\" >> p$(PERSON_ID)/Makefile; \
				echo "	RECORD_FILE=\$$(CWD)/microstates/$${prefix}all.edf; \\" >> p$(PERSON_ID)/Makefile; \
				echo "	FLAGS=\"-cf \$$\$${CONFIGURATION_FILE} -pz 120\"; \\" >> p$(PERSON_ID)/Makefile; \
				echo "	cd ../../3.phase_space_reconstruction; \\" >> p$(PERSON_ID)/Makefile; \
				echo "	python dataset_splitting_and_segmentation.py \$$\$${FLAGS}" >> p$(PERSON_ID)/Makefile; \
				echo "" >> p$(PERSON_ID)/Makefile; \
			done; \
		done; \
	done
	echo 'sequence_spiltting-all:' >> p$(PERSON_ID)/Makefile
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			for _EXPERIMENT_TYPE in dev eval; do \
				echo "	\$$(MAKE) sequence_spiltting-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}" >> p$(PERSON_ID)/Makefile; \
				if [ "$${_EXPERIMENT_TYPE}" = "eval" ]; then \
					echo "	\$$(MAKE) sequence_spiltting-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}-devms-on-eval" >> p$(PERSON_ID)/Makefile; \
				fi; \
			done; \
		done; \
	done;
	@for e in dev eval; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				echo "npy-to-plain-text-$$e-$$g-$$n:" >> p$(PERSON_ID)/Makefile; \
				echo "	@echo \"Running with $$e $$g $$n\"" >> p$(PERSON_ID)/Makefile; \
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/seizure_integrated_all_d2_s4.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}/seizure_integrated_all_d2_s4.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/pre-epileptic_integrated_all_d2_s4.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}/pre-epileptic_integrated_all_d2_s4.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/seizure_integrated_all_d2_s4_repetition.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}/seizure_integrated_all_d2_s4_repetition.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4_repetition.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4_repetition.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/pre-epileptic_integrated_all_d2_s4_repetition.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}/pre-epileptic_integrated_all_d2_s4_repetition.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "" >> p$(PERSON_ID)/Makefile; \
				done; \
		done; \
	done
	@for e in eval; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				echo "npy-to-plain-text-$$e-$$g-$$n-devms-on-eval:" >> p$(PERSON_ID)/Makefile; \
				echo "	@echo \"Running with $$e $$g $$n\"" >> p$(PERSON_ID)/Makefile; \
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/seizure_integrated_all_d2_s4.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/seizure_integrated_all_d2_s4.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/normal_integrated_all_d2_s4.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/normal_integrated_all_d2_s4.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/pre-epileptic_integrated_all_d2_s4.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/pre-epileptic_integrated_all_d2_s4.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/seizure_integrated_all_d2_s4_repetition.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/seizure_integrated_all_d2_s4_repetition.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/normal_integrated_all_d2_s4_repetition.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/normal_integrated_all_d2_s4_repetition.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "	python ../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/pre-epileptic_integrated_all_d2_s4_repetition.npy --output_file_path ./sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/pre-epileptic_integrated_all_d2_s4_repetition.txt" >> p$(PERSON_ID)/Makefile ;\
				echo "" >> p$(PERSON_ID)/Makefile; \
				echo "npy-to-plain-text-$$e-$$g-$$n-remap-to-dev:" >> p$(PERSON_ID)/Makefile; \
				echo "	@echo \"Running with $$e $$g $$n\"" >> p$(PERSON_ID)/Makefile; \
				mkdir -p $(PWD)/p$${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_remap_to_dev ; \
				echo "	MICROSTATE_BASE_PATH=$(PWD)/p${PERSON_ID}/microstates ; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	EVAL_MICROSTATE_FILE=\"\$$\$$(find \"\$$\$${MICROSTATE_BASE_PATH}\" -type f -regextype posix-extended -regex \".*/\\[\\[eval_$${g}_$${n}\\]\\]person_all_states4_gev_0\.[0-9]+\\.npy\" | sort | head -n 1)\" ; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	DEV_MICROSTATE_FILE=\"\$$\$$(find \"\$$\$${MICROSTATE_BASE_PATH}\" -type f -regextype posix-extended -regex \".*/\\[\\[dev_$${g}_$${n}\\]\\]person_all_states4_gev_0\.[0-9]+\\.npy\" | sort | head -n 1)\" ; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	cd \$$(CWD)/../../3.phase_space_reconstruction ; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	REMAP=\$$\$$(python $(TOPO_MAP_REMAPPING_SCRIPT_PATH) --input_microstate_file \$$\$${EVAL_MICROSTATE_FILE} --dst_microstate_file \$$\$${DEV_MICROSTATE_FILE} | sed -E 's/^\((.*)\)\$$\$$/[\\\\1]/' | tr -d ' ') ; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	python \$$(CWD)/../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/seizure_integrated_all_d2_s4.npy --output_file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_remap_to_dev/seizure_integrated_all_d2_s4.txt --remap \$$\$${REMAP} ; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	python \$$(CWD)/../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4.npy --output_file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_remap_to_dev/normal_integrated_all_d2_s4.txt --remap \$$\$${REMAP} ; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	python \$$(CWD)/../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/pre-epileptic_integrated_all_d2_s4.npy --output_file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_remap_to_dev/pre-epileptic_integrated_all_d2_s4.txt --remap \$$\$${REMAP} ; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	python \$$(CWD)/../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/seizure_integrated_all_d2_s4_repetition.npy --output_file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_remap_to_dev/seizure_integrated_all_d2_s4_repetition.txt ; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	python \$$(CWD)/../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4_repetition.npy --output_file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_remap_to_dev/normal_integrated_all_d2_s4_repetition.txt; \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	python \$$(CWD)/../../4.pcfg_training/sentence_plain_text_encoder.py --file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/pre-epileptic_integrated_all_d2_s4_repetition.npy --output_file_path $(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_remap_to_dev/pre-epileptic_integrated_all_d2_s4_repetition.txt ;" >> p$(PERSON_ID)/Makefile ;\
				echo "" >> p$(PERSON_ID)/Makefile; \
			done; \
		done; \
	done
	echo 'npy-to-plain-text-all:' >> p$(PERSON_ID)/Makefile
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			for _EXPERIMENT_TYPE in dev eval; do \
				echo "	\$$(MAKE) npy-to-plain-text-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}" >> p$(PERSON_ID)/Makefile; \
				if [ "$${_EXPERIMENT_TYPE}" = "eval" ]; then \
					echo "	\$$(MAKE) npy-to-plain-text-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}-devms-on-eval" >> p$(PERSON_ID)/Makefile; \
					echo "	\$$(MAKE) npy-to-plain-text-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}-remap-to-dev" >> p$(PERSON_ID)/Makefile; \
				fi; \
			done; \
		done; \
	done;
	@for e in dev eval; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				for a in normal pre-epileptic seizure; do \
					$(MAKE) _make_phase_space_reconstruction_yaml_config \
						INPUT_FILE=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/$${a}_integrated_all_d2_s4.txt \
						OUTPUT_FILE=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/$${a}_integrated_all_d2_s4_converted.txt \
						INITIAL_PCFG_PATH=$(INITIAL_PCFG_PATH) \
						EXPERIMENT_TYPE=$${e} \
						GFP_PEEK_DESC=$${g} \
						NORMAL_ONLY_DESC=$${n} \
						AREA_TYPE=$${a} \
						PSR_FILE_NAME_EXT=; \
					if [ "$${e}" = "eval" ]; then \
						$(MAKE) _make_phase_space_reconstruction_yaml_config \
								INPUT_FILE=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/$${a}_integrated_all_d2_s4.txt \
								OUTPUT_FILE=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_devms_on_eval/$${a}_integrated_all_d2_s4_converted.txt \
								INITIAL_PCFG_PATH=$(INITIAL_PCFG_PATH) \
								EXPERIMENT_TYPE=$${e} \
								GFP_PEEK_DESC=$${g} \
								NORMAL_ONLY_DESC=$${n} \
								AREA_TYPE=$${a} \
								PSR_FILE_NAME_EXT=_devms_on_eval ; \
						$(MAKE) _make_phase_space_reconstruction_yaml_config \
								INPUT_FILE=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_remap_to_dev/$${a}_integrated_all_d2_s4.txt \
								OUTPUT_FILE=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}_remap_to_dev/$${a}_integrated_all_d2_s4_converted.txt \
								INITIAL_PCFG_PATH=$(INITIAL_PCFG_PATH) \
								EXPERIMENT_TYPE=$${e} \
								GFP_PEEK_DESC=$${g} \
								NORMAL_ONLY_DESC=$${n} \
								AREA_TYPE=$${a} \
								PSR_FILE_NAME_EXT=_remap_to_dev ; \
					fi; \
				done; \
			done; \
		done; \
	done
	@for e in dev eval; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				echo "psr-$${e}-$${g}-$${n}:" >> p$(PERSON_ID)/Makefile; \
				for a in normal pre-epileptic seizure; do \
					CONFIGURATION_BASE_PATH="$(PWD)/p${PERSON_ID}/sequences/configurations/psr_$${e}_$${g}_$${n}_$${a}.yaml" ;  \
					echo "	${PSR_BINARY_PATH} $${CONFIGURATION_BASE_PATH}" >> p$(PERSON_ID)/Makefile; \
				done; \
				if [ "$${e}" = "eval" ]; then \
					echo "psr-$${e}-$${g}-$${n}-devms-on-eval:" >> p$(PERSON_ID)/Makefile; \
					for a in normal pre-epileptic seizure; do \
						CONFIGURATION_BASE_PATH="$(PWD)/p${PERSON_ID}/sequences/configurations/psr_$${e}_$${g}_$${n}_$${a}_devms_on_eval.yaml" ;  \
						echo "	${PSR_BINARY_PATH} $${CONFIGURATION_BASE_PATH}" >> p$(PERSON_ID)/Makefile; \
					done; \
					echo "psr-$${e}-$${g}-$${n}-remap-to-dev:" >> p$(PERSON_ID)/Makefile; \
					for a in normal pre-epileptic seizure; do \
						CONFIGURATION_BASE_PATH="$(PWD)/p${PERSON_ID}/sequences/configurations/psr_$${e}_$${g}_$${n}_$${a}_remap_to_dev.yaml" ;  \
						echo "	${PSR_BINARY_PATH} $${CONFIGURATION_BASE_PATH}" >> p$(PERSON_ID)/Makefile; \
					done; \
				fi; \
			done; \
		done; \
	done
	echo 'psr-all:' >> p$(PERSON_ID)/Makefile
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			for _EXPERIMENT_TYPE in dev eval; do \
				echo "	\$$(MAKE) psr-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}" >> p$(PERSON_ID)/Makefile; \
				if [ "$${_EXPERIMENT_TYPE}" = "eval" ]; then \
					echo "	\$$(MAKE) psr-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}-devms-on-eval" >> p$(PERSON_ID)/Makefile; \
					echo "	\$$(MAKE) psr-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}-remap-to-dev" >> p$(PERSON_ID)/Makefile; \
				fi; \
			done; \
		done; \
	done;
	@for e in dev eval; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				echo "convert-length-$$e-$$g-$$n:" >> p$(PERSON_ID)/Makefile; \
				echo "	@echo \"Running with $$e $$g $$n\"" >> p$(PERSON_ID)/Makefile; \
				echo "	cd ../../4.pcfg_training ;" \
				echo "	python convert_state_length.py --file_path $(PWD)/sequences/corpus_$${e}_$${g}_$${n}/seizure_integrated_all_d2_s4_repetition.npy --output_file_path $(PWD)/sequences/corpus_$${e}_$${g}_$${n}/seizure_integrated_all_d2_s4_repetition.txt \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	python convert_state_length.py --file_path $(PWD)/sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4_repetition.npy --output_file_path $(PWD)/sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4_repetition.txt \\" >> p$(PERSON_ID)/Makefile ;\
				echo "	python convert_state_length.py --file_path $(PWD)/sequences/corpus_$${e}_$${g}_$${n}/pre-epileptic_integrated_all_d2_s4_repetition.npy --output_file_path $(PWD)/sequences/corpus_$${e}_$${g}_$${n}/pre-epileptic_integrated_all_d2_s4_repetition.txt \\" >> p$(PERSON_ID)/Makefile ;\
				echo "" >> p$(PERSON_ID)/Makefile; \
				done; \
		done; \
	done
	echo 'convert-length-all:' >> p$(PERSON_ID)/Makefile
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			for _EXPERIMENT_TYPE in dev eval; do \
				echo "	\$$(MAKE) convert-length-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}" >> p$(PERSON_ID)/Makefile; \
			done; \
		done; \
	done;
	mkdir -p p${PERSON_ID}/pcfg/configurations
	@for e in dev; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				mkdir -p $(PWD)/p${PERSON_ID}/pcfg/$${e}_$${g}_$${n}/logs ; \
				$(MAKE) _make_pcfg_training_configuration \
				INITIAL_PCFG_FILE='$(INITIAL_PCFG_PATH)' \
				INPUT_FILE=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4_converted.txt \
				VAL_SENTENCES_PATH=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4_converted_val.txt \
				TRAIN_SENTENCES_PATH=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}/normal_integrated_all_d2_s4_converted_train.txt \
				EXPERIMENT_TYPE=$${e} \
				GFP_PEEK_DESC=$${g} \
				LOG_PATH=$(PWD)/p${PERSON_ID}/pcfg/$${e}_$${g}_$${n}/logs \
				NORMAL_ONLY_DESC=$${n}; \
			done; \
		done; \
	done
	@for e in dev; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				echo "train-pcfg-$${e}-$${g}-$${n}:" >> p$(PERSON_ID)/Makefile; \
				CONFIGURATION_BASE_PATH="$(PWD)/p${PERSON_ID}/pcfg/configurations/train_$${e}_$${g}_$${n}.yaml" ;  \
				echo "	${PCFG_ESTIMATOR_BINARY_PATH} $${CONFIGURATION_BASE_PATH}" >> p$(PERSON_ID)/Makefile; \
				echo "	mv $(PWD)/p${PERSON_ID}/pcfg/$${e}_$${g}_$${n}/logs/*.pcfg $(PWD)/p${PERSON_ID}/pcfg/$${e}_$${g}_$${n}/" ; \
				echo "" >> p$(PERSON_ID)/Makefile; \
			done; \
		done; \
	done
	echo 'train-all:' >> p$(PERSON_ID)/Makefile
	for _GFP_PEEK_DESC in nogfp gfp; do \
		for _NORMAL_ONLY_DESC in normal_only all; do \
			for _EXPERIMENT_TYPE in dev; do \
				echo "	\$$(MAKE) train-pcfg-$${_EXPERIMENT_TYPE}-$${_GFP_PEEK_DESC}-$${_NORMAL_ONLY_DESC}" >> p$(PERSON_ID)/Makefile; \
			done; \
		done; \
	done;
	for g in nogfp gfp; do \
		for n in normal_only all; do \
			for e in eval; do \
				GRAMMAR_FILE=$$(find "$(PWD)/p${PERSON_ID}/pcfg/dev_$${g}_$${n}/" -name "*.pcfg") ; \
				for __GRAMMAR_FILE in $$GRAMMAR_FILE; do \
					mkdir -p $(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}_devms_on_eval ; \
					mkdir -p $(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}_remap_to_dev ; \
					GRAMMAR_FILE_NAME="$$(basename "$$__GRAMMAR_FILE")" ; \
					sed -E 's/\[[0-9]+\]\s+//g; s/\s+->\s+/->/g' $${__GRAMMAR_FILE} > $(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}_devms_on_eval/$${GRAMMAR_FILE_NAME} ; \
					sed -E 's/\[[0-9]+\]\s+//g; s/\s+->\s+/->/g' $${__GRAMMAR_FILE} > $(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}_remap_to_dev/$${GRAMMAR_FILE_NAME} ; \
				done; \
			done ; \
		done; \
	done;
	mkdir -p p${PERSON_ID}/syntax_analysis/configurations
	@for e in eval; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				for a in normal seizure pre-epileptic; do \
					for et in _devms_on_eval _remap_to_dev; do \
						mkdir -p $(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}$${et} ; \
						mkdir -p $(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}$${et}/logs ; \
						mkdir -p $(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}$${et}/reports ; \
						$(MAKE) _make_syntax_analysis_configuration \
						GRAMMAR_FILE=$$(find "$(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}$${et}/" -name "*.pcfg" | sort -r | head -n 1)  \
						INPUT_FILE=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}$${EVAL_TYPE}/$${a}_integrated_all_d2_s4_converted.txt \
						REPETITION_FILE=$(PWD)/p${PERSON_ID}/sequences/corpus_$${e}_$${g}_$${n}$${EVAL_TYPE}/$${a}_integrated_all_d2_s4_repetition.txt \
						EXPERIMENT_TYPE=$${e} \
						GFP_PEEK_DESC=$${g} \
						LOG_PATH=$(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}$${et}/logs \
						REPORT_PATH=$(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}$${et}/reports \
						TREE_STORE_PATH=$(PWD)/p${PERSON_ID}/syntax_analysis/$${e}_$${g}_$${n}$${et}/serialized_trees/$${a} \
						NORMAL_ONLY_DESC=$${n} \
						AREA_TYPE=$${a} \
						EVAL_TYPE=$${et}; \
					done; \
				done; \
			done; \
		done; \
	done
	@for e in dev; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				echo "train-pcfg-$${e}-$${g}-$${n}:" >> p$(PERSON_ID)/Makefile; \
				CONFIGURATION_BASE_PATH="$(PWD)/p${PERSON_ID}/pcfg/configurations/train_$${e}_$${g}_$${n}.yaml" ;  \
				echo "	${PCFG_ESTIMATOR_BINARY_PATH} $${CONFIGURATION_BASE_PATH}" >> p$(PERSON_ID)/Makefile; \
				echo "	mv $(PWD)/p${PERSON_ID}/pcfg/$${e}_$${g}_$${n}/logs/*.pcfg $(PWD)/p${PERSON_ID}/pcfg/$${e}_$${g}_$${n}/" ; \
				echo "" >> p$(PERSON_ID)/Makefile; \
			done; \
		done; \
	done
	@for e in eval; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				for a in normal seizure pre-epileptic; do \
					echo "syntax-analysis-$${e}-$${g}-$${n}-$${a}-devms-on-eval:" >> p$(PERSON_ID)/Makefile; \
					CONFIGURATION_BASE_PATH="$(PWD)/p${PERSON_ID}/syntax_analysis/configurations/syntax_analysis_$${e}_$${g}_$${n}_$${a}_devms_on_eval.yaml" ;  \
					echo "	${SYNTAX_ANALYSIS_BINARY_PATH} $${CONFIGURATION_BASE_PATH}" >> p$(PERSON_ID)/Makefile; \
					echo "" >> p$(PERSON_ID)/Makefile; \
					echo "syntax-analysis-$${e}-$${g}-$${n}-$${a}-remap-to-dev:" >> p$(PERSON_ID)/Makefile; \
					CONFIGURATION_BASE_PATH="$(PWD)/p${PERSON_ID}/syntax_analysis/configurations/syntax_analysis_$${e}_$${g}_$${n}_$${a}_remap_to_dev.yaml" ;  \
					echo "	${SYNTAX_ANALYSIS_BINARY_PATH} $${CONFIGURATION_BASE_PATH}" >> p$(PERSON_ID)/Makefile; \
					echo "" >> p$(PERSON_ID)/Makefile; \
				done; \
			done; \
		done; \
	done
	echo 'syntax-analysis-all:' >> p$(PERSON_ID)/Makefile
	@for e in eval; do \
		for g in nogfp gfp; do \
			for n in normal_only all; do \
				for a in normal seizure pre-epileptic; do \
					for suffix in -remap-to-dev -devms-on-eval; do \
						echo "	\$$(MAKE) syntax-analysis-$${e}-$${g}-$${n}-$${a}$${suffix}" >> p$(PERSON_ID)/Makefile; \
					done; \
				done; \
			done; \
		done; \
	done

generate_microstate_dev:
	@$(MAKE) _generate_microstate_training_configuration_json \
		TYPE=dev \
		INDEXES="$(INDEXES)" \
		SAVE_PREFIX=$(subst, [, \\[ [$(MICROSTATE_DEV_SAVE_PREFIX)])  \
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

# signal-to-sequence-dev-nogfp-normal_only:
# 	MICROSTATE_FILE=$(CWD)/microstates/[dev_all_nogfp]all.npy; \
# 	RECORD_FILE=$(CWD)/microstates/[dev_preprocessed]all.edf; \
# 	FLAGS="--record_file_path $$RECORD_FILE --microstate_file_path $$MICROSTATE_FILE --sequence_file_save_path sequences/sequence_dev_nogfp_normal_only.npy"; \
# 	cd ../../2.microstate_training ; python signal_to_microstate_sequence.py $$FLAGS  

clean-all-projects-generations-only:
	find p*/ -type f \( -name '*.json' -o -name 'Makefile' -o -name '*.yaml' \) -delete

_make_phase_space_reconstruction_yaml_config:
	@INPUT_FILE='$(INPUT_FILE)' \
	OUTPUT_FILE='$(OUTPUT_FILE)' \
	INITIAL_PCFG_PATH='$(INITIAL_PCFG_PATH)' \
	EXPERIMENT_TYPE='$(EXPERIMENT_TYPE)' \
	GFP_PEEK_DESC='$(GFP_PEEK_DESC)' \
	NORMAL_ONLY_DESC='$(NORMAL_ONLY_DESC)' \
	AREA_TYPE='$(AREA_TYPE)' \
	PSR_FILE_NAME_EXT='$(PSR_FILE_NAME_EXT)' \
	envsubst < phase_space_reconstruction.yaml.in > p$(PERSON_ID)/sequences/configurations/psr_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)_$(AREA_TYPE)$(PSR_FILE_NAME_EXT).yaml

_make_pcfg_training_configuration:
	@INITIAL_PCFG_FILE='$(INITIAL_PCFG_FILE)' \
	INPUT_FILE='$(INPUT_FILE)' \
	VAL_SENTENCES_PATH='$(VAL_SENTENCES_PATH)' \
	TRAIN_SENTENCES_PATH='$(TRAIN_SENTENCES_PATH)' \
	envsubst < training_configuration.yaml.in > p$(PERSON_ID)/pcfg/configurations/train_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC).yaml

_make_syntax_analysis_configuration:
	@GRAMMAR_FILE='$(GRAMMAR_FILE)' \
	INPUT_FILE='$(INPUT_FILE)' \
	REPETITION_FILE='$(REPETITION_FILE)' \
	LOG_PATH='$(LOG_PATH)' \
	REPORT_PATH='$(REPORT_PATH)' \
	TREE_STORE_PATH='$(TREE_STORE_PATH)' \
	EVAL_TYPE='$(EVAL_TYPE)' \
	envsubst < syntax_analysis_configuration.yaml.in > p$(PERSON_ID)/syntax_analysis/configurations/syntax_analysis_$(EXPERIMENT_TYPE)_$(GFP_PEEK_DESC)_$(NORMAL_ONLY_DESC)_$(AREA_TYPE)$(EVAL_TYPE).yaml
