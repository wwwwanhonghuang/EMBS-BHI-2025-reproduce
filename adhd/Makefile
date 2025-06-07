sampling_dataset:
	K=50; \
	L="[30,20,10,5,2]" ; \
	SAVE_PATH=./data/adhd_microstate_dataset_samples; \
        python build_adhd_microstate_samples.py --K $$K --L $$L --save_base_path $$SAVE_PATH; \

train_pcfg:
	@BINARY=../lib/pcfg-cky-inside-outside/bin/train_pcfg; \
	LIMIT=15; \
	for LENGTH in 10; do \
	    echo "Iterate length $$LENGTH samples. Sample amount limitatin = $$LIMIT" ; \
	    ls ./data/training_configurations/$$LENGTH/*.yaml | head -n $$LIMIT | xargs -I{} $$BINARY {}; \
	done ;\

generate_adhd_dataset:
	python build_adhd_samples.py

process_adhd_dataset:
	python generate_adhd_sentences.py

phase_reconstruction_adhd_dataset:
	python phase_space_reconstruction_adhd.py

prepare_syntax_analysis:
	PY_BINARY=prepare_syntax_analysis_data.py ; \
	for LENGTH in 30 20; do \
	    ls data/trained_grammars/$$LENGTH | xargs -I{} python $$PY_BINARY --md5 {} --l $$LENGTH; \
	done ; \
	for LENGTH in 30 20; do \
           for ID in $$(ls data/trained_grammars/$$LENGTH); do \
             cat data/syntax_analysis_data/$$LENGTH/$$ID/adhd/A*_repetitions.txt | grep -v '^$$' > data/syntax_analysis_data/$$LENGTH/$$ID/adhd/merged_repetitions.txt ; \
             cat data/syntax_analysis_data/$$LENGTH/$$ID/adhd/A*_seg.txt | grep -v '^$$' > data/syntax_analysis_data/$$LENGTH/$$ID/adhd/merged_segs.txt ; \
	     cat data/syntax_analysis_data/$$LENGTH/$$ID/controls/C*_repetitions.txt | grep -v '^$$' > data/syntax_analysis_data/$$LENGTH/$$ID/controls/merged_repetitions.txt ; \
             cat data/syntax_analysis_data/$$LENGTH/$$ID/controls/C*_seg.txt | grep -v '^$$' > data/syntax_analysis_data/$$LENGTH/$$ID/controls/merged_segs.txt ; \
	   done ; \
	   ls data/trained_grammars/$$LENGTH | xargs -I{} python generate_syntax_analysis_configurations.py --md5 {} --l $$LENGTH ; \
       done ; \

syntax_analysis:
	SYNTAX_ANALYSIS_BINARY=../lib/pcfg-cky-inside-outside/bin/syntax_analysis ; \
	echo "SYNTAX_ANALYSIS_BINARY=$$SYNTAX_ANALYSIS_BINARY" ; \
	for LENGTH in 30 20; do \
           for ID in $$(ls data/trained_grammars/$$LENGTH); do \
               $$SYNTAX_ANALYSIS_BINARY data/syntax_analysis_data/$$LENGTH/$$ID/syntax_analysis_configuration_adhd.yaml ; \
	       $$SYNTAX_ANALYSIS_BINARY data/syntax_analysis_data/$$LENGTH/$$ID/syntax_analysis_configuration_controls.yaml ; \
	   done ; \
       done ; \


