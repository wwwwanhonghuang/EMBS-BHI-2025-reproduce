sampling_dataset:
	K=50; \
	L="[30,20,10,5,2]" ; \
	SAVE_PATH=./data/adhd_microstate_dataset_samples; \
        python build_adhd_microstate_samples.py --K $$K --L $$L --save_base_path $$SAVE_PATH; \

train_pcfg:
	@BINARY=../lib/pcfg-cky-inside-outside/bin/train_pcfg; \
	LIMIT=15; \
	for LENGTH in 5 10; do \
	    echo "Iterate length $$LENGTH samples. Sample amount limitatin = $$LIMIT" ; \
	    ls ./data/training_configurations/$$LENGTH/*.yaml | head -n $$LIMIT | xargs -I{} $$BINARY {}; \
	done ;\
