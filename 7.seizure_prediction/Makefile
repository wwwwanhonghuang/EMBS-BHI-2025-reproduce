train-exp3:
	SERIALIZED_TREES_PATH=../data/exp3-nogfp-no-normal-only-general-ms-pcfg/serialized_tree ; \
	python gcn-seizure-prediction.py --tree_records_base_path $$SERIALIZED_TREES_PATH --binary --save_folder ./save/exp3 ; \

train-exp2:
	SERIALIZED_TREES_PATH=../data/exp2-nogfp-no-normal-only-general-ms-pcfg/serialized_tree ; \
        python gcn-seizure-prediction.py --tree_records_base_path $$SERIALIZED_TREES_PATH --binary --save_folder ./save/exp1 ; \

train-exp1:
	SERIALIZED_TREES_PATH=../data/exp1-distinguishability-nogfp-no-normal-only/serialized_tree ; \
        python gcn-seizure-prediction.py --tree_records_base_path $$SERIALIZED_TREES_PATH --binary --save_folder ./save/exp1 ; \

