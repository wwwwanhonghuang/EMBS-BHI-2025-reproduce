all: train_microstate_dev

# Default values (can be overridden by command-line)
FORCE_REPREPROCESSING ?= 0
PREEPILEPTIC_LENGTH_SEC ?= 120
CONFIGURATION_FILE ?= ./configs/config-all-person-microstate-dev.json

train-microstate-dev:
	@echo "Running microstate training (Dev dataset) with:"
	@echo "  FORCE_REPREPROCESSING=$(FORCE_REPREPROCESSING)"
	@echo "  PREEPILEPTIC_LENGTH_SEC=$(PREEPILEPTIC_LENGTH_SEC)"
	@echo "  CONFIGURATION_FILE=$(CONFIGURATION_FILE)"
	@FLAGS="--pre_epileptic_zone $(PREEPILEPTIC_LENGTH_SEC) --database_index_configuration $(CONFIGURATION_FILE) --gfp"; \
	if [ "$(FORCE_REPREPROCESSING)" -eq 1 ]; then \
		FLAGS="$$FLAGS --force_repreprocessing"; \
	fi; \
	python microstate_extraction.py $$FLAGS --no_normal_only

signal-to-microstate:

