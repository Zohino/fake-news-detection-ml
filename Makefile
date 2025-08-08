#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = fake-news-detection-ml
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .venv
	rm -rf models/*
	rm -rf data/processed/*
	rm -rf reports/figures/*
	rm -f reports/sample-predictions.md
	rm -f reports/*.html

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Virtual environment created. Activate with:"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Download raw data and validate structure
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) detection/dataset.py download

## Preprocess raw data and save to processed/
.PHONY: preprocess
preprocess: data
	$(PYTHON_INTERPRETER) notebooks/1.0-data-cleaning-and-feature-creation.ipynb

## Generate exploratory visualizations
.PHONY: visualize
visualize:
	$(PYTHON_INTERPRETER) notebooks/2.0-visualizations.ipynb

## Train model and generate results
.PHONY: train
train: preprocess
	$(PYTHON_INTERPRETER) train.py

## Predict on test set using trained model
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) predict.py

## Generate publication summary
.PHONY: publish
publish:
	$(PYTHON_INTERPRETER) notebooks/4.0-publication.ipynb

## Run the full pipeline: download -> preprocess -> train -> predict -> publish
.PHONY: pipeline
pipeline: train predict publish

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

## Show this help
.PHONY: help
help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
