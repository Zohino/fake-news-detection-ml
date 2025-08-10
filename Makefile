#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = fake-news-detection-ml
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python3
VENV_ACTIVATE = . .venv/bin/activate

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create Python virtual environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Virtual environment created. Activate with:"
	@echo ">>> source ./.venv/bin/activate"

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Setup environment and install dependencies
.PHONY: setup
setup: create_environment requirements
	@echo ">>> Environment setup complete. Activate with:"
	@echo ">>> source ./.venv/bin/activate"

## Download and prepare raw dataset
.PHONY: data
data: requirements
	$(VENV_ACTIVATE) && $(PYTHON_INTERPRETER) src/dataset.py download

## Preprocess raw data
.PHONY: preprocess
preprocess: data
	$(VENV_ACTIVATE) && $(PYTHON_INTERPRETER) src/preprocessing/preprocessing.py

## Train machine learning model
.PHONY: train
train: preprocess
	$(VENV_ACTIVATE) && $(PYTHON_INTERPRETER) src/modeling/train.py

## Generate predictions using trained model
.PHONY: predict
predict: train
	$(VENV_ACTIVATE) && $(PYTHON_INTERPRETER) src/modeling/predict.py

## Run initial exploratory data analysis notebook
.PHONY: eda
eda: data
	$(VENV_ACTIVATE) && papermill notebooks/0.1-mz-initial-eda.ipynb notebooks/0.1-mz-initial-eda.ipynb

## Run data cleaning and feature creation notebook
.PHONY: features
features: preprocess
	$(VENV_ACTIVATE) && papermill notebooks/1.1-mz-data-cleaning-and-feature-creation.ipynb notebooks/1.1-mz-data-cleaning-and-feature-creation.ipynb

## Run visualization notebook
.PHONY: visualize
visualize: features
	$(VENV_ACTIVATE) && papermill notebooks/2.1-mz-visualization.ipynb notebooks/2.1-mz-visualization.ipynb

## Run all notebooks in sequence
.PHONY: notebooks
notebooks: eda features visualize

## Run complete ML pipeline
.PHONY: pipeline
pipeline: setup data preprocess train predict notebooks

## Delete compiled Python files and temporary outputs
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Lint code with ruff
.PHONY: lint
lint:
	ruff format --check
	ruff check

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available targets:\n'); \
print('\n'.join(['{:20}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

## Show available targets and descriptions
.PHONY: help
help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
	@echo "\nRecommended usage:"
	@echo "  make pipeline    # Run complete ML pipeline from setup to notebooks"