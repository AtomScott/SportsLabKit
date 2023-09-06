.ONESHELL:
.PHONY: clean data lint requirements tests create_environment install_develop develop_tests full_develop_test pyupgrade autofix all_checks ruff mypy docstring help

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = sportslabkit
SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate $(PROJECT_NAME)_dev_env

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff
ruff:
	$(CONDA_ACTIVATE) && \
	ruff check $(i)

## Type check using mypy
mypy:
	$(CONDA_ACTIVATE) && \
	mypy $(i)

## Upgrade code to Python 3.10
pyupgrade:
	$(CONDA_ACTIVATE) && \
	pyupgrade --py310-plus $(i)	

## Check docstrings with pep257
docstring:
	$(CONDA_ACTIVATE) && \
	pep257 $(i)

## Run auto-fix
autofix:
	$(CONDA_ACTIVATE) && \
	find $(i) -name '*.py' -exec pyupgrade --py310-plus {} \; && \
	ruff check --fix $(i) && \
	black $(i)

## Run all checks
all_checks: lint mypy docstring

## Run tests using pytest
tests:
	$(CONDA_ACTIVATE) && pytest --cov=./ --cov-report xml tests

## Create a conda environment for development
create_environment:
	conda env create --file dev_environment.yaml --name $(PROJECT_NAME)_dev_env
	$(CONDA_ACTIVATE) && poetry install

## Update conda environment for development
update_environment:
	conda env update --file dev_environment.yaml --name $(PROJECT_NAME)_dev_env
	$(CONDA_ACTIVATE) && poetry install

## Install package from develop branch
install_develop:
	$(CONDA_ACTIVATE) && pip install git+https://github.com/atomscott/sportslabkit.git@develop#egg=sportslabkit --upgrade

## Run tests on develop branch
develop_tests: install_develop
	$(CONDA_ACTIVATE) && python -m pytest --cov=./ --cov-report xml tests

## Full test cycle on develop branch
full_develop_test: create_environment install_develop develop_tests


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
