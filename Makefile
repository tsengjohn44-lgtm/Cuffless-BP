# One command per pipeline step. `make all` rebuilds every validation result from scratch.
# Settings come from .env if present (see .env.example).

-include .env
export BP_DATA_DIR

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
RUN = PYTHONPATH=src $(PYTHON) -m

.PHONY: all data inspect windows features baselines train eval final demo test lint format install

all: data windows features baselines train eval

install:
	$(PYTHON) -m pip install -r requirements.txt

data:            ## download the zip, unzip Part_1.mat into data/raw
	$(RUN) bp.download

inspect:         ## print and explain the layout of one record
	$(RUN) bp.load --inspect

windows:         ## 10 s windows + SBP/DBP labels, then the record-level split
	$(RUN) bp.windows
	$(RUN) bp.split

features:
	$(RUN) bp.features

baselines:
	$(RUN) bp.baselines

train:
	$(RUN) bp.train

eval:            ## validation split only
	$(RUN) bp.evaluate

final:           ## Phase 4 only: the one and only run on the test split
	$(RUN) bp.evaluate --final

demo:
	$(PYTHON) -m streamlit run app/demo.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
