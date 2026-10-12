.PHONY: install test lint features train evaluate figures reproduce serve

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check . && ruff format --check .

# Needs the 300 tile pairs in data/tiles/ (see docs/DATA.md).
features:
	kemet1 extract

train:
	kemet1 train

evaluate:
	kemet1 evaluate

figures:
	kemet1 figures

# Everything behind the README from the committed feature table (about 2 minutes on a laptop).
reproduce: train evaluate figures

serve:
	kemet1 serve
