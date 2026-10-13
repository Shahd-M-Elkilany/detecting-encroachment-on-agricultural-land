.PHONY: install test lint features train evaluate figures reproduce serve pipeline-demo pipeline-score pipeline-evaluate

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
	python pipeline/evaluate_pipeline.py figures

# The 8-step pipeline. pipeline-score needs the tiles; pipeline-evaluate works from results/pipeline_scores.csv.
pipeline-demo:
	cd pipeline && python run.py --test

pipeline-score:
	python pipeline/evaluate_pipeline.py score --tiles data/tiles

pipeline-evaluate:
	python pipeline/evaluate_pipeline.py report

# Everything behind the README from the committed feature table (about 2 minutes on a laptop).
reproduce: train evaluate pipeline-evaluate figures

serve:
	kemet1 serve
