# Convenience targets for the Anemia-Screen pipeline.
# Assumes the 'anemia' conda env is active (conda activate anemia).

.PHONY: install samples test serve batch cli notebook export clean

install:
	pip install -r requirements.txt

samples:
	python -m scripts.make_sample_images

test:
	pytest

serve:
	uvicorn app.main:app --reload

# Terminal analyzer with a pop-up 3-panel Grad-CAM window.
# Usage: make cli IMG=data/sample/pale_pregnant.png
IMG ?= data/sample/pale_pregnant.png
cli:
	python -m scripts.analyze_cli --image $(IMG)

batch:
	python -m scripts.batch_process --mode local --input data/sample --out outputs/triage.csv

export:
	python -m ml.export_onnx

notebook:
	jupyter lab notebooks/walkthrough.ipynb

clean:
	rm -rf outputs __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
