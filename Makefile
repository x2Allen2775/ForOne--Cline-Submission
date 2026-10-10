# Convenience targets for the Anemia-Screen pipeline.
# Assumes the 'anemia' conda env is active (conda activate anemia).

.PHONY: install samples test serve batch cli manifest train train-eye train-palm evaluate notebook export clean

install:
	pip install -r requirements.txt

samples:
	python -m scripts.make_sample_images

test:
	pytest

serve:
	uvicorn app.main:app --reload

# Build manifests from the raw datasets (Eyes-defy-anemia, CP-AnemiC, Palm).
manifest:
	python -m ml.build_manifest

train: train-eye train-palm

train-eye:
	python -m ml.train --modality eye --epochs 12 --batch-size 32

train-palm:
	python -m ml.train --modality palm --epochs 8 --batch-size 64

# Honest evaluation through the serving preprocessing path.
evaluate:
	python -m ml.evaluate --modality eye --max-images 200
	python -m ml.evaluate --modality palm --max-images 200

# Interactive dual-modality pipeline with folder/image UI selector & symptom intake
pipeline:
	python -m scripts.analyze_cli

# Terminal analyzer (supports multimodal or single image)
# Usage: make cli IMG=data/sample/pale_pregnant.png MODALITY=eye
IMG ?= data/sample/pale_pregnant.png
MODALITY ?= eye
cli:
	python -m scripts.analyze_cli --image $(IMG) --modality $(MODALITY)

batch:
	python -m scripts.batch_process --mode local --input data/sample --out outputs/triage.csv

export:
	python -m ml.export_onnx

notebook:
	jupyter lab notebooks/walkthrough.ipynb

clean:
	rm -rf outputs __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
