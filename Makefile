.PHONY: help install collect label features train reports clean

help:
	@echo "Antibody Trial Predictor - Available Commands"
	@echo "=============================================="
	@echo "make install      Install Python dependencies"
	@echo "make collect      Download and parse bulk XML data from S3"
	@echo "make label        Label trials as success/failure"
	@echo "make features     Engineer features from labeled data"
	@echo "make train        Train all 5 models in parallel"
	@echo "make reports      Generate all reports (model dashboard + EDA reports)"
	@echo "make pipeline     Run complete pipeline (collect → label → features → train → reports)"
	@echo "make clean        Remove generated data and model files"
	@echo ""
	@echo "Quick Start: make pipeline"

install:
	pip install -r requirements.txt

collect:
	@echo "Downloading and parsing bulk XML data..."
	python scripts/parse_bulk_xml.py

label:
	@echo "Labeling trials..."
	python run_pipeline.py --steps label

features:
	@echo "Engineering features..."
	python run_pipeline.py --steps features

train:
	@echo "Training all models in parallel..."
	@mkdir -p models results
	@python scripts/train_single_model.py lr data/clinical_trials_features.csv models/ & \
	python scripts/train_single_model.py dt data/clinical_trials_features.csv models/ & \
	python scripts/train_single_model.py rf data/clinical_trials_features.csv models/ & \
	python scripts/train_single_model.py gb data/clinical_trials_features.csv models/ & \
	python scripts/train_single_model.py xgb data/clinical_trials_features.csv models/ & \
	wait
	@echo "✓ All models trained"

reports:
	@echo "Generating all reports..."
	@mkdir -p reports docs
	@echo "Generating model dashboard..."
	python scripts/generate_overview.py models reports/model_comparison.csv docs/index.html
	@echo "Generating EDA reports..."
	python scripts/generate_eda_raw.py data/clinical_trials_binary.csv docs/eda_raw_data.html \
		--timestamp "$$(date -u +'%Y-%m-%d %H:%M:%S UTC')" \
		--data-source "ClinicalTrials.gov API"
	python scripts/generate_eda_features.py data/clinical_trials_features.csv docs/eda_features.html \
		--timestamp "$$(date -u +'%Y-%m-%d %H:%M:%S UTC')" \
		--data-source "ClinicalTrials.gov API"
	@echo "✓ All reports generated"
	@echo "  - Model Dashboard: docs/index.html"
	@echo "  - Raw Data EDA: docs/eda_raw_data.html"
	@echo "  - Features EDA: docs/eda_features.html"
	@echo "View: open docs/index.html"

pipeline: collect label features train reports
	@echo ""
	@echo "✓ Pipeline Complete!"
	@echo "  - Data: data/clinical_trials_features.csv"
	@echo "  - Models: models/"
	@echo "  - Reports: docs/index.html (+ EDA reports)"

clean:
	@echo "Cleaning generated files..."
	rm -rf data/*.csv data/*.json
	rm -rf models/*.pkl models/*.json
	rm -rf reports/*.csv
	rm -rf docs/*.html
	@echo "✓ Cleaned"
