.PHONY: help install collect label features train dashboard clean

help:
	@echo "Antibody Trial Predictor - Available Commands"
	@echo "=============================================="
	@echo "make install      Install Python dependencies"
	@echo "make collect      Download and parse bulk XML data from S3"
	@echo "make label        Label trials as success/failure"
	@echo "make features     Engineer features from labeled data"
	@echo "make train        Train all 5 models in parallel"
	@echo "make dashboard    Generate model comparison dashboard"
	@echo "make pipeline     Run complete pipeline (collect → label → features → train → dashboard)"
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

dashboard:
	@echo "Generating reports..."
	@mkdir -p reports docs
	python scripts/aggregate_results.py models reports/model_comparison.csv
	python scripts/generate_report.py models reports/model_comparison.csv docs/model_dashboard.html
	@echo "✓ Dashboard generated at docs/model_dashboard.html"
	@echo "View: open docs/index.html"

pipeline: collect label features train dashboard
	@echo ""
	@echo "✓ Pipeline Complete!"
	@echo "  - Data: data/clinical_trials_features.csv"
	@echo "  - Models: models/"
	@echo "  - Landing Page: docs/index.html"
	@echo "  - Model Dashboard: docs/model_dashboard.html"

clean:
	@echo "Cleaning generated files..."
	rm -rf data/*.csv data/*.json
	rm -rf models/*.pkl models/*.json
	rm -rf reports/*.csv
	rm -rf docs/*.html
	@echo "✓ Cleaned"
