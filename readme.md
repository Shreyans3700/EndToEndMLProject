# End-to-End Machine Learning Project

## Overview

This repository contains an end-to-end machine learning application for predicting student math performance using demographic and exam score features. It includes data ingestion, preprocessing, model selection, hyperparameter tuning, model persistence, and a Flask-based web interface for live prediction.

## Project Structure

- `application.py` - Flask application entrypoint for serving the web UI and prediction endpoint.
- `src/` - Python package containing pipeline components, utility helpers, and configuration.
  - `components/data_ingestion.py` - loads raw student data and creates train/test splits.
  - `components/data_transformation.py` - builds preprocessing pipelines for numeric and categorical features.
  - `components/model_trainer.py` - evaluates multiple regression models and saves the best model.
  - `pipeline/predict_pipeline.py` - loads the saved model and preprocessor for inference.
  - `configs/models_hyperparameter_configs.py` - hyperparameter search space for model tuning.
  - `utils.py` - serialization helpers, model evaluation helpers, and runtime utilities.
- `artifacts/` - generated artifacts such as `train.csv`, `test.csv`, `data.csv`, pretrained `model.pkl`, and `preprocessor.pkl`.
- `notebooks/` - exploratory notebooks and the raw dataset under `notebooks/data/students.csv`.
- `templates/` - HTML templates used by the Flask web application.
- `requirements.txt` - Python dependencies.
- `setup.py` - package setup for installing project dependencies.

## Features

- Regression modeling for `math_score` prediction
- Categorical encoding for student demographic fields
- Standard scaling for numeric scores
- Model comparison across multiple regressors, including XGBoost
- Web interface for interactive predictions
- Artifact persistence using `dill`

## Installation

1. Create a Python virtual environment:

```bash
python -m venv .venv
```

2. Activate the environment:

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Flask app

Start the web application:

```bash
python application.py
```

Then open `http://127.0.0.1:5000` in your browser.

### Prediction flow

- The app accepts student demographic inputs and reading/writing scores.
- It converts the form values into a DataFrame using `src.pipeline.predict_pipeline.CustomData`.
- `PredictPipeline` loads the saved preprocessor and trained model from `artifacts/`.
- The output is a predicted math score displayed in the web UI.

## Training and Model Pipeline

The repository is designed to support an ML pipeline with the following steps:

1. Data ingestion from `notebooks/data/students.csv`.
2. Saving raw data and train/test splits under `artifacts/`.
3. Data transformation using column-wise pipelines for numeric and categorical data.
4. Model training and hyperparameter tuning for several regressors.
5. Saving the best model and preprocessing pipeline to `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

> Note: The web app depends on prebuilt artifacts in `artifacts/`. If you want to retrain, run the pipeline components manually or extend `src/pipeline/train_pipeline.py`.

## Dependencies

Key libraries used:

- Flask
- NumPy
- pandas
- scikit-learn
- XGBoost
- dill
- gunicorn

## Notes

- The dataset is located at `notebooks/data/students.csv`.
- Output artifacts are generated in the `artifacts/` directory.
- The prediction target is `math_score`.
- The app expects categorical values under keys like `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, and `test_preparation_course`.

## Contact

Author: Shreyansh

Email: paneyshreyansh46@gmail.com
