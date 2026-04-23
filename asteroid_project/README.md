# Asteroid Hazard Prediction

An end-to-end mini ML project for detecting whether an asteroid is hazardous based on physical and orbital features.

This repository combines:

- `Week 1`: data exploration, model training, and an interactive Streamlit app
- `Week 2`: a FastAPI service, Docker packaging, and MLflow experiment tracking

## Project Goal

The goal of the project is to show how a machine learning model can move beyond a notebook into a usable product workflow.
We start with asteroid data, train a classifier to predict hazard risk, visualize and explore the model in Streamlit, then expose the same prediction system through an API and track experiments with MLflow.

## Repository Structure

```text
asteroid-hazard-prediction/
|-- README.md
|-- .gitignore
|-- .dockerignore
|-- requirements.txt
|-- environment.yml
|-- data/
|   `-- asteroids.csv
|-- models/
|   |-- asteroid_model.pkl
|   `-- asteroid_features.pkl
|-- week1/
|   |-- week1_asteroid.ipynb
|   `-- app.py
|-- week2/
|   |-- api.py
|   |-- train_mlflow.py
|   `-- Dockerfile
`-- deliverables/
    |-- mlflow/
    `-- screenshots/
```

## Setup

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate asteroid-week2
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

## Week 1

### Notebook

Open the notebook from the `week1/` folder:

```bash
jupyter notebook week1/week1_asteroid.ipynb
```

The notebook reads the dataset from `data/` and saves the trained artifacts into `models/`.

### Streamlit App

Run the Week 1 interactive app:

```bash
streamlit run week1/app.py
```

Open:

- `http://localhost:8501`

## Week 2

### FastAPI

Run the API locally:

```bash
python -m uvicorn week2.api:app --reload
```

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

### Example `/predict` Request

```json
{
  "est_diameter_km": 1.2,
  "relative_velocity_km_s": 25.0,
  "miss_distance_mKm": 5.0,
  "absolute_magnitude": 17.5,
  "eccentricity": 0.7,
  "inclination_deg": 15.0
}
```

### Docker

Build from the repository root:

```bash
docker build -f week2/Dockerfile -t asteroid-api .
docker run -p 8000:8000 asteroid-api
```

## MLflow Experiments

Run the Week 2 experiment script:

```bash
python week2/train_mlflow.py
python -m mlflow ui --backend-store-uri sqlite:///deliverables/mlflow/mlflow.db
```

Open:

- `http://localhost:5000`

Artifacts are written into:

- `deliverables/mlflow/mlflow.db`
- `deliverables/mlflow/mlflow_analysis.txt`
- `deliverables/mlflow/plots/`

## Main Features

- Random Forest based hazard classification
- Streamlit dashboard for interactive asteroid risk exploration
- FastAPI endpoints for prediction, explanation, and comparison
- Dockerized API service
- MLflow-based experiment tracking and recall-focused model comparison

## Notes

- `recall` is treated as the most important metric because missing a hazardous asteroid is more costly than a false alarm.
- The repository keeps both the trained model artifacts and the code used to evaluate and serve them.
