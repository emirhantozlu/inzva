"""
Asteroid Hazard Predictor FastAPI
Applied AI Study Group #10 Week 2

Run with: uvicorn api:app --reload
Test with: Postman -> http://localhost:8000
"""

import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "asteroid_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "asteroid_features.pkl"
DATASET_PATH = PROJECT_ROOT / "data" / "asteroids.csv"

model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)
DATASET = pd.read_csv(DATASET_PATH)
FEATURE_MEANS = DATASET[FEATURES].mean()
FEATURE_STDS = DATASET[FEATURES].std(ddof=0).replace(0, 1.0)

app = FastAPI(
    title="Asteroid Hazard Predictor API",
    description="Applied AI Study Group #10 - Week 2 Workshop",
    version="1.1.0",
)


class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class AsteroidInput(APIModel):
    est_diameter_km: float = Field(..., gt=0, description="Estimated diameter in km", examples=[0.5])
    relative_velocity_km_s: float = Field(..., gt=0, description="Velocity relative to Earth in km/s", examples=[15.0])
    miss_distance_mKm: float = Field(..., gt=0, description="Miss distance in million km", examples=[20.0])
    absolute_magnitude: float = Field(..., description="Absolute magnitude", examples=[20.0])
    eccentricity: float = Field(..., ge=0, lt=1, description="Orbital eccentricity", examples=[0.4])
    inclination_deg: float = Field(..., ge=0, description="Orbital inclination in degrees", examples=[10.0])


class PredictionResponse(APIModel):
    is_hazardous: bool
    hazard_probability: float
    risk_level: str
    prediction_time_ms: float
    model_version: str
    timestamp: str
    most_influential_feature: str
    most_influential_feature_value: float


class HealthResponse(APIModel):
    status: str
    model_loaded: bool
    features: list[str]
    timestamp: str


class FeatureExplanation(APIModel):
    feature: str
    value: float
    dataset_average: float
    deviation_from_average: float
    normalized_deviation: float
    model_importance: float
    weighted_impact_score: float
    impact_level: str


class ExplainResponse(APIModel):
    is_hazardous: bool
    hazard_probability: float
    risk_level: str
    verdict: str
    most_influential_feature: str
    feature_breakdown: list[FeatureExplanation]


class CompareRequest(APIModel):
    asteroid_a: AsteroidInput
    asteroid_b: AsteroidInput


class FeatureDifference(APIModel):
    feature: str
    asteroid_a_value: float
    asteroid_b_value: float
    absolute_difference: float
    normalized_difference: float


class CompareResponse(APIModel):
    more_dangerous: str
    asteroid_a_probability: float
    asteroid_b_probability: float
    probability_difference: float
    most_different_feature: FeatureDifference
    summary: str


def asteroid_to_dict(asteroid: AsteroidInput) -> dict[str, float]:
    raw = asteroid.model_dump()
    return {feature: float(raw[feature]) for feature in FEATURES}


def asteroid_to_frame(asteroid: AsteroidInput) -> pd.DataFrame:
    return pd.DataFrame([asteroid_to_dict(asteroid)])[FEATURES]


def get_model_importances() -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        return np.abs(np.ravel(model.coef_))
    raise HTTPException(status_code=500, detail="Loaded model does not expose feature importance information.")


MODEL_IMPORTANCES = {
    feature: float(importance)
    for feature, importance in zip(FEATURES, get_model_importances())
}


def get_feature_impact_scores(feature_values: dict[str, float]) -> dict[str, float]:
    impact_scores: dict[str, float] = {}
    for feature, value in feature_values.items():
        normalized_deviation = abs(float(value) - float(FEATURE_MEANS[feature])) / float(FEATURE_STDS[feature])
        impact_scores[feature] = normalized_deviation * MODEL_IMPORTANCES[feature]
    return impact_scores


def get_top_feature(feature_values: dict[str, float]) -> tuple[str, float]:
    impact_scores = get_feature_impact_scores(feature_values)
    top_feature = max(impact_scores, key=impact_scores.get)
    return top_feature, float(feature_values[top_feature])


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    if probability < 0.6:
        return "MEDIUM"
    if probability < 0.8:
        return "HIGH"
    return "CRITICAL"


def score_asteroid(asteroid: AsteroidInput) -> dict[str, object]:
    feature_values = asteroid_to_dict(asteroid)
    input_df = pd.DataFrame([feature_values])[FEATURES]
    probability = float(model.predict_proba(input_df)[0][1])
    is_hazardous = bool(model.predict(input_df)[0])
    top_feature, top_value = get_top_feature(feature_values)
    return {
        "feature_values": feature_values,
        "probability": probability,
        "is_hazardous": is_hazardous,
        "risk_level": get_risk_level(probability),
        "top_feature": top_feature,
        "top_value": top_value,
    }


def get_impact_level(normalized_deviation: float) -> str:
    if normalized_deviation >= 1.5:
        return "HIGH"
    if normalized_deviation >= 0.75:
        return "MEDIUM"
    return "LOW"


def build_verdict(
    probability: float,
    is_hazardous: bool,
    breakdown: list[FeatureExplanation],
) -> str:
    standout = breakdown[0]
    direction = "above" if standout.deviation_from_average >= 0 else "below"
    if is_hazardous:
        return (
            f"The asteroid looks hazardous because the model assigns a {probability:.1%} risk "
            f"and {standout.feature} is notably {direction} the dataset average."
        )
    return (
        f"The asteroid appears less concerning with a {probability:.1%} risk, and its strongest signal "
        f"comes from {standout.feature}, which stays relatively manageable versus the dataset average."
    )


@app.get("/", tags=["Info"])
def root():
    """API info."""
    return {
        "name": "Asteroid Hazard Predictor",
        "version": "1.1.0",
        "endpoints": ["/predict", "/explain", "/compare", "/health", "/docs"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if the API and model are healthy."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        features=FEATURES,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(asteroid: AsteroidInput):
    """
    Predict whether an asteroid is hazardous.

    Input: Asteroid physical and orbital features
    Output: Hazard prediction with probability and key driver
    """
    start_time = time.time()
    score = score_asteroid(asteroid)
    prediction_time_ms = (time.time() - start_time) * 1000

    return PredictionResponse(
        is_hazardous=score["is_hazardous"],
        hazard_probability=round(score["probability"], 4),
        risk_level=score["risk_level"],
        prediction_time_ms=round(prediction_time_ms, 2),
        model_version="random_forest_v1.0",
        timestamp=datetime.utcnow().isoformat(),
        most_influential_feature=score["top_feature"],
        most_influential_feature_value=round(score["top_value"], 4),
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"])
def explain(asteroid: AsteroidInput):
    """
    Explain why the model sees an asteroid as hazardous or not.
    """
    score = score_asteroid(asteroid)

    breakdown: list[FeatureExplanation] = []
    impact_scores = get_feature_impact_scores(score["feature_values"])
    for feature, value in score["feature_values"].items():
        mean_value = float(FEATURE_MEANS[feature])
        deviation = float(value - mean_value)
        normalized_deviation = float(abs(deviation) / FEATURE_STDS[feature])
        breakdown.append(
            FeatureExplanation(
                feature=feature,
                value=round(value, 4),
                dataset_average=round(mean_value, 4),
                deviation_from_average=round(deviation, 4),
                normalized_deviation=round(normalized_deviation, 4),
                model_importance=round(MODEL_IMPORTANCES[feature], 4),
                weighted_impact_score=round(impact_scores[feature], 4),
                impact_level=get_impact_level(normalized_deviation),
            )
        )

    breakdown.sort(key=lambda item: item.weighted_impact_score, reverse=True)

    return ExplainResponse(
        is_hazardous=score["is_hazardous"],
        hazard_probability=round(score["probability"], 4),
        risk_level=score["risk_level"],
        verdict=build_verdict(score["probability"], score["is_hazardous"], breakdown),
        most_influential_feature=score["top_feature"],
        feature_breakdown=breakdown,
    )


@app.post("/compare", response_model=CompareResponse, tags=["Comparison"])
def compare(payload: CompareRequest):
    """
    Compare two asteroids and return which one is more dangerous.
    """
    score_a = score_asteroid(payload.asteroid_a)
    score_b = score_asteroid(payload.asteroid_b)

    feature_differences: list[FeatureDifference] = []
    for feature in FEATURES:
        value_a = float(score_a["feature_values"][feature])
        value_b = float(score_b["feature_values"][feature])
        raw_difference = abs(value_a - value_b)
        normalized_difference = raw_difference / FEATURE_STDS[feature]
        feature_differences.append(
            FeatureDifference(
                feature=feature,
                asteroid_a_value=round(value_a, 4),
                asteroid_b_value=round(value_b, 4),
                absolute_difference=round(raw_difference, 4),
                normalized_difference=round(float(normalized_difference), 4),
            )
        )

    most_different_feature = max(feature_differences, key=lambda item: item.normalized_difference)

    probability_a = float(score_a["probability"])
    probability_b = float(score_b["probability"])
    probability_difference = abs(probability_a - probability_b)

    if probability_a > probability_b:
        more_dangerous = "asteroid_a"
    elif probability_b > probability_a:
        more_dangerous = "asteroid_b"
    else:
        more_dangerous = "tie"

    summary = (
        f"{more_dangerous} is more dangerous based on hazard probability. "
        f"The biggest input gap is {most_different_feature.feature}."
    )

    return CompareResponse(
        more_dangerous=more_dangerous,
        asteroid_a_probability=round(probability_a, 4),
        asteroid_b_probability=round(probability_b, 4),
        probability_difference=round(probability_difference, 4),
        most_different_feature=most_different_feature,
        summary=summary,
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(asteroids: list[AsteroidInput]):
    """
    Predict for multiple asteroids at once.
    """
    if len(asteroids) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 asteroids per batch request",
        )

    results = []
    for asteroid in asteroids:
        score = score_asteroid(asteroid)
        results.append(
            {
                "is_hazardous": score["is_hazardous"],
                "hazard_probability": round(score["probability"], 4),
                "risk_level": score["risk_level"],
                "most_influential_feature": score["top_feature"],
            }
        )

    return {
        "count": len(results),
        "hazardous_count": sum(result["is_hazardous"] for result in results),
        "predictions": results,
    }
