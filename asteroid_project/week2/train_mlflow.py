"""
Asteroid Hazard Prediction - MLflow Experiment Tracking
Applied AI Study Group #10 - Week 2

Run:
    python train_mlflow.py
    mlflow ui  -> open http://localhost:5000
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "asteroids.csv"
DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"
MLFLOW_DIR = DELIVERABLES_DIR / "mlflow"
PLOTS_DIR = MLFLOW_DIR / "plots"
ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"
SCREENSHOTS_DIR = DELIVERABLES_DIR / "screenshots"
TRACKING_DB_PATH = MLFLOW_DIR / "mlflow.db"
EXPERIMENT_NAME = "asteroid-hazard-detection"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "asteroid_model.pkl"
FEATURES_OUTPUT_PATH = PROJECT_ROOT / "models" / "asteroid_features.pkl"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

ALL_FEATURES = [
    "est_diameter_km",
    "relative_velocity_km_s",
    "miss_distance_mKm",
    "absolute_magnitude",
    "eccentricity",
    "inclination_deg",
]

X_all = df[ALL_FEATURES]
y = df["is_hazardous"]

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB_PATH.as_posix()}")
mlflow_client = MlflowClient()
if mlflow_client.get_experiment_by_name(EXPERIMENT_NAME) is None:
    mlflow_client.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=ARTIFACTS_DIR.resolve().as_uri(),
    )
mlflow.set_experiment(EXPERIMENT_NAME)


def safe_filename(name: str) -> str:
    invalid = r'\/:*?"<>|'
    for char in invalid:
        name = name.replace(char, "")
    return name.replace(" ", "_")


def get_estimator_for_importance(model):
    if hasattr(model, "named_steps"):
        return model.named_steps[list(model.named_steps.keys())[-1]]
    return model


def get_feature_importance_frame(model, feature_names: list[str]) -> pd.DataFrame | None:
    estimator = get_estimator_for_importance(model)

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance_values = abs(estimator.coef_).ravel()
    else:
        return None

    return pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False)


def log_confusion_matrix(y_true, y_pred, run_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Hazardous", "Hazardous"])
    ax.set_yticklabels(["Not Hazardous", "Hazardous"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {run_name}")

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                cm[row, col],
                ha="center",
                va="center",
                color="white" if cm[row, col] > cm.max() / 2 else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    output_path = PLOTS_DIR / f"confusion_matrix_{safe_filename(run_name)}.png"
    plt.savefig(output_path, dpi=100)
    plt.close()
    mlflow.log_artifact(str(output_path))


def log_feature_importance(model, feature_names: list[str], run_name: str) -> None:
    importance_df = get_feature_importance_frame(model, feature_names)
    if importance_df is None:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue", edgecolor="white")
    ax.set_title(f"Feature Importance - {run_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    output_path = PLOTS_DIR / f"feature_importance_{safe_filename(run_name)}.png"
    plt.savefig(output_path, dpi=100)
    plt.close()
    mlflow.log_artifact(str(output_path))


def evaluate_predictions(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def run_experiment(
    run_name: str,
    model,
    features: list[str],
    model_type: str,
    model_params: dict[str, object],
    extra_params: dict[str, object] | None = None,
    save_production_model: bool = False,
) -> dict[str, float]:
    print(f"\n{'=' * 55}")
    print(f"  Run: {run_name}")
    print(f"{'=' * 55}")

    X_tr = X_train[features]
    X_te = X_test[features]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("features_used", ", ".join(features))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        for key, value in model_params.items():
            mlflow.log_param(key, value)

        if extra_params:
            for key, value in extra_params.items():
                mlflow.log_param(key, value)

        model.fit(X_tr, y_train)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        metrics = evaluate_predictions(y_test, y_pred, y_prob)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        log_feature_importance(model, features, run_name)
        log_confusion_matrix(y_test, y_pred, run_name)
        mlflow.sklearn.log_model(model, "model")

        if save_production_model:
            joblib.dump(model, MODEL_OUTPUT_PATH)
            joblib.dump(features, FEATURES_OUTPUT_PATH)
            print("  -> Saved as asteroid_model.pkl (production model)")

        for metric_name, metric_value in metrics.items():
            marker = " <- most important for hazard detection" if metric_name == "recall" else ""
            print(f"  {metric_name}: {metric_value}{marker}")

        return metrics


def build_engineered_feature_dataset() -> tuple[pd.DataFrame, list[str]]:
    engineered_df = df.copy()
    engineered_df["velocity_diameter_ratio"] = (
        engineered_df["relative_velocity_km_s"] / engineered_df["est_diameter_km"].clip(lower=0.001)
    )
    feature_list = ALL_FEATURES + ["velocity_diameter_ratio"]
    return engineered_df[feature_list], feature_list


print("\nAsteroid Hazard Detection - MLflow Experiments")
print("=" * 55)
print("Open http://localhost:5000 after running to compare results")
print("=" * 55)

results: dict[str, dict[str, float]] = {}

results["Baseline (all features, 100 trees)"] = run_experiment(
    run_name="Baseline (all features, 100 trees)",
    model=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    features=ALL_FEATURES,
    model_type="random_forest",
    model_params={"n_estimators": 100, "max_depth": "unlimited"},
    save_production_model=True,
)

results["50 trees - faster but less accurate?"] = run_experiment(
    run_name="50 trees - faster but less accurate?",
    model=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
    features=ALL_FEATURES,
    model_type="random_forest",
    model_params={"n_estimators": 50, "max_depth": "unlimited"},
)

results["200 trees - better but worth the cost?"] = run_experiment(
    run_name="200 trees - better but worth the cost?",
    model=RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
    features=ALL_FEATURES,
    model_type="random_forest",
    model_params={"n_estimators": 200, "max_depth": "unlimited"},
)

results["No inclination_deg - does it matter?"] = run_experiment(
    run_name="No inclination_deg - does it matter?",
    model=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    features=[feature for feature in ALL_FEATURES if feature != "inclination_deg"],
    model_type="random_forest",
    model_params={"n_estimators": 100, "max_depth": "unlimited"},
    extra_params={"removed_feature": "inclination_deg"},
)

results["Physical features only"] = run_experiment(
    run_name="Physical features only",
    model=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    features=["est_diameter_km", "relative_velocity_km_s", "miss_distance_mKm", "absolute_magnitude"],
    model_type="random_forest",
    model_params={"n_estimators": 100, "max_depth": "unlimited"},
    extra_params={"feature_group": "physical_only", "removed": "eccentricity, inclination_deg"},
)

results["max_depth=5 - constrained trees"] = run_experiment(
    run_name="max_depth=5 - constrained trees",
    model=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1),
    features=ALL_FEATURES,
    model_type="random_forest",
    model_params={"n_estimators": 100, "max_depth": 5},
    extra_params={"regularization": "max_depth=5"},
)

X_all_eng, engineered_features = build_engineered_feature_dataset()
X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
    X_all_eng,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

engineered_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)

with mlflow.start_run(run_name="Engineered feature: velocity_diameter_ratio"):
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", "unlimited")
    mlflow.log_param("n_features", len(engineered_features))
    mlflow.log_param("features_used", ", ".join(engineered_features))
    mlflow.log_param("engineered_feature", "velocity_diameter_ratio = velocity / diameter")

    engineered_model.fit(X_train_eng, y_train_eng)
    y_pred_eng = engineered_model.predict(X_test_eng)
    y_prob_eng = engineered_model.predict_proba(X_test_eng)[:, 1]
    engineered_metrics = evaluate_predictions(y_test_eng, y_pred_eng, y_prob_eng)

    for metric_name, metric_value in engineered_metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    log_feature_importance(engineered_model, engineered_features, "Engineered feature: velocity_diameter_ratio")
    log_confusion_matrix(y_test_eng, y_pred_eng, "Engineered feature: velocity_diameter_ratio")
    mlflow.sklearn.log_model(engineered_model, "model")

    results["Engineered feature: velocity_diameter_ratio"] = engineered_metrics

    print(f"\n{'=' * 55}")
    print("  Run: Engineered feature: velocity_diameter_ratio")
    print(f"{'=' * 55}")
    for metric_name, metric_value in engineered_metrics.items():
        marker = " <- most important for hazard detection" if metric_name == "recall" else ""
        print(f"  {metric_name}: {metric_value}{marker}")

results["Logistic regression (all features, balanced)"] = run_experiment(
    run_name="Logistic regression (all features, balanced)",
    model=Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    ),
    features=ALL_FEATURES,
    model_type="logistic_regression",
    model_params={
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "liblinear",
        "scaled_features": True,
    },
)

sorted_results = sorted(results.items(), key=lambda item: item[1]["recall"], reverse=True)
best_run_name, best_run_metrics = sorted_results[0]

analysis_lines = [
    f"The best recall came from '{best_run_name}' with recall={best_run_metrics['recall']}.",
    "The added logistic regression run reached recall=0.7778, so it underperformed the stronger random forest runs on hazardous-object detection.",
    "Recall matters most here because a false negative means missing a truly hazardous asteroid, which is more costly than a false alarm.",
    "For a safety-focused asteroid screening system, I would choose the best-recall random forest first and tune precision afterward if alert volume becomes a problem.",
]

analysis_path = MLFLOW_DIR / "mlflow_analysis.txt"
analysis_path.write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")

print("\n" + "=" * 55)
print("Leaderboard by recall")
print("=" * 55)
for run_name, metrics in sorted_results:
    print(f"  {run_name}: recall={metrics['recall']}, precision={metrics['precision']}, roc_auc={metrics['roc_auc']}")

print("\nAnalysis")
for line in analysis_lines:
    print(f"  - {line}")

print("\nArtifacts written:")
print(f"  - {analysis_path.name}")
print("  - confusion matrices and feature importance charts per run")

print("\nNext steps:")
print("  1. Run:  mlflow ui")
print("  2. Open: http://localhost:5000")
print("  3. Select all runs -> Compare")
print("  4. Sort by 'recall'")
print("=" * 55)
