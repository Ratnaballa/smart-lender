import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = DATA_DIR / "loan_prediction.csv"
MODEL_PATH = DATA_DIR / "model.pkl"
HISTORY_PATH = DATA_DIR / "history.json"


TARGET_COLUMN = "Loan_Status"
CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERIC_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Dependents",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def ensure_dataset() -> Path:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Place loan_prediction.csv in the project root."
        )
    return DATASET_PATH


def load_dataset() -> pd.DataFrame:
    path = ensure_dataset()
    df = pd.read_csv(path)
    return df


def preprocess_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")
    df["LoanAmount"] = pd.to_numeric(df["LoanAmount"], errors="coerce")
    df["Loan_Amount_Term"] = pd.to_numeric(df["Loan_Amount_Term"], errors="coerce")
    df["Credit_History"] = pd.to_numeric(df["Credit_History"], errors="coerce")
    df["ApplicantIncome"] = pd.to_numeric(df["ApplicantIncome"], errors="coerce")
    df["CoapplicantIncome"] = pd.to_numeric(df["CoapplicantIncome"], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).fillna("Unknown")

    y = df[TARGET_COLUMN].map({"Y": 1, "N": 0, "Yes": 1, "No": 0}).fillna(0)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def build_models() -> Dict[str, Any]:
    return {
        "DecisionTree": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
            ]
        ),
    }


def train_models() -> Dict[str, Dict[str, Any]]:
    X, y = preprocess_dataset(load_dataset())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = build_models()
    results: Dict[str, Dict[str, Any]] = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        results[name] = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "report": classification_report(y_test, predictions, output_dict=True),
        }
    return results


def select_best_model() -> Tuple[str, Any, Dict[str, Any]]:
    results = train_models()
    best_name = max(results, key=lambda name: (results[name]["accuracy"], results[name]["f1_score"]))
    models = build_models()
    X, y = preprocess_dataset(load_dataset())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models[best_name].fit(X_train, y_train)
    return best_name, models[best_name], results


def save_model(model: Any, path: Path = MODEL_PATH) -> None:
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def load_model(path: Path = MODEL_PATH) -> Any:
    if not path.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_prediction_payload(payload: Dict[str, Any], prediction: int, probability: float) -> Dict[str, Any]:
    approved = bool(prediction == 1)
    return {
        "approved": approved,
        "confidence": round(float(probability) * 100, 2),
        "status": "Approved" if approved else "Rejected",
        "recommendation": (
            "Proceed with standard underwriting and verify repayment capacity."
            if approved
            else "Request more supporting documents and review credit exposure."
        ),
        "score": round(float(probability) * 100, 2),
    }


def validate_input(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_fields = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
    ]
    for field in required_fields:
        if field not in payload:
            errors.append(f"Missing required field: {field}")
    if payload.get("ApplicantIncome") is not None and payload.get("ApplicantIncome") < 0:
        errors.append("ApplicantIncome cannot be negative")
    if payload.get("CoapplicantIncome") is not None and payload.get("CoapplicantIncome") < 0:
        errors.append("CoapplicantIncome cannot be negative")
    if payload.get("LoanAmount") is not None and payload.get("LoanAmount") < 0:
        errors.append("LoanAmount cannot be negative")
    if payload.get("Loan_Amount_Term") is not None and payload.get("Loan_Amount_Term") <= 0:
        errors.append("Loan_Amount_Term must be greater than zero")
    return errors


def prepare_feature_row(payload: Dict[str, Any]) -> pd.DataFrame:
    row = {
        "Gender": payload.get("Gender", "Male"),
        "Married": payload.get("Married", "No"),
        "Dependents": payload.get("Dependents", 0),
        "Education": payload.get("Education", "Graduate"),
        "Self_Employed": payload.get("Self_Employed", "No"),
        "ApplicantIncome": payload.get("ApplicantIncome", 0),
        "CoapplicantIncome": payload.get("CoapplicantIncome", 0),
        "LoanAmount": payload.get("LoanAmount", 0),
        "Loan_Amount_Term": payload.get("Loan_Amount_Term", 360),
        "Credit_History": payload.get("Credit_History", 1),
        "Property_Area": payload.get("Property_Area", "Urban"),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_history(history: List[Dict[str, Any]]) -> None:
    with HISTORY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def append_history(entry: Dict[str, Any]) -> None:
    history = load_history()
    history.insert(0, entry)
    save_history(history[:25])


def train_and_save_model() -> Dict[str, Any]:
    best_name, model, results = select_best_model()
    save_model(model)
    return {"best_model": best_name, "results": results}
