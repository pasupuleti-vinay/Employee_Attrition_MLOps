from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "attrition_model.joblib"


def main():
    print("Loading employee attrition dataset...")

    df = pd.read_csv(DATA_PATH)

    if "Attrition" not in df.columns:
        raise ValueError("Dataset must contain an Attrition column.")

    # Remove columns that contain only one value.
    df = df.loc[:, df.nunique(dropna=False) > 1].copy()

    # Remove employee identifiers when present.
    for column in ["EmployeeNumber", "EmployeeCount", "Over18"]:
        if column in df.columns:
            df = df.drop(columns=column)

    y = df.pop("Attrition").map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Attrition must contain Yes/No values.")

    categorical = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "ROC-AUC:",
        round(roc_auc_score(y_test, probabilities), 4),
    )
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
