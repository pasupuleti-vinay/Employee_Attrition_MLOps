import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "attrition_model.joblib"


def main():
    print("Loading employee attrition dataset...")

    df = pd.read_csv(DATA_PATH)

    # Remove columns that do not help prediction.
    df = df.drop(
        columns=[
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours",
        ],
        errors="ignore",
    )

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"].map({"Yes": 1, "No": 0})

    categorical = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        exclude=["object", "str"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
        (
            "numerical",
            "passthrough",
            numerical,
        ),
    ])

    model = Pipeline([
        ("preprocessing", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model saved successfully.")


if __name__ == "__main__":
    main()
