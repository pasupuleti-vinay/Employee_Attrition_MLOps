from kfp import compiler, dsl


IMAGE = "employee-attrition-mlops:latest"


@dsl.component(base_image=IMAGE)
def validate_data() -> str:
    import pandas as pd

    path = "/app/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(path)

    if "Attrition" not in df.columns:
        raise ValueError("Missing Attrition column")

    if len(df) == 0:
        raise ValueError("Dataset is empty")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return "Dataset validation passed"


@dsl.component(base_image=IMAGE)
def train_model(validation_result: str) -> str:
    import os
    import subprocess

    if validation_result != "Dataset validation passed":
        raise ValueError("Dataset validation failed")

    subprocess.run(["python", "train.py"], check=True)

    if not os.path.exists("/app/models/attrition_model.joblib"):
        raise FileNotFoundError("Model was not saved")

    return "Model training completed"


@dsl.component(base_image=IMAGE)
def evaluate_model(training_result: str) -> str:
    import os
    import joblib
    import pandas as pd

    if training_result != "Model training completed":
        raise ValueError("Training did not complete")

    model_path = "/app/models/attrition_model.joblib"
    data_path = "/app/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found")

    df = pd.read_csv(data_path)
    y = df["Attrition"].map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Unexpected target labels")

    # Use the same input columns as the training script.
    X = df.drop(columns=["Attrition"])
    for col in ["EmployeeNumber", "EmployeeCount", "Over18"]:
        if col in X.columns:
            X = X.drop(columns=col)

    model = joblib.load(model_path)
    predictions = model.predict(X)

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y, predictions)
    print(f"Full-dataset accuracy: {accuracy:.4f}")

    return f"Evaluation completed; accuracy={accuracy:.4f}"



@dsl.pipeline(name="employee-attrition-mlops")
def employee_attrition_pipeline():
    validation = validate_data()

    training = train_model(
        validation_result=validation.output
    ).after(validation)

    evaluate_model(
        training_result=training.output
    ).after(training)



if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=employee_attrition_pipeline,
        package_path="employee_attrition_pipeline.yaml",
    )
    print("Pipeline compiled successfully.")
