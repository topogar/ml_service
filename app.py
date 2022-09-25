import io
import pandas as pd
import subprocess
from datetime import datetime

from flask import Flask, request
from flask import jsonify
from model.model import load_model, run_model, train_xgb_model
from model.data_processing import preprocess_label, preprocess_msg, preprocess_data
from model.constants import num_to_label, SEP, TRAIN_DATA_PATH, TEST_DATA_PATH

app = Flask("ml service")
model = load_model("model/saved_models/model.joblib")


def get_df_from_blob(file):
    in_memory_file = io.BytesIO(file.read())

    return pd.read_csv(in_memory_file, sep=SEP)


@app.route("/")
def index():
    return "ok"


@app.route("/forward", methods=["POST"])
def forward_model():
    input_fields = request.get_json()
    if "message" in input_fields and input_fields["message"] is not None:
        try:
            model_input = preprocess_msg(input_fields["message"])
            prediction = run_model(model, [model_input, ])[0]

            return jsonify({
                "prediction": num_to_label[prediction]
            })
        except:
            return "модель не смогла обработать данные", 403
    else:
        return "bad request", 400


@app.route("/forward_batch", methods=["POST"])
def forward_batch():
    df_blob = request.files.get("df")
    df_to_evaluate = get_df_from_blob(df_blob)

    if "message" in df_to_evaluate.columns:
        try:
            X = df_to_evaluate["message"].apply(preprocess_msg)
            predictions = run_model(model, X)

            return jsonify({
                "predictions": [num_to_label[prediction] for prediction in predictions]
            })
        except:
            return "модель не смогла обработать данные", 403
    else:
        return "bad request", 400


@app.route("/evaluate", methods=["POST"])
def evaluate():
    df_blob = request.files.get("df")
    df_to_evaluate = get_df_from_blob(df_blob)

    if "message" in df_to_evaluate.columns and "target" in df_to_evaluate.columns:
        try:
            X, y = preprocess_data(df_to_evaluate)
            predictions, metrics = run_model(model, X, y)

            return jsonify({
                "predictions": [num_to_label[prediction] for prediction in predictions],
                "metric": metrics
            })
        except:
            return "модель не смогла обработать данные", 403
    else:
        return "bad request", 400


@app.route("/add_data", methods=["PUT"])
def add_data():
    df_blob = request.files.get("df")
    df_to_add = get_df_from_blob(df_blob)
    current_df = pd.read_csv(TRAIN_DATA_PATH, sep=SEP)

    if set(current_df.columns) == set(df_to_add.columns):
        try:
            new_train_df = pd.concat([current_df, df_to_add], axis=0)
            new_train_df.to_csv(TRAIN_DATA_PATH, sep="\t", index=None)
            
            subprocess.run(
                f"dvc add {TRAIN_DATA_PATH} && dvc push {TRAIN_DATA_PATH}.dvc", shell=True, check=True
            )

            return "Train data has updated!", 200
        except Exception as e:
            return str(e), 403
    else:
        return "bad request", 400


@app.route("/retrain", methods=["PUT"])
def retrain():
    try:
        exp_id = "exp_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        subprocess.run(
            f"dvc exp run -n {exp_id}", shell=True, check=True
        )
        model = load_model("model/saved_models/model.joblib")

        return f"Success! Experiment_id: {exp_id}", 200
    except:
        return "Can't update model!", 400


@app.route("/metrics/<experiment_id>")
def get_metrics(experiment_id):
    try:
        metrics = subprocess.run(
            f"dvc exp show --csv", shell=True, check=True, capture_output=True
        ).stdout
        metrics_pd = pd.read_csv(io.BytesIO(metrics))

        experiment_metrics = metrics_pd.query(f"Experiment == '{experiment_id}'")[
            ["Experiment", "train.f1_score", "test.f1_score"]
        ].values.tolist()[0]
        
        return jsonify({
            "Experiment": experiment_metrics[0],
            "Train f1 score": experiment_metrics[1],
            "Test f1 score": experiment_metrics[2],
        }), 200
    except:
        return f"Couldn't find: {experiment_id}!", 400


@app.route("/deploy/<experiment_id>")
def deploy_experiment_model(experiment_id):
    try:
        subprocess.run(
            f"dvc exp apply {experiment_id}", shell=True, check=True
        )
        model = load_model("model/saved_models/model.joblib")

        return f"Success! Model changed to experiment_id: {experiment_id}", 200
    except:
        return f"Couldn't update model to experiment_id: {experiment_id}!", 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9002)
