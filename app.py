import io
import pandas as pd


from flask import Flask, request
from flask import jsonify
from model.model import load_model, run_model, train_xgb_model
from model.data_processing import preprocess_label, preprocess_msg, preprocess_data
from model.constants import num_to_label, SEP

app = Flask("ml service")
model = load_model("model/model.joblib")


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



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9001)
