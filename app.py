from flask import Flask, request
from flask import jsonify
from model.model import load_model, run_model, train_xgb_model
from model.data_processing import preprocess_label, preprocess_msg
from model.constants import num_to_label

app = Flask("ml service")
model = load_model("model/model.joblib")


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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9001)
