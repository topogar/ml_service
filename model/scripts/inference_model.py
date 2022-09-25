import click
import json

from ..constants import TRAIN_DATA_PATH, TEST_DATA_PATH
from ..data_processing import preprocess_data
from ..model import load_model, run_model


@click.command()
@click.option("--model-path", required=False, type=str, default="model/saved_models/model.joblib")
def train_pipeline(model_path: str):
    X_train, y_train = preprocess_data(df_path=TRAIN_DATA_PATH)
    X_test, y_test = preprocess_data(df_path=TEST_DATA_PATH)

    model = load_model(model_path)

    _, metrics_test = run_model(model, X_test, y_test)
    _, metrics_train = run_model(model, X_train, y_train)

    
    with open('metrics.json', 'w') as fp:
        json.dump({
                'train': {
                    "f1_score": metrics_train["f1_score"]
                },
                'test': {
                    "f1_score": metrics_test["f1_score"]
                }
            },
        fp)


if __name__ == "__main__":
    train_pipeline()
