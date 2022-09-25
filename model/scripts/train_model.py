import click

from ..constants import TRAIN_DATA_PATH, TEST_DATA_PATH
from ..data_processing import preprocess_data
from ..model import train_xgb_model, run_model


@click.command()
@click.option("--model-path", required=False, type=str, default="model/saved_models/model.joblib")
def train_pipeline(model_path: str):
    X_train, y_train = preprocess_data(df_path=TRAIN_DATA_PATH)

    train_xgb_model(X_train, y_train, model_path)
    

if __name__ == "__main__":
    train_pipeline()
