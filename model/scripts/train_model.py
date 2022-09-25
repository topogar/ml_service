import click
import yaml


from ..constants import TRAIN_DATA_PATH
from ..data_processing import preprocess_data
from ..model import train_xgb_model


def read_config():
    with open('params.yaml', 'r') as fp:
        return yaml.safe_load(fp)['model']


@click.command()
@click.option("--model-path", required=False, type=str, default="model/saved_models/model.joblib")
def train_pipeline(model_path: str):
    X_train, y_train = preprocess_data(df_path=TRAIN_DATA_PATH)

    model_params = read_config()
    train_xgb_model(X_train, y_train, model_params, model_path)
    

if __name__ == "__main__":
    train_pipeline()
