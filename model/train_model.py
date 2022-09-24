# import click # if it will be nececcery to add params in future

from .constants import TRAIN_DATA_PATH, TEST_DATA_PATH
from .data_processing import preprocess_data
from .model import train_xgb_model, run_model


# @click.command()
# @click.option("--param-name", required=False, type=...)
def train_pipeline():
    X_train, y_train = preprocess_data(TRAIN_DATA_PATH)
    X_test, y_test = preprocess_data(TEST_DATA_PATH)

    model = train_xgb_model(X_train, y_train)
    print("Model has trained!")
    _, metrics = run_model(model, X_test, y_test)
    print(f"Test f1_score: {metrics['f1_score']}")
    

if __name__ == "__main__":
    train_pipeline()
