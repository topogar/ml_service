import joblib

from xgboost import XGBClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


from .constants import PATH_TO_MODEL_DUMP


def train_xgb_model(X, y, xgb_params, path_to_save=PATH_TO_MODEL_DUMP):
    model = Pipeline([
        ('bow', CountVectorizer()), 
        ('tfid', TfidfTransformer()),  
        ('model', XGBClassifier(**xgb_params))
    ])

    model.fit(X, y)
    joblib.dump(model, path_to_save)

    return model


def run_model(model, X, y=None):
    preds = model.predict(X)
    if y is not None:
        return preds, {"f1_score": f1_score(y, preds)}
    return preds


def load_model(model_path):
    model = joblib.load(model_path)
    return model
