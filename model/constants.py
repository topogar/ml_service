SEP = "\t"
SEED = 42


DATA_PATH = ".data/data.tsv"
TRAIN_DATA_PATH = "data/train_data.tsv"
TEST_DATA_PATH = "data/test_data.tsv"


PATH_TO_MODEL_DUMP = "model/saved_models/model.joblib"


label_to_num = {
    "ham": 0,
    "spam": 1
}


num_to_label = {
    value: key for key, value in label_to_num.items()
}