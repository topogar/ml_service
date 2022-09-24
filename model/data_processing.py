import re
import string
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split


from .constants import *


stop_words = stopwords.words('english')
stemmer = nltk.SnowballStemmer("english")


preprocess_label = lambda label: label_to_num[label]


def make_initial_train_test_split(file_path):
    data = pd.read_csv(file_path, sep=SEP)

    train_data, test_data = train_test_split(data, random_state=SEED)
    
    train_data.to_csv(
        TRAIN_DATA_PATH, sep=SEP, index=None
    )
    test_data.to_csv(
        TEST_DATA_PATH, sep=SEP, index=None
    )


def clean_text(text):
    """
    Clean text from unnecessary symbols and
    """
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def preprocess_msg(msg, stop_words=stop_words, stemmer=stemmer):

    msg = clean_text(msg)
    msg = ' '.join(word for word in msg.split(' ') if word not in stop_words)
    msg = ' '.join(stemmer.stem(word) for word in msg.split(' '))
    
    return msg


def preprocess_data(df=None, df_path=None):
    if df is None:
        df = pd.read_csv(df_path, sep=SEP)
    
    df['message'] = df['message'].apply(preprocess_msg)
    df['target'] = df['target'].apply(preprocess_label)
    
    return df["message"], df["target"]