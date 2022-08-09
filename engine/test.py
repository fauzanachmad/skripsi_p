import json
import os
import pickle
import pandas as pd

import nltk
import numpy as np

nltk.download('stopwords')
from engine.utils.text_mining_util import basic_clean
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def load_all_data():
    data = pd.read_csv('resources/created/preprocessing_result_new.csv', encoding='ISO-8859-1')
    data = data[data.title != '[none]']
    return data


class Testing:
    def __init__(self):
        self.vectorized = None
        self.model = None
        self.data = load_all_data()

        self.train, self.test = train_test_split(self.data, test_size=0.2, random_state=0)
        self.pickle_parser()

    def pickle_parser(self):
        if os.path.exists("resources/models/vectorized.pickle") & os.path.exists("resources/models/model.pickle"):
            self.vectorized = pickle.load(open("resources/models/vectorized.pickle", 'rb'))
            self.model = pickle.load(open("resources/models/model.pickle", 'rb'))

    def title_cleaner(self, title):
        if self.vectorized is not None:
            title_clean = title
            title_clean = np.append(title_clean, basic_clean(title_clean))

            val_data_features = self.vectorized.transform(title_clean)
            np.asarray(val_data_features)

            return val_data_features
        else:
            return None

    def predict(self, title):
        if self.model is not None:
            predict = self.model.predict(title)
            return predict[0]
        else:
            return None

    def confusion_matrix(self):
        desc = np.asarray(self.test['title'].values.astype('U'))
        predict = self.model.predict(self.title_cleaner(desc))
        cm = confusion_matrix(self.test['label'], predict[:-1])
        ac = accuracy_score(self.test['label'], predict[:-1])

        return json.dumps({
            "conf_matrix" : cm.tolist(),
            "accuracy": ac
        })

    def classification_report(self):
        desc = np.asarray(self.test['title'].values.astype('U'))
        predict = self.model.predict(self.title_cleaner(desc))
        cm = classification_report(self.test['label'], predict[:-1], output_dict=True)

        return json.dumps({
            "classification_report": cm,
        })
