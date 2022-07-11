import os
import pickle

import nltk
import numpy as np

nltk.download('stopwords')
from engine.utils.text_mining_util import basic_clean


class Testing:
    def __init__(self):
        self.vectorized = None
        self.model = None

        self.pickle_parser()

    def pickle_parser(self):
        if os.path.exists("resources/models/vectorized.pickle") & os.path.exists("resources/models/model.pickle"):
            self.vectorized = pickle.load(open("resources/models/vectorized.pickle", 'rb'))
            self.model = pickle.load(open("resources/models/model.pickle", 'rb'))

    def title_cleaner(self, title):
        if self.vectorized is not None:
            title_clean = title
            title_clean.append(basic_clean(title_clean))

            val_data_features = self.vectorized.transform(title)
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
