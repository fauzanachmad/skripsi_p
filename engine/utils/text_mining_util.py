import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


def basic_clean(description):
    lowercase_sentence = re.sub("[^a-zA-Z]", " ", str(description))
    lowercase_sentence = lowercase_sentence.lower()

    lowercase_sentence = lowercase_sentence.translate(str.maketrans("", "", string.punctuation))

    lowercase_sentence = lowercase_sentence.strip()

    lowercase_sentence = re.sub('\s+', ' ', lowercase_sentence)
    tokens = nltk.tokenize.word_tokenize(lowercase_sentence)
    stops = set(stopwords.words("indonesian"))

    meaningful_words = [w for w in tokens if not w in stops]

    return ",".join(meaningful_words)
