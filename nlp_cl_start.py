import numpy as np

def print_cl(arr):
    s = ' '
    for x in arr:
        s = s.join(x)
        s+=' '
    print(s)


import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, BooleanType

PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))

def tokenize(text):
    regex = re.compile('<.+?>|[^a-zA-Z]')
    clean_txt = regex.sub(' ', text)
    tokens = clean_txt.split()
    lowercased = [t.lower() for t in tokens]

    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]

    STEMMER = PorterStemmer()
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed if w]

udf_tokenize = udf(f=tokenize, returnType=ArrayType(StringType()))
#bad_sample = bad_sample.withColumn('token', udf_tokenize('text'))
#cv = CountVectorizer(minDF=10, vocabSize=5000, inputCol='token', outputCol='vectors')

def if_restaurant(text):
    if text is None:
        return False
    else:
        return 'Restaurants' in text

if_rest_udf = udf(if_restaurant, BooleanType())

def data_tokenizer(dataset, colText = 'text', colToken = 'token'):
    return dataset.withColumn('token', udf_tokenize('text'))
