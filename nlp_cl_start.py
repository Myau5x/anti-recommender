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

class preparation():

    def __init__(self, minDF=10, vocabSize=5000, inputCol='token', outputCol='features'   ):
        self.udf_tokenize = udf_tokenize
        self.minDF = minDF
        self.vocabSize = vocabSize
        self.inputCol = inputCol
        self.outputCol = outputCol

    def tokenize(self, dataset, textCol = 'text'):
        '''can tokenize'''
        return dataset.withColumn('token', udf_tokenize('text'))

    def fit(self, dataset):
        cv = CountVectorizer(minDF = self.minDF, vocabSize = self.vocabSize,
         inputCol = self.inputCol, outputCol = 'vectors')
        self.m_cv = cv.fit(dataset)
        idf = IDF(inputCol= 'vectors', outputCol= 'features')
        medium = self.m_cv.transform(dataset)
        medium.cache()
        self.m_tfidf = idf.fit(medium)
        return self.m_tfidf

    def transform(self, dataset):

        return self.m_tfidf.transform(dataset)

def kmean_counts(sample, k = 2, minDF=10, vocabSize=5000,
    inputCol='token', outputCol='vectors'):
    cv = CountVectorizer(minDF=minDF, vocabSize=vocabSize,
        inputCol='token', outputCol='vectors')
    model = cv.fit(sample)
    sample_vect = model.transform(sample)
    sample_vect.cache()

    km = KMeans(k = k, featuresCol='vectors', maxIter= 30)
    model_km = km.fit(sample_vect)
    return model_km
