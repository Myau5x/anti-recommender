import numpy as np


import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, BooleanType
from pyspark.ml import Pipeline

from nlp_cl_start import data_tokenizer



cv = CountVectorizer(minDF=10, vocabSize=5000, inputCol='token', outputCol='vectors')
km1 = KMeans(k = 20, featuresCol='vectors', maxIter= 30)


pipe_count = Pipeline(stages=[cv, km1])


idf = IDF(inputCol="vector", outputCol="features")
km2 = KMeans(k = 20, featuresCol='features', maxIter= 30)
pipe_idf = Pipeline(stages = [cv, idf, km2])
