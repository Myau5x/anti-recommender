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

##modeling

cv = CountVectorizer(minDF=10, vocabSize=5000, inputCol='token', outputCol='vectors')
km1 = KMeans(k = 20, featuresCol='vectors', maxIter= 30)


pipe_count = Pipeline(stages=[cv, km1])


idf = IDF(inputCol="vector", outputCol="features")
km2 = KMeans(k = 20, featuresCol='features', maxIter= 30)
pipe_idf = Pipeline(stages = [cv, idf, km2])


###fitting
#train_vect = data_tokenizer(dataset)
#model_cv_km = pipe_count.fit(train_vect)

#model_tf_km = pipe_count.fit(train_vect)

def cluster_user_by_review(data_review, model):
    pred = model.transform(data_review)
    data = data_review.select('user_id', 'prediction').withColumnRenamed('prediction','user_cl')
    return data

def cluster_biz_by_review(data_review, model):
    pred = model.transform(data_review)
    data = data_review.select('bussiness_id', 'prediction').withColumnRenamed('prediction','biz_cl')
    return data

### Testing
