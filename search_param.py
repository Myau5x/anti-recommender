import numpy as np
import pandas as pd

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

from nlp_cl_start import data_tokenizer, if_rest_udf
from pipe_spar import api_f, cluster_biz_by_review , cluster_user_by_review
from my_metr import transform_to_score, my_scorer, transform_aggregated
from sklearn.metrics import accuracy_score, recall_score



import pyspark as ps
spark = (ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("yelp_academic")
        .getOrCreate()
        )
sc = spark.sparkContext
seed = 91

##load dataset


def create_test_set(both, pipe_model, rest_rev):


    user_with_cl = cluster_user_by_review(both, pipe_model)
    biz_with_cl = cluster_biz_by_review(both, pipe_model)
    train_rev_id =both.select('review_id')

    known_rev2 = user_with_cl.join(rest_rev.select('business_id',
    'review_id',
    'stars',
    'user_id',
    'rating'), 'user_id').join(biz_with_cl, 'business_id')
    new_t = known_rev2.join(train_rev_id, 'review_id','left_anti' )

    new_t = new_t.withColumn('similar', (new_t.user_cl == new_t.biz_cl).cast("int"))
    regroup = new_t.groupBy('review_id').agg({'rating': 'mean', 'stars':'mean', 'similar':'sum' })
    new_grf = regroup.toPandas()
    new_grf = transform_aggregated(new_grf)
    return new_grf


if __name__ == '__main__':
    biz = spark.read.json('yelp_dataset/yelp_academic_dataset_business.json')
    rev = spark.read.json('yelp_dataset/yelp_academic_dataset_review.json')

    rests = biz.filter(if_rest_udf(biz.categories))

    rest_rev = rev.join(rests.select('business_id','stars').withColumnRenamed('stars','rating'),'business_id')
    bad_reviews = rest_rev.filter('stars < 3')

    bad_sample = bad_reviews.sample(False, 0.127, seed =seed)

    bad_sample.cache()
    sample_token= data_tokenizer(bad_sample)

    splits = sample_token.randomSplit([0.8, 0.1, 0.1], seed = seed)
    train = splits[0]
    add_cl = splits[1]
    test = splits[2]

    params_kcv = {'minDF' : 10, 'vocabSize':5000, 'k':15}

    cv = CountVectorizer(minDF=params_kcv['minDF'],
        vocabSize=params_kcv['vocabSize'], inputCol='token', outputCol='vectors')
    km1 = KMeans(k = params_kcv['vocabSize'], featuresCol='vectors', maxIter= 30)
    pipe_count = Pipeline(stages=[cv, km1])

    idf = IDF(inputCol="vector", outputCol="features")
    km2 = KMeans(k = 20, featuresCol='features', maxIter= 30)
    pipe_idf = Pipeline(stages = [cv, idf, km2])

    both = train.union(add_cl)

    count_model = pipe_count.fit(train)

    new_grf = create_test_set(both, count_model, rest_rev)

    results = my_scorer(new_grf)

    idf_model = pipe_idf.fit(train)

    new_gr_idf = create_test_set(both, idf_model, rest_rev)
    results_idf = my_scorer(new_gr_idf)
