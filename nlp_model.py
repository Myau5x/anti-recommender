import numpy as np

import pandas as pd
import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from pyspark.ml.clustering import KMeans, KMeansModel, KMeansSummary
from pyspark.ml.feature import CountVectorizer, IDF, CountVectorizerModel, IDFModel
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, BooleanType
from pyspark.ml import Pipeline, PipelineModel

from nlp_cl_start import data_tokenizer
from src.pipe_spar import api_f, cluster_biz_by_review , cluster_user_by_review
from src.nlp_cl_start import if_rest_udf

import pyspark as ps
from src.my_metr import transform_to_score, my_scorer, transform_aggregated
from sklearn.metrics import accuracy_score, recall_score
spark = (ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("yelp_academic")
        .getOrCreate()
        )
sc = spark.sparkContext

### load data
biz = spark.read.json('yelp_dataset/yelp_academic_dataset_business.json')
rev = spark.read.json('yelp_dataset/yelp_academic_dataset_review.json')



##filter data
def train_idf_model():
    rests = biz.filter(if_rest_udf(biz.categories))

    rest_rev = rev.join(rests.select('business_id','stars').withColumnRenamed('stars','rating'),'business_id')
    bad_reviews = rest_rev.filter('stars < 3')

    #sample for train

    bad_sample = bad_reviews.sample(False, 0.127, seed =91)
    sample_token= data_tokenizer(bad_sample)
    splits = sample_token.randomSplit([0.8, 0.1, 0.1], seed = 91)

    train = splits[0]
    add_cl = splits[1]
    test = splits[2]

    cv = CountVectorizer(minDF=5, vocabSize=5000, inputCol='token', outputCol='vectors')
    idf = IDF(minDocFreq=7, inputCol="vectors", outputCol="features")
    km2 = KMeans(k = 18, featuresCol='features', maxIter= 30)
    pipe_idf = Pipeline(stages = [cv, idf, km2])

    pipe_idf_model = pipe_idf.fit(train)
    return pipe_idf

pipe_idf = train_idf_model()
user_with_cl = cluster_user_by_review(both, pipe_idf_model)
biz_with_cl = cluster_biz_by_review(both, pipe_idf_model)

##transform test set, filter it
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
### looking on results
my_scorer(new_grf)
