from train_model import *
from pyspark.ml.recommendation import ALS
import pandas as pd

biz = spark.read.json('yelp_dataset/yelp_academic_dataset_business.json')

rev = spark.read.json('yelp_dataset/yelp_academic_dataset_review.json')

user = spark.read.json('yelp_dataset/yelp_academic_dataset_user.json')

rests = biz.filter(if_rest_udf(biz.categories))
base_pred = rests.select('business_id', 'stars').withColumnRenamed('stars',
'mean_rating').join(rev.select('business_id', 'stars'), 'business_id')







splits = rev_dd.randomSplit([0.7,0.3], seed =91)
train = splits[0]
test = splits[1]

als = ALS(rank=10, regParam=0.1,
          itemCol='biz_num',
          userCol='user_num',
          ratingCol='stars')
mdl = als.fit(train)
