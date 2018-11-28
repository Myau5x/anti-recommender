import pyspark as ps
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import pandas as pd

spark = (ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("yelp_academic")
        .getOrCreate()
        )
sc = spark.sparkContext
#biz = spark.read.json('yelp_dataset/yelp_academic_dataset_business.json')

#rev = spark.read.json('yelp_dataset/yelp_academic_dataset_review.json')

#user = spark.read.json('yelp_dataset/yelp_academic_dataset_user.json')


def str_to_l(text):
    if text is None:
        return []
    else:
        return [word.strip() for word in text.split(',') ]

def if_restaurant(text):
    if text is None:
        return False
    else:
        return 'Restaurants' in text

if_rest_udf = udf(if_restaurant, BooleanType())
str_to_list_udf = udf(str_to_l, ArrayType(StringType()))
hash_udf = udf(lambda x: hash(x), IntegerType())

def filter_rev_by_rest(rev, biz):
    rests = biz.filter(if_rest_udf(biz.categories))

def look_how_dence_rating_matrix(rev, m):
    rating = rev.select('user_id', 'business_id', 'stars')
    user_id = rating.select('user_id').groupBy('user_id').count().selectExpr('user_id', 'count AS user_count')

    filt_rat = user_id.filter(user_id.user_count>m).join(rating, 'user_id')
    u  = user_id.filter(user_id.user_count>m).count()
    v = filt_rat.select('business_id').groupBy('business_id').count().count()
    r = filt_rat.count()
    k = r/(u+v)
    return u, v, r, k


def construct_set_for_ALS(rating, m):
    u_id = rating.select('user_id').groupBy('user_id').count().selectExpr('user_id', 'count AS user_count')
#    u_df = u_id.toPandas()
#    u_df['user_num'] = u_df.index
#    b_df = rating.select('business_id').groupBy('business_id').count().toPandas()
#    b_df['biz_num'] = b_df.index
#    user_id = spark.createDataFrame(u_df)
#    b_id = spark.createDataFrame(b_df)
    r1 = rating.join(u_id, 'user_id')
    r2 = r1.filter(r1.user_count > m)
    return r2



if __name__ == '__main__':
    pass
