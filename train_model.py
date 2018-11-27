import pyspark as ps
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.sql.functions import udf

spark = (ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("yelp_academic")
        .getOrCreate()
        )
sc = spark.sparkContext

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
