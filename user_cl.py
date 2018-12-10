from pyspark.ml.clustering import KMeans, KMeansModel, KMeansSummary
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel, IDF, IDFModel
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, BooleanType, StructType, StructField
from pyspark.ml import Pipeline, PipelineModel

from nlp_cl_start import data_tokenizer, tokenize

from scraping.add_by_scrap import scrap_by_users

from pipe_spar import cluster_user_by_review

import pyspark as ps
spark = (ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("yelp_academic")
        .getOrCreate()
        )
sc = spark.sparkContext


pipe = PipelineModel.load('pipe-idf-5-7-18/')

def new_user_clust_from_yelp(reviews):
    df = spark.createDataFrame(reviews)
    df = df.filter((df.rating == '2.0')|(df.rating == '1.0'))
    df = data_tokenizer(df)
    clusters = cluster_user_by_review(df, pipe)
    return clusters

def transform_users(cls):
    a = []
    for cl in cls:
        a.append('cl_t'+str(cl[1]))
    return a
#########
if (__name__ == "__main__"):
#    spark = ps.sql.SparkSession.builder \
#                .master("local[4]") \
#                .appName("df lecture") \
#                .getOrCreate()


    kk= scrap_by_users('https://www.yelp.com/user_details?userid=KbU6Q0oy5X2mTShnRXiggg')
    cl = new_user_clust_from_yelp(kk)

    transform_users(cl.collect())

    r1 = """This review is based only on the drinks and service.

    Our group got seated at a booth. The decor and ambiance was pretty cool and trendy. When the server came to take our drink order, I had asked her what the Whiteclaw drink was listed under the Beer, Wine, etc. section of the menu. She had told me that it was like a cider that was a little sweet with a hint of grapefruit. That piqued my interest so I ordered that.

    When she brought our drinks, she placed a really tall can in front of me. I was pretty surprised and wasn't expecting a can, but it was fine. When I took my first sips, I barely tasted anything. I turned the can around to see the front side and noticed at the bottom it said, "Sparkling water with a hint of grapefruit".

    I was pretty confused and bummed out. I wasn't really sure why she would tell me that the Whiteclaw drink was cider. Cider and sparkling water are two very different drinks.

    I kept sipping and just tried to enjoy it since it was a huge $10 can. But after a few minutes, I decided to order some wine so I can actually enjoy myself. When a different server came to check up on us, my boyfriend asked if I could get the Anderson Valley Rosé. The server told us that that isn't actually a rosé but a completely different kind of drink. It was confusing, but they do actually have a rosé on the menu, which we ordered, but it's not the Anderson Valley drink listed on the menu.

    When our original server stopped by, she noticed I had the rosé and wasn't drinking my Whiteclaw sparkling water. She asked, "Did you not like it?". I said, "no, not really." And all she said was, "Well I like it." and walked away. Had she told me that it was sparkling water, I definitely wouldn't have ordered that and waste $10.

    Maybe this was just one of those off-chance, bad experiences. Maybe the food is good? The space is definitely pretty cool to hang with friends after dinner or for happy hour. Not sure if I'd return. There's a lot of other bars that have less confusing drink menus and better service that I would prefer going to.
    """
    token = tokenize(r1)
    rw_df = spark.createDataFrame([{'token': token, 'user_id' :0},{'token': token, 'user_id' :0},{'token': token, 'user_id' :0} ])
    g = cluster_user_by_review(rw_df, pipe)

    transform_users(g.collect)
