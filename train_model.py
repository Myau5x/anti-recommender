import pyspark as ps
spark = (ps.sql.SparkSession.builder
        .master("local[4]")
        .appName("yelp_academic")
        .getOrCreate()
        )
sc = spark.sparkContext
