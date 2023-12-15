from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import lit, explode, regexp_replace

# spark_df = spark.sql("SELECT * FROM `spotify`.`default`.`k_core_50_edges`").drop('_rescued_data')
spark_df = spark.sql("SELECT * FROM `spotify`.`default`.`k_core_25_edges`").drop('_rescued_data')
# load df_track_info which contains unique integer track_num for each track
spark_track_df = spark.sql("SELECT * FROM `spotify`.`default`.`df_track_info`").drop('_rescued_data')
# join the two dataframes on track_name and only keep pid and track_num
spark_df = spark_df.join(spark_track_df, spark_df.track_name == spark_track_df.track_name, 'inner')

# drop the columns that are not needed
spark_df = spark_df.drop('track_name', 'trackid', 'artist_name', 'artistid')
# convert track_num and pid to integer
spark_df = spark_df.withColumn("track_num", spark_df["track_num"].cast("int"))
spark_df = spark_df.withColumn("pid", spark_df["pid"].cast("int"))

# add column count for implcit ALS
implicit_data = spark_df.withColumn("count", lit(1))
implicit_data = implicit_data.withColumn("count", implicit_data["count"].cast("double"))
implicit_data = implicit_data.na.drop()

# split the data into training and test set
(training, test) = implicit_data.randomSplit([0.8, 0.2], seed=1234)
# cache the data for faster training
training = training.cache()

als = ALS(maxIter=25, rank=50, regParam=0.1,userCol="pid", itemCol="track_num", 
          ratingCol="count", implicitPrefs=True, nonnegative=True, coldStartStrategy="drop")
model = als.fit(training)

# generate recommendations for all users
userRecs = model.recommendForUserSubset(test, 500)
df = userRecs.select('pid', 'recommendations.track_num')
# remove the square brackets
df = df.withColumn("track_num", explode("track_num"))

# get the hit rate
count = 0
for row in test.collect():
    pid = row.pid
    track_num = row.track_num
    if df.where((df.pid == pid) & (df.track_num == track_num)).count() > 0:
        count += 1
all = test.count()
# print hit rate
print("Hit Rate: %f" % (count/all))

# get the rmse
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                predictionCol="prediction")
print("RMSE: %f" % (evaluator.evaluate(predictions)))