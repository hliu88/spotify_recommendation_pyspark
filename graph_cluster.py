import sys
import random
from graphframes import *
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, count
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# reduce a list of dataframes to a single dataframe
def pairwise_reduce(operation, x):
    while len(x) > 1:
        sys.stdout.flush()
        sys.stdout.write("%d" % (len(x)))

        # split the list into pairs
        lst = [operation(i, j) for i, j in zip(x[::2], x[1::2])]
        # if the list has an odd number of elements, perform 
        # the operation on the last element in x with last element in lst
        if len(x) > 1 and len(x) % 2 == 1:
            lst[-1] = operation(lst[-1], x[-1])
        x = lst
    return x[0]

# get the neighbors of a playlist
def get_neighbors_playlists(user_neighbour, Graph):    
    result_df = spark.createDataFrame([], user.schema)

    dfs_to_union = []
    # convert dataframe to list
    iter_user_neighbors_list = user_neighbors.collect()
    for track in iter_user_neighbors_list:
        # get the number of playlists that contain the track
        track_neighbors = Graph.edges.filter(f"dst = {track[0]}").select("src").distinct()
        dfs_to_union.append(track_neighbors)

    result_df = pairwise_reduce(DataFrame.unionAll, dfs_to_union)
    # get the playlists that appear more than 15 times
    result_df = result_df.groupBy("id").count().filter(col("count") > 15).select("id")
    return result_df

# populate the nearest neighbor playlists by filtering the dataframe
def populate_playlists(input):
    iter_result_list = input.collect()
    count = 0
    dfs_to_union = []
    for id in iter_result_list:
        if count % 1000 == 0:
            sys.stdout.flush()
            sys.stdout.write("%d   \r" % (count) )
        temp = df.filter(col("pid") == id[0])
        dfs_to_union.append(temp)

        count += 1
    return dfs_to_union

# spark_df = spark.sql("SELECT * FROM `spotify`.`default`.`k_core_50_edges`").drop('_rescued_data')
spark_df = spark.sql("SELECT * FROM `spotify`.`default`.`k_core_25_edges`").drop('_rescued_data')
# load df_track_info which contains unique integer track_num for each track
spark_track_df = spark.sql("SELECT * FROM `spotify`.`default`.`df_track_info`").drop('_rescued_data')
# join the two dataframes on track_name and only keep pid and track_num
spark_df = spark_df.join(spark_track_df, spark_df.track_name == spark_track_df.track_name, 'inner')
df = spark_df.select('pid', 'track_num')

# remove playlists with more than 75 tracks
grouped_df = df.groupBy('pid').agg(count('*').alias('row_count'))
filtered_df = grouped_df.filter(grouped_df['row_count'] <= 75)
result_df = spark_df.join(filtered_df, on='pid', how='inner')

df = result_df.select('pid', 'track_num')

# convert pid and track_num to integer
df = df.withColumn("track_num", df["track_num"].cast("int"))
df = df.withColumn("pid", df["pid"].cast("int"))
df = df.na.drop()

# get the unique playlists and tracks
user = df.select('pid').distinct().withColumnRenamed('pid', 'id')
track = df.select('track_num').distinct().withColumnRenamed('track_num', 'id')

# create edges
user_track = df.select('pid', 'track_num').withColumnRenamed('pid', 'src').withColumnRenamed('track_num', 'dst')

# create graph
vertices = user.union(track)
edges = user_track
G = GraphFrame(vertices, edges)

# randomly select 100 playlists for evaluation
user_lst = [row.id for row in user.collect()]
random_user_lst = random.sample(user_lst, 100)

rmse_lst = []
hit_rate_lst = []

for user in random_user_lst:
    user_neighbors = G.edges.filter(f"src = {user}").select("dst").distinct()

    # randomly select 80% of the neighbors for training
    train_user_neighbor = user_neighbors.sample(False, 0.8, seed=0)
    test_user_neighbor = user_neighbors.subtract(train_user_neighbor)

    # get the playlists that contain the neighbors
    train_neighbors_playlists = get_neighbors_playlists(train_user_neighbor, G)
    test_neighbors_playlists = get_neighbors_playlists(test_user_neighbor, G)

    # get the tracks in the playlists
    train_playlists = populate_playlists(train_neighbors_playlists)
    test_playlists = populate_playlists(test_neighbors_playlists)

    # reduce the list of dataframes to a single dataframe
    train = pairwise_reduce(DataFrame.unionAll, train_playlists)
    test = pairwise_reduce(DataFrame.unionAll, test_playlists)

    train = train.withColumn("count", lit(1))
    test = test.withColumn("count", lit(1))
    train = train.cache()

    als = ALS(maxIter=20, rank=10, regParam=0.1,userCol="pid", itemCol="track_num", 
            ratingCol="count", implicitPrefs=True, nonnegative=True, coldStartStrategy="drop")
    model = als.fit(train)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                    predictionCol="prediction")
    rmse_lst.append(evaluator.evaluate(predictions))

    # calculate hit rate
    userRecs = model.recommendForAllUsers(50)
    iter = userRecs.filter(f"pid = {user}").select("recommendations.track_num").collect()[0]
    test_neighbor = test_user_neighbor.collect()
    test_neighbor_lst = [row.dst for row in test_neighbor]
    hit_rate = 0
    for track in iter[0]:
        if track in test_neighbor_lst:
            hit_rate += 1
    hit_rate_lst.append(hit_rate / test_user_neighbor.count())

print(sum(rmse_lst) / len(rmse_lst))
print(sum(hit_rate_lst) / len(hit_rate_lst))