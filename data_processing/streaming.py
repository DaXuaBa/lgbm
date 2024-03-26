import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from configparser import ConfigParser
from sentiment import predict

conf_file_path = "/home/luongdb123/lgbm/data_processing/"
conf_file_name = conf_file_path + "stream_app.conf"
config_obj = ConfigParser()
config_read_obj = config_obj.read(conf_file_name)

kafka_host_name = config_obj.get('kafka', 'host')
kafka_port_no = config_obj.get('kafka', 'port_no')
input_kafka_topic_name = config_obj.get('kafka', 'input_topic_name')
kafka_bootstrap_servers = kafka_host_name + ':' + kafka_port_no

if __name__ == "__main__":
    print("Real-Time Data Processing Application Started ...")
    print(time.strftime("%Y-%m-%d %H:%M:%S"))

    spark = SparkSession \
        .builder \
        .appName("SentimentAnalysis") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    tweet_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", input_kafka_topic_name) \
        .option("startingOffsets", "latest") \
        .load()
    print("Printing Schema from Apache Kafka: ")
    tweet_df.printSchema()

    tweet_df1 = tweet_df.selectExpr("CAST(value AS STRING)", "timestamp")

    tweet_schema = StructType() \
        .add("created_at", StringType()) \
        .add("tweet_id", StringType()) \
        .add("tweet", StringType())

    tweet_df2 = tweet_df1\
        .select(from_json(col("value"), tweet_schema)\
        .alias("data"), "timestamp")

    tweet_df3 = tweet_df2.select("data.*", "timestamp")

    tweet_df3 = tweet_df3.withColumn("partition_date", to_date("created_at"))
    tweet_df3 = tweet_df3.withColumn("partition_hour", hour(to_timestamp("created_at", 'yyyy-MM-dd HH:mm:ss')))

    tweet_agg_write_stream_pre = tweet_df3 \
        .writeStream \
        .trigger(processingTime='10 seconds') \
        .outputMode("update") \
        .option("truncate", "false")\
        .format("console") \
        .start()
    
    print("Printing Schema of Bronze Layer: ")
    tweet_df3.printSchema()

    predict_udf = udf(lambda text: predict(text), StringType())

    # Áp dụng hàm dự đoán cho cột "tweet" trong DataFrame
    tweet_df4 = tweet_df3.withColumn("sentiment", predict_udf("tweet"))

    tweet_agg_write_stream = tweet_df4 \
        .writeStream \
        .trigger(processingTime='10 seconds') \
        .outputMode("update") \
        .option("truncate", "false")\
        .format("console") \
        .start()

    print("Printing Schema of Sentiment: ")
    tweet_df4.printSchema()

    tweet_agg_write_stream.awaitTermination()

    print("Real-Time Data Processing Application Completed.")
