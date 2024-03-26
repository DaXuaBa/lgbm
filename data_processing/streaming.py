import re
import pickle
import joblib
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
download('wordnet')

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

mystopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

stopwordlist = stopwords.words('english') + mystopwordlist

def preprocess(textdata):
    processedText = []
    wordLemma = WordNetLemmatizer()
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)" 
    userPattern       = '@[^\s]+' # e.g @FagbamigbeK check this out
    alphaPattern      = "[^a-zA-Z0-9]" # e.g I am *10 better!
    sequencePattern   = r"(.)\1\1+"  # e.g Heyyyyyyy, I am back!
    seqReplacePattern = r"\1\1" # e.g Replace Heyyyyyyy with Heyy
    
    for tweet in textdata:
        tweet = tweet.lower()
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet) 
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])  
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)  
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet) # e.g I am *10 better!
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet) # e.g Replace Heyyyyyyy with Heyy
         
        tweetwords = ''
        for word in tweet.split():
            if len(word) > 2 and word.isalpha():
                word = wordLemma.lemmatize(word)
                tweetwords += (word + ' ')
        processedText.append(tweetwords)
    return processedText

def load_model():
    vectoriser = joblib.load('tfidf_vectoriser.pkl')
    with open('Sentiment-LightGBM.pickle', 'rb') as file:
        LGBMmodel = pickle.load(file)
    return vectoriser, LGBMmodel

vectoriser, LGBMmodel = load_model()

def preprocess_text(text):
    processed_text = preprocess(text)
    return processed_text

def predict(text):
    processed_text = preprocess_text(text)
    textdata = vectoriser.transform(processed_text)
    sentiment = LGBMmodel.predict(textdata)
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
    # 0 is Negative, 1 is Positive
    df = pd.DataFrame(data, columns = ['text','sentiment'])

    return df

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from configparser import ConfigParser

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
