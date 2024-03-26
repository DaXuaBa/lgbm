import re
import pickle
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
download('wordnet')

vectoriser = joblib.load('tfidf_vectoriser.pkl')

with open('Sentiment-LightGBM.pickle', 'rb') as file:
    LGBMmodel = pickle.load(file)

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

def predict(vectoriser, model, text):
    textdata = vectoriser.transform(preprocess(text)) 
    sentiment = model.predict(textdata)
    
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

# if __name__=="__main__": 
#     # Text to classify should be in a list.
#     text = ["Today is so great!",
#             "May the Good Lord be with you.", "I hate peanuts!",
#             "Mr. Kehinde, what are you doing next? this is great!"]
    
#     df = predict(vectoriser, LGBMmodel, text)
#     print(df.head())

# from kafka import KafkaConsumer
# import json
# consumer = KafkaConsumer('test', bootstrap_servers=['leesin.click:9092'])

# # Đọc dữ liệu từ Kafka và gọi hàm predict
# for message in consumer:
    
#     data = json.loads(message.value)
#     text_to_predict = data['tweet'] 
    
#     # Gọi hàm predict để phân loại sentiment cho tin nhắn
#     result_df = predict(vectoriser, LGBMmodel, [text_to_predict])
    
#     # Xử lý kết quả ở đây (ví dụ: lưu vào cơ sở dữ liệu, gửi đi qua Kafka producer, in ra màn hình, v.v.)
#     print(result_df.head())

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, udf
from pyspark.sql.types import StructType, StringType, TimestampType

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .getOrCreate()

# Định nghĩa schema cho dữ liệu JSON
schema = StructType() \
    .add("created_at", TimestampType()) \
    .add("tweet_id", StringType()) \
    .add("tweet", StringType())

# Đọc dữ liệu từ Kafka vào DataFrame
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "leesin.click:9092") \
    .option("subscribe", "test") \
    .load()
  
# Parse dữ liệu JSON từ Kafka
parsed_df = kafka_df \
    .selectExpr("CAST(value AS STRING)") \
    .select(from_json("value", schema).alias("data")) \
    .select("data.*")

predict_udf = udf(lambda tweet: predict(vectoriser, LGBMmodel, [tweet]), StringType())

# Gọi hàm predict để phân loại sentiment cho mỗi tin nhắn
result_df = parsed_df \
    .select("tweet") \
    .withColumn("sentiment", predict_udf("tweet"))

# In kết quả ra màn hình
query = result_df \
    .writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

# Chờ cho query kết thúc
query.awaitTermination()
