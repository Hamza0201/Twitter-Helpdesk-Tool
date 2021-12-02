import tweepy
import json
import pandas as pd
from tabulate import tabulate
from sqlalchemy import create_engine
from geopy.geocoders import Nominatim
import gmplot
import re
from string import punctuation
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import emoji
import sys
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
import joblib
from textblob import Word
import collections
from nltk import bigrams
import itertools
import networkx as nx
import csv
import schedule
import time
from datetime import datetime
import os
import pathlib
from s_functions import processTweet, textProcessing, text_process, processTicket
import pyrebase
import datetime
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize


with open("twitter_key.json", "r") as file:
	creds = json.load(file)

auth = tweepy.OAuthHandler(creds["CONSUMER_KEY"], creds["CONSUMER_SECRET"])
auth.set_access_token(creds["ACCESS_TOKEN"], creds["ACCESS_TOKEN_SECRET"])

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

try:
	api.verify_credentials()
	print("Authentication OK")
except:
	print("Error during authentication")

user = api.search("AppleSupport -filter:retweets", lang = "en", tweet_mode='extended', count=200)

tweet = {'t_id': [], 'u_id' : [], 'screen_name' : [], 'date': [], 'text': [], 'likes': [], 'retweets' : [], 'hashtags' : [], 'user_location' : [], 'ticket_status' : []}
users = {'u_id': [], 'followers_count' : [], 'default_profile' : []}

firebaseConfig = {'apiKey': "AIzaSyC2WymgSdxypUoy49udAFTyyvTXPyc1nNc",
	'authDomain': "twitter-d34a5.firebaseapp.com",
	'databaseURL': "https://twitter-d34a5-default-rtdb.europe-west1.firebasedatabase.app",
	'projectId': "twitter-d34a5",
	'storageBucket': "twitter-d34a5.appspot.com",
	'messagingSenderId': "513871312680",
	'appId': "1:513871312680:web:291286d02f7cd0d367821d",
	'measurementId': "G-4XEWTVF57Y"}

firebase = pyrebase.initialize_app(firebaseConfig)

db=firebase.database()

#auth=firebase.auth()
#storage=firebase.storage()
#user_id=1339945975289409538

db.child('live_counter').set({'open_tickets' : 0, 'processing_tickets' : 0, 'closed_tickets' : 0})
db.child('tweets').remove();
db.child('sentiment_recorder').remove();
db.child('ticket_recorder').remove();

def ner(tweet):
	nltk.internals.config_java('C:/Program Files/Java/jdk-15.0.1/bin/java.exe')
	java_path = "C:/Program Files/Java/jdk-15.0.1/bin/java.exe"
	os.environ['JAVAHOME'] = java_path
	st = StanfordNERTagger('./stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   './stanford-ner/stanford-ner.jar',
					   encoding='utf-8')

	text = tweet

	tokenized_text = word_tokenize(text)
	classified_text = st.tag(tokenized_text)

	named_entity = [list(elem) for elem in classified_text]

	for index,element in enumerate(named_entity):
		try:
			if named_entity[index][1] == "PERSON" and named_entity[index + 1][1] == "PERSON":
				named_entity[index + 1][0] = named_entity[index][0] + '-' + named_entity[index+1][0]
				named_entity[index][1] = 'Combined'
			elif named_entity[index][1] == 'ORGANIZATION' and named_entity[index + 1][1] == 'ORGANIZATION':
				named_entity[index + 1][0] = named_entity[index][0] + '-' + named_entity[index+1][0]
				named_entity[index][1] = 'Combined'
			elif named_entity[index][1] == 'LOCATION' and named_entity[index + 1][1] == 'LOCATION':
				named_entity[index + 1][0] = named_entity[index][0] + '-' + named_entity[index+1][0]
				named_entity[index][1] = 'Combined'
		except IndexError:
			break

	filter_list = ['DOC_NUMBER','PERSON','LOCATION','ORGANIZATION']
	entityWordList = [element for element in named_entity if any(i in element for i in filter_list)]

	dict_entities = {'PERSON' : [], 'LOCATION' : [], 'ORGANIZATION' : []}

	for every_entity in entityWordList:
		if every_entity[1] == 'PERSON':
			dict_entities['PERSON'].append(every_entity[0])
		elif every_entity[1] == 'LOCATION':
			dict_entities['LOCATION'].append(every_entity[0])
		else:
			dict_entities['ORGANIZATION'].append(every_entity[0])

	return dict_entities

#for status in user:
#	if not(status.in_reply_to_status_id):
#
#		tweets1 = {}
#		tweets1['t_id'] = status.id_str
#		tweets1['u_id'] = status.user.id
#		tweets1['screen_name'] = status.user.screen_name
#		tweets1['date'] = status.created_at.strftime('%d-%m-%Y')
#		tweets1['time'] = status.created_at.strftime('%H:%M:%S')
#		tweets1['text'] = status.full_text
#		tweets1['processedtext'] = text_process(processTweet(status.full_text))
#		tweets1['processedticket'] = processTicket(status.full_text)
#		tweets1['likes'] = status.favorite_count
#		tweets1['ner'] = ner(status.full_text)
#		tweets1['topic'] = 'N/A'
#		tweets1['retweets'] = status.retweet_count
#		tweets1['hashtags'] = [d['text'] for d in status.named_entity.get('hashtags', {}) if 'text' in d]
#		tweets1['sentimence'] = TextBlob(status.full_text).sentiment.polarity
#		tweets1['user_location'] = status.user.location
#		tweets1['ticket_status'] = 'Open'
#		db.child('tweets').push(tweets1)

#df_tweets = textProcessing(pd.DataFrame(tweet))

#df_tweets.to_csv(r'C:\Users\Hamza\Documents\Project\tweets.csv', index = False)

users = db.child('tweets').get()

class MyStreamListener(tweepy.StreamListener):

	def on_status(self, status):
		if(hasattr(status, 'retweeted_status') == False):
			if not status.truncated:
				text = status.text
			else:
				text = status.extended_tweet['full_text']
			if not(status.in_reply_to_status_id):
				tweets1 = {}
				tweets1['t_id'] = status.id
				tweets1['u_id'] = status.user.id
				tweets1['screen_name'] = status.user.screen_name
				tweets1['date'] = status.created_at.strftime('%d-%m-%Y')
				tweets1['time'] = status.created_at.strftime('%H:%M:%S')
				tweets1['text'] = text
				tweets1['processedtext'] = text_process(processTweet(text))
				tweets1['processedticket'] = processTicket(text)
				tweets1['ner'] = ner(text)
				tweets1['topic'] = 'N/A'
				tweets1['likes'] = status.favorite_count
				tweets1['retweets'] = status.retweet_count
				tweets1['hashtags'] = [d['text'] for d in status.entities.get('hashtags', {}) if 'text' in d]
				tweets1['sentimence'] = TextBlob(text).sentiment.polarity
				tweets1['user_location'] = status.user.location
				tweets1['ticket_status'] = 'Open'
				db.child('tweets').push(tweets1)
		
	def on_error(self, status_code):
		print(status_code)
		return False


def beg_stream(api):
	myStreamListener = MyStreamListener()
	stream = tweepy.Stream(auth=api.auth, listener=myStreamListener)

	try:
		print('Start streaming.')
		stream.filter(track=['TwitterSupport'], languages = ["en"])
	except KeyboardInterrupt:
		print("Stopped.")
	finally:
		print('Done.')
		stream.disconnect()

beg_stream(api)


#print(df_tweets)
#df_tweets = textProcessing(pd.DataFrame(tweet))
#df_tweets.ticket_status = pd.Categorical(df_tweets.ticket_status,categories=['open','closed'],ordered=True)
#df_tweets.sort_values(by=['ticket_status'])
#print(df_tweets)
#df_tweets.to_csv(r'C:\Users\Hamza\Documents\Project\tweets.csv', index = False)
#df_users.to_csv(r'C:\Users\Hamza\Documents\Project\users.csv', index = False)


