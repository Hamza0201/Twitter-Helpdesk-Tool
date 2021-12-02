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
import matplotlib
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
from nltk import ngrams
import itertools
import networkx as nx
import csv
import schedule
import time
from datetime import datetime, timedelta
import os
import pathlib
import pyrebase
from bokeh.plotting import figure, output_file, show
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.palettes import Spectral6, Category20c
from bokeh.transform import factor_cmap, cumsum
import math
from math import pi
import bokeh.layouts
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from textblob.sentiments import NaiveBayesAnalyzer
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import html
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
import imp

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

clouddb = firestore.client()

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

users = db.child('tweets').get()

auth=firebase.auth()

def processTweet(tweet):
	numbers = r'(?:(?:\d+,?)+(?:\.?\d+)?)'
	URL = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	html_tag = r'<[^>]+>'
	hash_tag = r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"
	at_sign = r'(?:@[\w_]+)'
	dash_quote = r"(?:[a-z][a-z'\-_]+[a-z])"
	other_word = r'(?:[\w_]+)'
	other_stuff = r'(?:\S)' # anything else - NOT USED
	start_pound = r"([#?])(\w+)" # Start with #
	start_quest_pound = r"(?:^|\s)([#?])(\w+)" # Start with ? or with #
	cont_number = r'(\w*\d\w*)' # Words containing numbers
	slash_all = r'\s*(?:[\w_]*[/\\](?:[\w_]*[/\\])*[\w_]*)'
	tweet = html.unescape(tweet)
	tweet = re.sub('&', "and",tweet)
	tweet = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", tweet)
	tweet = re.sub(URL, "", tweet)
	tweet = re.sub(html_tag, "",tweet)
	tweet = re.sub(hash_tag, "",tweet)
	tweet = re.sub(slash_all,"", tweet)
	tweet = re.sub(cont_number, "",tweet)
	tweet = re.sub(numbers, "",tweet)
	tweet = re.sub(start_pound, "",tweet)
	tweet = re.sub(start_quest_pound, "",tweet)
	tweet = re.sub(at_sign, "",tweet)
	tweet = re.sub("'", "",tweet)
	tweet = re.sub("`", "",tweet)
	tweet = re.sub("’", "",tweet)
	tweet = re.sub('"', "",tweet)
	tweet = re.sub(r'(?:^|\s)[@#].*?(?=[,;:.!?]|\s|$|£)', r'', tweet) # Removes # and @ in words (lookahead)
	tweet = emoji.get_emoji_regexp().sub(r'', tweet)
	return tweet

def processTicket(tweet):
	numbers = r'(?:(?:\d+,?)+(?:\.?\d+)?)'
	URL = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	html_tag = r'<[^>]+>'
	hash_tag = r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"
	at_sign = r'(?:@[\w_]+)'
	dash_quote = r"(?:[a-z][a-z'\-_]+[a-z])"
	other_word = r'(?:[\w_]+)'
	other_stuff = r'(?:\S)'
	start_pound = r"([#?])(\w+)"
	start_quest_pound = r"(?:^|\s)([#?])(\w+)"
	cont_number = r'(\w*\d\w*)'
	slash_all = r'\s*(?:[\w_]*[/\\](?:[\w_]*[/\\])*[\w_]*)'
	tweet = html.unescape(tweet)
	tweet = re.sub('&', "and",tweet)
	tweet = re.sub(URL, "", tweet)
	tweet = re.sub(html_tag, "",tweet)
	tweet = re.sub("'", "",tweet)
	tweet = re.sub("`", "",tweet)
	tweet = re.sub("’", "",tweet)
	tweet = re.sub('"', "",tweet)
	tweet = emoji.get_emoji_regexp().sub(r'', tweet)
	return tweet
	
def text_process(raw_text):
	raw_text = remove_stopwords(raw_text)

	nopunc = TextBlob(raw_text).words
	
	temp = [word.lemmatize() for word in nopunc.lower() if word.lower() not in stopwords.words('english')]

	temp = [w for w in temp if len(w) >2 ]

	joined = ' '.join(temp)

	return joined

def textProcessing(df):
	df['sentimence'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
	df['text'] = df['text'].apply(lambda x: processTweet(x))
	df['tokens'] = df['text'].apply(text_process)
	df['tokens'] = df['tokens'].apply(lambda x: [y.lemmatize() for y in x])
	df['text'] = df['tokens'].apply(lambda x: " ".join(x))
	df['nouns'] = df['text'].apply(lambda x: TextBlob(x).noun_phrases)
	return df

def gen_wordcloud():
	all_words = []

	for user in users:
		temp = user.val().get('processedtext')
		all_words.extend(TextBlob(temp).words)
		
	# create a word frequency dictionary
	wordfreq = Counter(all_words)

	# draw a Word Cloud with word frequencies
	wordcloud = WordCloud(width=800,
						  height=400,
						  max_words=500,
						  relative_scaling=0.5,
						  colormap='Blues',
						  normalize_plurals=True).generate_from_frequencies(wordfreq)

	plt.figure(figsize=(20,10))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()

def getLocation():
	apikey = 'AIzaSyCc2G01bScWagXqXlVoiRrOWigbBZZW6-4'
	geolocator = Nominatim(user_agent="u1813057.Project@gmail.com")
	latitude = []
	longitude = []
	locationsList = []
	for user in users:
		if (user.val().get('user_location') != ""):
			locationsList.append(user.val().get('user_location'))

	for user_loc in locationsList:
		try:
			location = geolocator.geocode(user_loc)
			
			# If coordinates are found for location
			if location:
				latitude.append(location.latitude)
				longitude.append(location.longitude)
				
		# If too many connection requests
		except:
			pass

	tile_provider = get_provider(CARTODBPOSITRON)

	# range bounds supplied in web mercator coordinates
	p = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
		   x_axis_type="mercator", y_axis_type="mercator", width=700,height=350)

	p.sizing_mode = 'scale_width'
	p.add_tile(tile_provider)

	k = 6378137

	longitude = [item * (k * math.pi/180.0) for item in longitude]

	latitude = [math.log(math.tan((90 + item) * math.pi/360.0)) * k for item in latitude]


	source = ColumnDataSource(data=dict(lat=(latitude), lon=(longitude)))

	p.circle(x="lon", y="lat", size=4, fill_color="tomato", fill_alpha=0.3, line_color="tomato", source=source)

	script1, div1, = components(p)
	cdn_js = CDN.js_files
	cdn_css = CDN.css_files

	show(p)


def df_word_freq(df_tweets):
	count_no_urls = " ".join(df_tweets.text.tolist())
	terms = collections.Counter(count_no_urls.split())
	df_most_common = pd.DataFrame(terms.most_common(15), columns=['words', 'count'])
	return df_most_common

def df_ngram_freq(df_tweets, number):
	count_no_urls = " ".join(df_tweets.text.tolist())
	hi2 = count_no_urls.split()
	hi = ngrams(hi2, number)
	terms = collections.Counter(list(hi))
	df_most_common = pd.DataFrame(terms.most_common(15), columns=['words', 'count'])
	return df_most_common

def gen_graph():
	tweets = []
	tokens=[]

	for user in db.child('tweets').get():
		tweets.append(user.val().get('processedtext'))

	terms = collections.Counter(ngrams((" ".join(tweets)).split(), number)).most_common(12)

	keys = [" ".join(i[0]) for i in terms]
	
	words  = {el:[] for el in keys}

	for user in db.child('tweets').get():
		for tweet in ngrams(TextBlob(user.val().get('processedtext')).words, number):
			new_tweet = ' '.join((tweet))
			if new_tweet in keys:
				if (user.val().get('sentimence') < -0.05):
					words[new_tweet].append('neg')
				elif ((user.val().get('sentimence') >= -0.05) & (user.val().get('sentimence') <= 0.05)):
					words[new_tweet].append('neutral')
				else:
					words[new_tweet].append('pos')

	positive = []
	neutral = []
	negative = []

	for key in words.items():
		positive_counter = 0
		neutral_counter = 0
		negative_counter = 0
		for element in key[1]:
			if (element == 'pos'):
				positive_counter+=1
			elif (element == 'neutral'):
				neutral_counter+=1
			else:
				negative_counter+=1
		positive.append(positive_counter)
		neutral.append(neutral_counter)
		negative.append(negative_counter)

	colors = ["#c9d9d3", "#718dbf", "#e84d60"]
	sentimence_values = ['Negative', 'Neutral', 'Positive']

	data = {'keys' : keys,
			'Positive' : positive,
			'Neutral' : neutral,
			'Negative' : negative
	}

	p = figure(x_range=keys, plot_height=250,
			   toolbar_location=None, tooltips="$name: @$name", tools="hover")

	p.vbar_stack(sentimence_values, x='keys', width=0.9, color=colors, source=data, legend_label=sentimence_values)

	p.xgrid.grid_line_color = None
	p.sizing_mode = 'stretch_both'

	script1, div1, = components(p)

	show(p)

#	show(graph)

def gen_network(df_tweets):
	# Create dictionary of bigrams and their counts
	d = df_tweets.set_index('words').T.to_dict('records')
	# Create network plot 
	G = nx.Graph()

	# Create connections between nodes
	for k, v in d[0].items():
		G.add_edge(k[0], k[1], weight=(v * 10))

	fig, ax = plt.subplots(figsize=(10, 8))

	pos = nx.spring_layout(G, k=2)

	# Plot networks
	nx.draw_networkx(G, pos,
					 font_size=16,
					 width=3,
					 edge_color='grey',
					 node_color='purple',
					 with_labels = False,
					 ax=ax)

	# Create offset labels
	for key, value in pos.items():
		x, y = value[0]+.135, value[1]+.045
		ax.text(x, y,
				s=key,
				bbox=dict(facecolor='red', alpha=0.25),
				horizontalalignment='center', fontsize=13)
		
	plt.show()

def polarity_his(df):
	fig, ax = plt.subplots(figsize=(8, 6))

	#df = df[df.sentimence != 0]
	# Plot histogram of the polarity values
	df.sentimence.hist(bins=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
				 ax=ax,
				 color="purple")

	plt.title("Sentiments from Tweets")
	plt.show()

def signup():
	email=input("Enter Email: ")
	password=input("Enter Password ")
	user=auth.create_user_with_email_and_password(email, password)
	return user

def signin():
	email=input("Enter Email: ")
	password=input("Enter Password ")
	user = auth.sign_in_with_email_and_password(email, password)
	return user

def piechart():
	opentickets = 0
	processing = 0
	closedtickets = 0

	for user in db.child('tweets').get():
		if (user.val().get('ticket_status') == 'Open'):
			opentickets+=1
		elif (user.val().get('ticket_status') == 'Processing'):
			processing+=1
		else:
			closedtickets+=1

	x = {
		'Open Tickets': opentickets,
		'Tickets Processing': processing,
		'Closed Tickets': closedtickets
	}

	data = pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})
	data['angle'] = data['value']/data['value'].sum() * 2*pi
	data['color'] = Category20c[len(x)]

	p = figure(plot_height=350, title="Pie Chart", toolbar_location=None,
			   tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

	p.wedge(x=0, y=1, radius=0.4,
			start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
			line_color="white", fill_color='color', legend_field='country', source=data)

	p.axis.axis_label=None
	p.axis.visible=False
	p.grid.grid_line_color = None

	script1, div1, = components(p)

	return (script1, div1, opentickets, processing, closedtickets)

def sentiment():
	sentimence = []
	positive=[]
	neutral=[]
	negative=[]

	for user in db.child('tweets').get():
		sentimence.append(TextBlob(user.val().get('processedtext'), analyzer=NaiveBayesAnalyzer()))


	for x in sentimence:
		print(x.sentiment.classification)
		if (x.sentiment.classification == 'neg'):
			negative.append(1)
		else:
			positive.append(1)

	x = {
		'Positive': len(positive),
		'Negative': len(negative)
	}

	data = pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})
	data['angle'] = data['value']/data['value'].sum() * 2*pi
	data['color'] = Spectral6[len(x)]

	p = figure(plot_height=350, title="Pie Chart", toolbar_location=None,
			   tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

	p.wedge(x=0, y=1, radius=0.4,
			start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
			line_color="white", fill_color='color', legend_field='country', source=data)

	p.axis.axis_label=None
	p.axis.visible=False
	p.grid.grid_line_color = None

	show(p)

def hashtag_freq():
	hashtags=[]

	for user in db.child('tweets').get():
		if (user.val().get('hashtags') is not None):
			hashtags.append(user.val().get('hashtags'))

	flattened_list = [y for x in hashtags for y in x]

	terms = collections.Counter(ngrams((" ".join(flattened_list)).split(), 1)).most_common(6)

	keys = [" ".join(i[0]) for i in terms]

	sentimence  = {el:[] for el in keys}

	for user in db.child('tweets').get():
		if (user.val().get('hashtags') is not None):
			for hashtag in user.val().get('hashtags'):
				if hashtag in sentimence:
					if (user.val().get('sentimence') < -0.05):
						sentimence[hashtag].append('neg')
					elif ((user.val().get('sentimence') >= -0.05) & (user.val().get('sentimence') <= 0.05)):
						sentimence[hashtag].append('neutral')
					else:
						sentimence[hashtag].append('pos')

	positive = []
	neutral = []
	negative = []

	for key in sentimence.items():
		positive_counter = 0
		neutral_counter = 0
		negative_counter = 0
		for element in key[1]:
			if (element == 'pos'):
				positive_counter+=1
			elif (element == 'neutral'):
				neutral_counter+=1
			else:
				negative_counter+=1
		positive.append(positive_counter)
		neutral.append(neutral_counter)
		negative.append(negative_counter)

	values = [i[1] for i in terms]

	colors = ["#c9d9d3", "#718dbf", "#e84d60"]
	sentimence_values = ['Negative', 'Neutral', 'Positive']

	data = {'keys' : keys,
			'Positive' : positive,
			'Neutral' : neutral,
			'Negative' : negative
	}

	p = figure(x_range=keys, plot_height=250,
			   toolbar_location=None, tooltips="$name: @$name", tools="hover")

	p.vbar_stack(sentimence_values, x='keys', width=0.9, color=colors, source=data, legend_label=sentimence_values)

	p.xgrid.grid_line_color = None
	p.sizing_mode = 'stretch_both'

	script1, div1, = components(p)

	show(p)

#optimal number of tipics
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
	coherence_values_topic = []
	model_list_topic = []
	for num_topics in range(start, limit, step):
		model = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary)
		model_list_topic.append(model)
		coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
		coherence_values_topic.append(coherencemodel.get_coherence())

	return model_list_topic, coherence_values_topic

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')


def ml():
	tweets = []
	process = []

	for user in db.child('tweets').get():
		tweets.append(TextBlob(user.val().get('processedtext')).words)
		process.append(user.val().get('processedtext'))

	dataframe = pd.DataFrame({'tweets' : tweets, 'process' : process})

	id2word = Dictionary(dataframe['tweets'])

	corpus = [id2word.doc2bow(d) for d in dataframe['tweets']]

	base_model = LdaMulticore(corpus=corpus, num_topics=5, id2word=id2word, workers=12, passes=5)

	words = [re.findall(r'"([^"]*)"',t[1]) for t in base_model.print_topics()]

	topics = [' '.join(t[0:10]) for t in words]

	for id, t in enumerate(topics): 
		print(f"------ Topic {id} ------")
		print(t, end="\n\n")

	base_perplexity = base_model.log_perplexity(corpus)
	print('\nPerplexity: ', base_perplexity) 

	coherence_model = CoherenceModel(model=base_model, texts=dataframe['tweets'], 
									   dictionary=id2word, coherence='c_v')
	coherence_lda_model_base = coherence_model.get_coherence()
	print('\nCoherence Score: ', coherence_lda_model_base)

	model_list_topic, coherence_values_topic = compute_coherence_values(dictionary=id2word,
                                                        corpus=corpus,
                                                        texts=dataframe['tweets'],
                                                        start=2, limit=200, step=6)
	print(model_list_topic, coherence_values_topic)

	vectorizer = CountVectorizer()
	data_vectorized = vectorizer.fit_transform(dataframe['process'])

	search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

	lda = LatentDirichletAllocation()

	model = GridSearchCV(lda, param_grid=search_params)

	model = model.fit(data_vectorized)

	GridSearchCV(cv=None, error_score='raise',
		 estimator=LatentDirichletAllocation(batch_size=128, 
											 doc_topic_prior=None,
											 evaluate_every=-1, 
											 learning_decay=0.7, 
											 learning_method=None,
											 learning_offset=10.0, 
											 max_doc_update_iter=100, 
											 max_iter=10,
											 mean_change_tol=0.001, 
											 n_components=10, 
											 n_jobs=1,
											 perp_tol=0.1, 
											 random_state=None,
											 topic_word_prior=None, 
											 total_samples=1000000.0, 
											 verbose=0),
		 n_jobs=1,
		 param_grid={'n_topics': [10, 15, 20, 25, 30], 
					 'learning_decay': [0.5, 0.7, 0.9]},
		 pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
		 scoring=None, verbose=0)

	best_lda_model = model.best_estimator_

	print("best paraemeters: ", model.best_params_)

	print("log score: ", model.best_score_)

	print("perplexity: ", best_lda_model.perplexity(data_vectorized))

	print(model.cv_results_)

	idk = {}

	for id, t in enumerate(topics):
		idk[id] = TextBlob(t).words

	print(idk)

	n_topics = [10, 15, 20, 25, 30]
	learning_decay = [0.5, 0.7, 0.9]
	
	plot_grid_search(model.cv_results_, learning_decay, n_topics, 'N Estimators', 'Max Features')

def mll():
	tweets = []
	process = []

	for user in db.child('tweets').get():
		tweets.append(user.val().get('t_id'))
		process.append(user.val().get('processedtext'))

	dataframe = pd.DataFrame({'tweets' : tweets, 'process' : process})

	vectorizer = CountVectorizer(
	analyzer='word',       
	min_df=3,
	stop_words='english',
	lowercase=True,
	token_pattern='[a-zA-Z0-9]{3,}',
	max_features=5000,
								)

	data_matrix = vectorizer.fit_transform(dataframe.process)

	lda_model = LatentDirichletAllocation(
	n_components=5,
	learning_method='online',
	random_state=20,       
	n_jobs = -1
										 )
	lda_output = lda_model.fit_transform(data_matrix)

	for i,topic in enumerate(lda_model.components_):
		print(f'Top 10 words for topic #{i}:')
		print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-5:]])
		print('\n')

	topic_values = lda_model.transform(data_matrix)
	dataframe['Topic'] = topic_values.argmax(axis=1)

	data = pyLDAvis.sklearn.prepare(lda_model, data_matrix, vectorizer, mds='tsne')

	base_perplexity = lda_output.log_perplexity(data_matrix)
	print('\nPerplexity: ', base_perplexity) 

	pyLDAvis.save_html(data, './templates/lda.html')

if __name__ == "__main__":
	print("")