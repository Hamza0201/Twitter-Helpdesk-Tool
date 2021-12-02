from flask import Flask, render_template, url_for, flash, redirect, session
from python_forms import RegistrationForm, LoginForm, CreateUserForm, PasswordForm, TicketForm, NoteForm
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
matplotlib.use('Agg')
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
from bokeh.palettes import Spectral6, Category20b, Category20c
from bokeh.transform import factor_cmap, cumsum
import math
from math import pi
import bokeh.layouts
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from io import BytesIO
import base64
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import secrets
from flask_login import LoginManager
from textblob.sentiments import NaiveBayesAnalyzer
import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

#Flask Configuration
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SECRET_KEY'] = 'a8348fe597185041f83d84532da7c906'
#login_manager = LoginManager(app)

#Firebase Configuration
firebaseConfig = {'apiKey': "AIzaSyC2WymgSdxypUoy49udAFTyyvTXPyc1nNc",
	'authDomain': "twitter-d34a5.firebaseapp.com",
	'databaseURL': "https://twitter-d34a5-default-rtdb.europe-west1.firebasedatabase.app",
	'projectId': "twitter-d34a5",
	'storageBucket': "twitter-d34a5.appspot.com",
	'messagingSenderId': "513871312680",
	'appId': "1:513871312680:web:291286d02f7cd0d367821d",
	'measurementId': "G-4XEWTVF57Y"}

firebase = pyrebase.initialize_app(firebaseConfig)

#Initialising Cloud Firestore for Archives
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
clouddb = firestore.client()

#Intiailising Real-Time database for tweets
db=firebase.database()

#Intialise Firebase Authentication
auth=firebase.auth()

#Background Scheduler for archiving and updating graphs
sched = BackgroundScheduler(daemon=True)

#Global variables used in updating map
latitude = []
longitude = []

#Gets user locations for geo-mapping.
def getLocation():
	global latitude, longitude
	latitude2 = []
	longitude2 = []
	apikey = 'AIzaSyCc2G01bScWagXqXlVoiRrOWigbBZZW6-4'
	geolocator = Nominatim(user_agent="u1813057.Project@gmail.com")
	locationsList = []
	for user in db.child('tweets').get():
		if (user.val().get('user_location') != ""):
			locationsList.append(user.val().get('user_location'))
	for user_loc in locationsList:
		try:
			location = geolocator.geocode(user_loc)
			
			# If coordinates are found for location
			if location:
				latitude2.append(location.latitude)
				longitude2.append(location.longitude)
				
		# If too many connection requests
		except:
			pass
	latitude = latitude2.copy()
	longitude = longitude2.copy()

	k = 6378137

	longitude = [item * (k * math.pi/180.0) for item in longitude]

	latitude = [math.log(math.tan((90 + item) * math.pi/360.0)) * k for item in latitude]

#Generate n-gram graph for trends web-page.
def gen_graph(number):
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

	colors = ["#1A237E", "#2970B8", "#4EBCD5"]
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

	return (script1, div1)

#Generate topics using LDA.
def topic_modelling():
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
		list_of_topics = [vectorizer.get_feature_names()[i] for i in topic.argsort()[-5:]]
		db.child('topics').update({i : list_of_topics})

	topic_values = lda_model.transform(data_matrix)
	dataframe['Topic'] = topic_values.argmax(axis=1)

	list_of_tweets = list(dataframe['process'])

	for tweet in db.child('tweets').get():
		if tweet.val().get('processedtext') in list_of_tweets:
			temp = dataframe.loc[dataframe['tweets'] == tweet.val().get('t_id')]
			temp1 = temp['Topic'].values.astype(int)
			db.child('tweets').child(tweet.key()).update({'topic' : temp1[0].item()})

	data = pyLDAvis.sklearn.prepare(lda_model, data_matrix, vectorizer, mds='tsne')

	pyLDAvis.save_html(data, './templates/lda.html')

#Get most liked and retweets Tweets for Trends web-page.
def get_likes_retweets():
	likes_id = None
	likes = 0
	for user in db.child('tweets').get():
		if (user.val().get('likes') > likes):
			likes = user.val().get('likes')
			likes_id = user.val().get('t_id')
			screen_name_likes = user.val().get('screen_name')

	retweets_id = None
	retweets = 0
	for user in db.child('tweets').get():
		if (user.val().get('retweets') > retweets):
			retweets = user.val().get('retweets')
			retweets_id = user.val().get('t_id')
			screen_name_retweets = user.val().get('screen_name')

	return (likes_id, screen_name_likes, retweets_id, screen_name_retweets)

# Sentiment analysis.
def sentiment():
	positive=[]
	negative=[]
	neutral=[]

	for user in db.child('tweets').get():
		if (user.val().get('sentimence') < -0.05):
			negative.append(0)
		elif ((user.val().get('sentimence') >= -0.05) & (user.val().get('sentimence') <= 0.05)):
			positive.append(0)
		else:
			neutral.append(0)

	x = {
		'Positive': len(positive),
		'Neutral' : len(neutral),
		'Negative': len(negative)
	}

	data = pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})
	data['angle'] = data['value']/data['value'].sum() * 2*pi
	data['color'] = ["#4EBCD5", "#2970B8", "#1A237E"]

	p = figure(plot_height=450, toolbar_location=None,
			   tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

	p.annular_wedge(x=0, y=1, inner_radius=0.23, outer_radius=0.35, direction="anticlock",
			start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
			line_color="white", fill_color='color', legend_field='country', source=data)

	p.axis.axis_label=None
	p.axis.visible=False
	p.grid.grid_line_color = None
	p.sizing_mode = 'stretch_both'

	script1, div1, = components(p)

	return (script1, div1)

#Pie chart for dashboard tickets.
def piechart():
	opentickets = 0
	processing = 0
	closedtickets = 0
	overdue = 0

	now = datetime.now()
	for user in db.child('tweets').get():
		time_difference = datetime.strptime(user.val().get('date'), '%d-%m-%Y')
		if (now-timedelta(days=7) >= time_difference):
			overdue+=1
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
	data['color'] = Category20b[len(x)]

	p = figure(plot_height=350, toolbar_location=None,
			   tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

	p.wedge(x=0, y=1, radius=0.4,
			start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
			line_color="white", fill_color='color', legend_field='country', source=data)

	p.axis.axis_label=None
	p.axis.visible=False
	p.grid.grid_line_color = None
	p.sizing_mode = 'stretch_both'

	script1, div1, = components(p)

	return (script1, div1, opentickets, processing, closedtickets, overdue)

#Hashtag frequency +sentiment analysis.
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

	colors = ["#1A237E", "#2970B8", "#4EBCD5"]
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

	return (script1, div1)

#Archive for reports.
def archiver():
	for user in db.child('tweets').get():
		if (user.val().get('ticket_status') == 'Closed'):
			clouddb.collection(datetime.today().strftime('%d-%m-%Y')).document(user.val().get('t_id')).set(user.val())
			db.child('tweets').child(user.key()).remove()

#Live graph updates.
def update_counter():
		current_date = datetime.now().strftime("%d-%m-%Y")
		current_time = (datetime.now()+ timedelta(seconds=10)).strftime("%H:%M:%S")
		past_time_delta = datetime.now() - timedelta(minutes=5)
		past_time = past_time_delta.strftime("%H:%M:%S")
		positive = 0
		neutral = 0
		negative = 0
		open_tickets = 0
		processing_tickets = 0
		closed_tickets = 0
		for tweet in db.child('tweets').get():
			date = tweet.val().get('date')
			time = tweet.val().get('time')
			if ((date == current_date) & (time >= past_time)):
				if (tweet.val().get('sentimence') < -0.05):
					negative+=1
				elif ((tweet.val().get('sentimence') >= -0.05) & (tweet.val().get('sentimence') <= 0.05)):
					positive+=1
				else:
					neutral+=1
			if (tweet.val().get('ticket_status') =='Open'):
				open_tickets+=1
			elif (tweet.val().get('ticket_status') =='Processing'):
				processing_tickets+=1
			else:
				closed_tickets+=1
		old_open_tickets = 0
		old_processing_tickets = 0
		old_closed_tickets = 0
		tweet = db.child('live_counter').get()
		old_open_tickets += tweet.val().get('open_tickets')
		old_processing_tickets += tweet.val().get('processing_tickets')
		old_closed_tickets += tweet.val().get('closed_tickets')
		new_open_tickets = open_tickets - old_open_tickets if (open_tickets - old_open_tickets >= 0) else 0
		new_processing_tickets = processing_tickets - old_processing_tickets if (processing_tickets - old_processing_tickets >= 0) else 0
		new_closed_tickets = closed_tickets - old_closed_tickets if (closed_tickets - old_closed_tickets >= 0) else 0
		db.child('live_counter').set({'time' : current_time, 'positive' : positive, 'neutral' : neutral, 'negative' : negative, 'open_tickets' : open_tickets, 'processing_tickets' : processing_tickets, 'closed_tickets' : closed_tickets})
		db.child('sentiment_recorder').push({'time': current_time,'positive' : positive, 'neutral' : neutral, 'negative' : negative})
		db.child('ticket_recorder').push({'time': current_time,'open_tickets' : new_open_tickets, 'processing_tickets' : new_processing_tickets, 'closed_tickets' : new_closed_tickets})

#Adding jobs to Scheduler
sched.add_job(getLocation,'interval',seconds=600)
sched.add_job(archiver,'cron',hour=9, minute=32)
sched.add_job(update_counter,'interval',seconds=290)
sched.add_job(topic_modelling,'interval',seconds=120)

#Starting Scheduling of added jobs
sched.start()

#\Route for browser
@app.route("/", methods=['GET','POST'])
@app.route("/login", methods=['GET','POST'])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		try:
			auth.sign_in_with_email_and_password(form.username.data, form.password.data)
			session['usr'] = form.username.data
			return redirect(url_for('home'))
		except Exception as e:
			app.logger.warning(e)
			flash('There was an error with your E-Mail/Password combination. Please try again.', 'danger')
	return render_template('login.html', form=form)

@app.route("/adminlogin", methods=['GET','POST'])
def adminlogin():
	try:
		session['usr']
		form = LoginForm()
		if form.validate_on_submit():
			try:
				if (form.username.data != "u1813057.project@gmail.com"):
					raise Exception('I know Python!')
				auth.sign_in_with_email_and_password(form.username.data, form.password.data)
				session['admin'] = form.username.data
				return redirect(url_for('admin_dashboard'))
			except:
				flash('Admin Access Required.', 'danger')
		return render_template('adminlogin.html', form=form)
	except:
		return redirect(url_for('login'))

@app.route("/admin_dashboard", methods=['GET','POST'])
def admin_dashboard():
	try:
		session['admin']
		createForm = CreateUserForm()
		passwordForm = PasswordForm()

		if createForm.validate_on_submit():
			try:
				user=auth.create_user_with_email_and_password(createForm.createEmail.data, secrets.token_hex(16))
				auth.send_password_reset_email(createForm.createEmail.data)
				info = {'email' : createForm.createEmail.data, 'createdAt' : datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
				db.child('users').push(info)
				return redirect(url_for('admin_dashboard'))
			except:
				flash('Email is already associated with an account.', 'danger')
				return redirect(url_for('admin_dashboard'))

		if passwordForm.validate_on_submit():
			for user in db.child('users').get():
				if (user.val().get('email') == passwordForm.changeEmail.data):
					auth.send_password_reset_email(passwordForm.changeEmail.data)
					return redirect(url_for('admin_dashboard'))
			flash('Email does not exist.', 'danger')
			return redirect(url_for('admin_dashboard'))

		return render_template('admin_dashboard.html', createForm=createForm, passwordForm=passwordForm)
	except:
		return redirect(url_for('adminlogin'))

@app.route("/home")
def home():
	try:
		session['usr']
		global latitude, longitude

		tile_provider = get_provider(CARTODBPOSITRON)

		# range bounds supplied in web mercator coordinates
		p = figure(x_range=(-1000000, 3000000), y_range=(-7000000, 13000000),
			   x_axis_type="mercator", y_axis_type="mercator")
		p.sizing_mode = 'stretch_both'
		p.axis.visible = False
		p.add_tile(tile_provider)

		source = ColumnDataSource(data=dict(lat=(latitude), lon=(longitude)))

		p.circle(x="lon", y="lat", size=4, fill_color="tomato", fill_alpha=0.3, line_color="tomato", source=source)

		script1, div1, = components(p)
		cdn_js = CDN.js_files[0]

		img = BytesIO()
		
		all_words = []

		for user in db.child('tweets').get():
			temp = user.val().get('processedtext')
			all_words.extend(TextBlob(temp).words)
			
		converted_list = [x.upper() for x in all_words]
		# create a word frequency dictionary
		wordfreq = Counter(converted_list)

		cmap = matplotlib.cm.Blues(np.linspace(0,1,20))
		cmap = matplotlib.colors.ListedColormap(cmap[10:,:-1])

		font_path = 'static/fonts/poppins-v15-latin-500.ttf'
		mask = np.array(Image.open('static/images/logo.png'))

		wordcloud = WordCloud(width=mask.shape[1],
							  height=mask.shape[0],
							  max_words=500,
							  relative_scaling=0.5,
							  colormap=cmap,
							  background_color="white",
							  font_path=font_path,
							  mask=mask,
							  normalize_plurals=True).generate_from_frequencies(wordfreq)

		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis("off")

		plt.savefig(img, format='png', bbox_inches='tight', transparent="True", pad_inches=0)
		plt.close()
		img.seek(0)

		plot_url = base64.b64encode(img.getvalue()).decode('utf8')

		pie_chart = piechart()

		return render_template("dashboard.html", script1=script1, div1=div1, cdn_js=cdn_js, opentickets=pie_chart[2], processing=pie_chart[3], closedtickets=pie_chart[4], overdue=pie_chart[5], plot_url=plot_url,
			script3=pie_chart[0], div3=pie_chart[1])
	except Exception as e:
		app.logger.warning(e)
		return redirect(url_for('login'))

@app.route("/tickets")
def tickets():
	try:
		session['usr']
		return render_template('tickets.html')
	except:
		return redirect(url_for('login'))

@app.route("/logout")
def logout():
	session.pop('usr', None)
	session.pop('admin', None)
	return redirect(url_for('login'))

@app.route("/trends")
def trends():
	try:
		session['usr']
		graph = gen_graph(1)
		graph1 = gen_graph(2)
		graph2 = gen_graph(3)
		cdn_js = CDN.js_files[0]
		sentimence = sentiment()
		hashtag_frequency = hashtag_freq()
#		get_info = get_likes_retweets()
#		link_likes = 'https://twitter.com/' + get_info[1] + "/status/" + get_info[0]
#		link_retweets = 'https://twitter.com/' + get_info[3] + "/status/" + get_info[2]
		organization = []
		location = []
		person = []
		for tweet in db.child('tweets').get():
			if (tweet.val().get('ner') is not None):
				for x in tweet.val().get('ner').items():
					if x[0] == 'LOCATION':
						for y in x[1]:
							location.append(y)
					elif x[0] == 'PERSON':
						for y in x[1]:
							person.append(y)
					else:
						for y in x[1]:
							organization.append(y)

		organization = collections.Counter(organization).most_common(5)
		location = collections.Counter(location).most_common(5)
		person = collections.Counter(person).most_common(5)

		topics_dict = {}

		if db.child('topics').get() is not None:
			for topic in db.child('topics').get():
				topics_dict[topic.key() + 1] = topic.val()

		return render_template('trends.html',cdn_js=cdn_js, script2=graph[0], div2=graph[1], script3=graph1[0], div3=graph1[1], script4=graph2[0], div4=graph2[1],
			script5=sentimence[0], div5=sentimence[1], script6=hashtag_frequency[0], div6=hashtag_frequency[1],
			organization=organization, location=location, person=person,
			topics_dict=topics_dict)
	except Exception as e:
		app.logger.warning(e)
		return redirect(url_for('login'))

@app.route("/reports")
def reports():
	try:
		session['usr']
		yesterdaydate = (datetime.now()).strftime('%d-%m-%Y')
		return render_template('reports.html', yesterdaydate=yesterdaydate)
	except:
		return redirect(url_for('login'))

@app.route("/lda")
def lda():
	try:
		session['usr']
		return render_template('lda.html')
	except:
		return redirect(url_for('login'))

@app.route("/knowledgebase")
def knowledgebase():
	try:
		session['usr']
		return render_template('knowledgebase.html')
	except:
		return redirect(url_for('login'))

@app.route("/notes", methods=['GET','POST'])
def notes():
	try:
		session['usr']
		notesForm = NoteForm()

		if notesForm.validate_on_submit():
			for user in db.child('users').get():
				if (user.val().get('email') == session['usr']):
					note = {}
					note['time'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
					note['title'] = notesForm.createNoteTitle.data
					note['description'] = notesForm.createNoteDesc.data
					db.child('users').child(user.key()).child('notes').push(note)
			return redirect(url_for('notes'))

		return render_template('notes.html', notesForm=notesForm, email=session['usr'])
	except:
		return redirect(url_for('login'))

@app.route("/settings", methods=['GET','POST'])
def settings():
	try:
		session['usr']
		ticketForm = TicketForm()
		passwordForm=PasswordForm()

		if ticketForm.validate_on_submit():
			ticket = {}
			ticket['time'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
			ticket['title'] = ticketForm.createTitle.data
			ticket['description'] = ticketForm.createDesc.data
			db.child('tickets').push(ticket)
			return redirect(url_for('settings'))

		if passwordForm.validate_on_submit():
			for user in db.child('users').get():
				if (user.val().get('email') == passwordForm.changeEmail.data):
					auth.send_password_reset_email(passwordForm.changeEmail.data)
					return redirect(url_for('settings'))
			flash('Email does not exist.', 'danger')
			return redirect(url_for('settings'))

		return render_template('settings.html', ticketForm=ticketForm, passwordForm=passwordForm)
	except:
		return redirect(url_for('login'))

#Close Scheduler on exit
atexit.register(lambda: sched.shutdown())

if __name__ == '__main__':
	app.jinja_env.auto_reload = True
	app.config['TEMPLATES_AUTO_RELOAD'] = True
	app.run('0.0.0.0',port=5000, debug=True, use_reloader=False)