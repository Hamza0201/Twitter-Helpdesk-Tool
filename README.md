# Twitter-Helpdesk-Tool

The project offers to combine existing ticket-based social-media help-desk systems with advanced data analytics, mainly in the sub-field of Natural Language Processing. The encompassing objective is to develop help-desks so that previously unprecedented opportunities for connecting with customers can be explored. To do this, the Twitter Helpdesk Management Tool website was created.

The Stanford-ner directory is used for named entity recognition and can be downloaded from https://nlp.stanford.edu/software/CRFNER.html. 
The static/css file contains the css files necessary for the website. Style.css is a custom css file whereas epoch.css are third party css files for graphin from Bokeh. 
The static/fonts directory contains fonts from https://fonts.google.com/. It is used to change font-face in the website. The static/images directory contains images used in the website.
The static/js folder contains the epoch.js files provided by Bokeh for graphing. The dataTables.js file is provided by dataTables. The functions.js is the custom js file used for interactivity in the website.
The service account key is a json file containing API keys for the js files to access the Cloud databases.
The templates directory contains HTML files for each web-page in the website named accordingly.
The beginStream file in the home directory is used to initialize the Twitter Streaming API to collect data. It is intended to run in the background on a separate terminal tab to run.py which runs the website. 
The python_forms is an extension to the run.py file and Is used to store forms that are used in the website.
The requirements.txt contains all the libraries that need to be installed for usage.
The run.py file contains the Flask routes that allow navigation between the different HTML web-pages in the template’s directory.
The s_functions contain the functions needed to run specific modules within the website such as sentiment analysis, topic modelling.
The service account key is a json file containing the API keys to access Firebase services.

Installation (For Windows 10):
Enter in terminal
1)Setup Environment
pip install virtualenv
mkdir Environments
cd Environments
virtualenv helpdesk
cd helpdesk/Scripts
activate
navigate to Project Directory
pip install -r requirements.txt
2)run Twitter streaming API (optional)
python beginStream.py
3) run web application
Open new terminal tab
Enter created environment again
Navigate to project directory
Python run.py
4) View Application
navigate to localhost on browser

Explanation: First, a separate
environment from the local computer is
created and activated. Then, navigate
into the project directory and install
all the requirements stated above
using pip. Using run.py, website can
be accessed through local host.
