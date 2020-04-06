from flask import Flask, jsonify, request
import numpy as np
import pandas as pd

#For other warnings
import warnings
warnings.filterwarnings("ignore")

#https://stackoverflow.com/questions/879173/how-to-ignore-deprecation-warnings-in-python
#For warnings related to joblib
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import joblib

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import re

#https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__, template_folder='C:/Users/Bhavesh/Desktop/BE Project/Deployment/COGMUML/templates')

################### PRE-PROCESSING ############################

from nltk.corpus import stopwords

#loading stop words from nltk library
stop_words = set(stopwords.words('english'))

def nlp_preprocessing(total_text):
    if type(total_text) is not int:
        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub(r'\s+',' ', total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += word + " "
        
        return string

@app.route('/')
def hello_world():
    return 'Hello World!'

#https://www.pythonanywhere.com/forums/topic/1039/
@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/result', methods=['POST'])

def result():
	clf = joblib.load('LogReg_tfidf.pkl')
	clf_tfidf_nb = joblib.load('NaiveBayes_tfidf.pkl')
	clf_tfidf_knn = joblib.load('KNN_tfidf.pkl')
	clf_tfidf_rf = joblib.load('RandomForest_tfidf.pkl')
	clf_bow_lr = joblib.load('LogReg_bow.pkl')
	clf_bow_nb = joblib.load('NaiveBayes_bow.pkl')
	clf_bow_knn = joblib.load('KNN_bow.pkl')
	clf_bow_rf = joblib.load('RandomForest_bow.pkl')
	tfidf_vect = joblib.load('tfidf_vect.pkl')
	count_vect_gene = joblib.load('count_vect_gene.pkl')
	count_vect_var = joblib.load('count_vect_var.pkl')
	to_predict_list = request.form.to_dict()
	medical_text = nlp_preprocessing(to_predict_list['medical_text'])
	text = tfidf_vect.transform([medical_text])
	
	gene = count_vect_gene.transform([to_predict_list['gene']])
	variation = count_vect_var.transform([to_predict_list['variation']])
	
	from scipy.sparse import hstack
	concat_ = hstack((text, gene, variation))

    #https://acroz.dev/2017/11/03/data-science-apis-flask/

	################## TFIDF ###################
    #Logistic Regression TFIDF
	predicted_class = clf.predict(concat_)[0]
	probabilities = clf.predict_proba(concat_)[0]

    #Naive Bayes TFIDF
	predicted_class_nb_tfidf = clf_tfidf_nb.predict(concat_)[0]
	probabilities_nb_tfidf = clf_tfidf_nb.predict_proba(concat_)[0]

    #K-NN TFIDF
	predicted_class_knn_tfidf = clf_tfidf_knn.predict(concat_)[0]
	probabilities_knn_tfidf = clf_tfidf_knn.predict_proba(concat_)[0]

    #Random Forest TFIDF
	predicted_class_rf_tfidf = clf_tfidf_rf.predict(concat_)[0]
	probabilities_rf_tfidf = clf_tfidf_rf.predict_proba(concat_)[0]

	################## BOW #####################
	
	#Logistic Regression BOW
	predicted_class_lr_bow = clf_bow_lr.predict(concat_)[0]
	probabilities_lr_bow = clf_bow_lr.predict_proba(concat_)[0]
	
    #Naive Bayes BOW
	predicted_class_nb_bow = clf_bow_nb.predict(concat_)[0]
	probabilities_nb_bow = clf_bow_nb.predict_proba(concat_)[0]

    #K-NN BOW
	predicted_class_knn_bow = clf_bow_knn.predict(concat_)[0]
	probabilities_knn_bow = clf_bow_knn.predict_proba(concat_)[0]
	
    #Random Forest TFIDF
	predicted_class_rf_bow = clf_bow_rf.predict(concat_)[0]
	probabilities_rf_bow = clf_bow_rf.predict_proba(concat_)[0]

    #IMP: https://www.tutorialspoint.com/flask/flask_templates.htm
	probs = {
		'Predicted Class using Logistic Regression (TFIDF Encoding)': predicted_class,
        'Gain-of-function (Logistic Regression TFIDF)': round(probabilities[0], 3),
        'Inconclusive (Logistic Regression TFIDF)': round(probabilities[1], 3),
        'Likely Gain-of-function (Logistic Regression TFIDF)': round(probabilities[2], 3),
        'Likely Loss-of-function (Logistic Regression TFIDF)': round(probabilities[3], 3),
        'Likely Neutral (Logistic Regression TFIDF)': round(probabilities[4], 3),
        'Likely Switch-of-function (Logistic Regression TFIDF)': round(probabilities[5], 3),
        'Loss-of-function (Logistic Regression TFIDF)': round(probabilities[6], 3),
        'Neutral (Logistic Regression TFIDF)': round(probabilities[7], 3),
        'Switch-of-function (Logistic Regression TFIDF)': round(probabilities[8], 3),    
        
		
        'Predicted Class using Naive Bayes (TFIDF Encoding)': predicted_class_nb_tfidf,
        'Gain-of-function (Naive Bayes TFIDF)': round(probabilities_nb_tfidf[0], 3),
        'Inconclusive (Naive Bayes TFIDF)': round(probabilities_nb_tfidf[1], 3),
        'Likely Gain-of-function (Naive Bayes TFIDF)': round(probabilities_nb_tfidf[2], 3),
        'Likely Loss-of-function (Naive Bayes TFIDF)': round(probabilities_nb_tfidf[3], 3),
        'Likely Neutral (Naive Bayes TFIDF)': round(probabilities_nb_tfidf[4], 3),
        'Likely Switch-of-function (Naive Bayes TFIDF)': round(probabilities_nb_tfidf[5], 3),
        'Loss-of-function (Naive Bayes TFIDF)': round(probabilities_nb_tfidf[6], 3),
        'Neutral (Naive Bayes TFIDF)': round(probabilities_nb_tfidf[7], 3),
        'Switch-of-function (Naive Bayes TFIDF)': round(probabilities_nb_tfidf[8], 3),

        

        'Predicted Class using K-NN (TFIDF Encoding)': predicted_class_knn_tfidf,
        'Gain-of-function (K-NN TFIDF)': round(probabilities_knn_tfidf[0], 3),
        'Inconclusive (K-NN TFIDF)': round(probabilities_knn_tfidf[1], 3),
        'Likely Gain-of-function (K-NN TFIDF)': round(probabilities_knn_tfidf[2], 3),
        'Likely Loss-of-function (K-NN TFIDF)': round(probabilities_knn_tfidf[3], 3),
        'Likely Neutral (K-NN TFIDF)': round(probabilities_knn_tfidf[4], 3),
        'Likely Switch-of-function (K-NN TFIDF)': round(probabilities_knn_tfidf[5], 3),
        'Loss-of-function (K-NN TFIDF)': round(probabilities_knn_tfidf[6], 3),
        'Neutral (K-NN TFIDF)': round(probabilities_knn_tfidf[7], 3),
        'Switch-of-function (K-NN TFIDF)': round(probabilities_knn_tfidf[8], 3),


        'Predicted Class using Random Forest (TFIDF Encoding)': predicted_class_rf_tfidf,
        'Gain-of-function (Random Forest TFIDF)': round(probabilities_rf_tfidf[0], 3),
        'Inconclusive (Random Forest TFIDF)': round(probabilities_rf_tfidf[1], 3),
        'Likely Gain-of-function (Random Forest TFIDF)': round(probabilities_rf_tfidf[2], 3),
        'Likely Loss-of-function (Random Forest TFIDF)': round(probabilities_rf_tfidf[3], 3),
        'Likely Neutral (Random Forest TFIDF)': round(probabilities_rf_tfidf[4], 3),
        'Likely Switch-of-function (Random Forest TFIDF)': round(probabilities_rf_tfidf[5], 3),
        'Loss-of-function (Random Forest TFIDF)': round(probabilities_rf_tfidf[6], 3),
        'Neutral (Random Forest TFIDF)': round(probabilities_rf_tfidf[7], 3),
        'Switch-of-function (Random Forest TFIDF)': round(probabilities_rf_tfidf[8], 3),
		
		
		'Predicted Class using Logistic Regression (BOW Encoding)': predicted_class_lr_bow,
        'Gain-of-function (Logistic Regression BOW)': round(probabilities_lr_bow[0], 3),
        'Inconclusive (Logistic Regression BOW)': round(probabilities_lr_bow[1], 3),
        'Likely Gain-of-function (Logistic Regression BOW)': round(probabilities_lr_bow[2], 3),
        'Likely Loss-of-function (Logistic Regression BOW)': round(probabilities_lr_bow[3], 3),
        'Likely Neutral (Logistic Regression BOW)': round(probabilities_lr_bow[4], 3),
        'Likely Switch-of-function (Logistic Regression BOW)': round(probabilities_lr_bow[5], 3),
        'Loss-of-function (Logistic Regression BOW)': round(probabilities_lr_bow[6], 3),
        'Neutral (Logistic Regression BOW)': round(probabilities_lr_bow[7], 3),
        'Switch-of-function (Logistic Regression BOW)': round(probabilities_lr_bow[8], 3),

		'Predicted Class using Naive Bayes (BOW Encoding)': predicted_class_nb_bow,
        'Gain-of-function (Naive Bayes BOW)': round(probabilities_nb_bow[0], 3),
        'Inconclusive (Naive Bayes BOW)': round(probabilities_nb_bow[1], 3),
        'Likely Gain-of-function (Naive Bayes BOW)': round(probabilities_nb_bow[2], 3),
        'Likely Loss-of-function (Naive Bayes BOW)': round(probabilities_nb_bow[3], 3),
        'Likely Neutral (Naive Bayes BOW)': round(probabilities_nb_bow[4], 3),
        'Likely Switch-of-function (Naive Bayes BOW)': round(probabilities_nb_bow[5], 3),
        'Loss-of-function (Naive Bayes BOW)': round(probabilities_nb_bow[6], 3),
        'Neutral (Naive Bayes BOW)': round(probabilities_nb_bow[7], 3),
        'Switch-of-function (Naive Bayes BOW)': round(probabilities_nb_bow[8], 3),
		
		'Predicted Class using K-NN (BOW Encoding)': predicted_class_knn_bow,
        'Gain-of-function (K-NN BOW)': round(probabilities_knn_bow[0], 3),
        'Inconclusive (K-NN BOW)': round(probabilities_knn_bow[1], 3),
        'Likely Gain-of-function (K-NN BOW)': round(probabilities_knn_bow[2], 3),
        'Likely Loss-of-function (K-NN BOW)': round(probabilities_knn_bow[3], 3),
        'Likely Neutral (K-NN BOW)': round(probabilities_knn_bow[4], 3),
        'Likely Switch-of-function (K-NN BOW)': round(probabilities_knn_bow[5], 3),
        'Loss-of-function (K-NN BOW)': round(probabilities_knn_bow[6], 3),
        'Neutral (K-NN BOW)': round(probabilities_knn_bow[7], 3),
        'Switch-of-function (K-NN BOW)': round(probabilities_knn_bow[8], 3),
		
		'Predicted Class using Random Forest (BOW Encoding)': predicted_class_rf_bow,
        'Gain-of-function (Random Forest BOW)': round(probabilities_rf_bow[0], 3),
        'Inconclusive (Random Forest BOW)': round(probabilities_rf_bow[1], 3),
        'Likely Gain-of-function (Random Forest BOW)': round(probabilities_rf_bow[2], 3),
        'Likely Loss-of-function (Random Forest BOW)': round(probabilities_rf_bow[3], 3),
        'Likely Neutral (Random Forest BOW)': round(probabilities_rf_bow[4], 3),
        'Likely Switch-of-function (Random Forest BOW)': round(probabilities_rf_bow[5], 3),
        'Loss-of-function (Random Forest BOW)': round(probabilities_rf_bow[6], 3),
        'Neutral (Random Forest BOW)': round(probabilities_rf_bow[7], 3),
        'Switch-of-function (Random Forest BOW)': round(probabilities_rf_bow[8], 3)
		
	}

	return flask.render_template("result.html", result = probs)



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5050)    