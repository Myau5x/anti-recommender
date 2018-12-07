import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


from flask import Flask, request, render_template, jsonify
from concept_proof import build_restoraunts, cleaning_data, ComboModel
import pickle

app = Flask(__name__, static_url_path="")

colClust_n = ['cl_t0', 'cl_t1', 'cl_t2', 'cl_t3', 'cl_t4', 'cl_t5', 'cl_t7', 'cl_t8', 'cl_t9', 'cl_t10', 'cl_t11', 'cl_t12', 'cl_t13', 'cl_t14', 'cl_t16', 'cl_t17']
with open('colPred213', 'rb') as f:
    colPred_pr = pickle.load(f)
with open('thres_35_rf','rb') as f:
    th_rf35 = pickle.load(f )
rf = ComboModel('rforest/mrf_', th_rf35, colPred_pr, colClust_n)

with open('model_feat.pkl', 'rb') as f:
    model_features = pickle.load(f)

DEFAULT_USER = ['cl_t1',  'cl_t8', 'cl_t4']

colShow = ['name', 'rating', 'location' , 'url']

app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index_3.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    data = request.json
    loc = data['user_loc']
    rests  = build_restoraunts(loc)
    if rests is None:
        return jsonify({'error': 'Cant get food'})
    #prediction = model.predict_proba([data['user_input']])
    else:
        X = cleaning_data(rests, model_features)
        prediction = rf.predict(X, DEFAULT_USER)
        am_bad = int(prediction.sum())
        am_norm = int(len(prediction) - am_bad)
        #return jsonify({'bad restaurants': am_bad, 'norm restaurants': am_norm})
        return rests[prediction][colShow].to_html()

@app.route('/clusters', methods=['GET', 'POST'])
def clusters():
    """Return a random prediction."""
    data = request.json


    return jsonify({'cluster': DEFAULT_USER})
