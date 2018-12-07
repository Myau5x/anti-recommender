import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from flask import Flask, request, render_template, jsonify
mport pickle
from scraping.add_by_scrap import scrap_by_users

with open('model_parts/cv.mdl','rb') as f:
    cv = pickle.load(f)
with open('model_parts/km.mdl','rb') as f:
    km = pickle.load(f)
with open('model_parts/idf', 'rb') as f:
    new_idf =pickle.load( f)

ClustModel = ClusterReviews(cv, new_idf, km)

app = Flask(__name__, static_url_path="")

colClust_n = ['cl_t0', 'cl_t1', 'cl_t2', 'cl_t3', 'cl_t4', 'cl_t5', 'cl_t7', 'cl_t8', 'cl_t9', 'cl_t10', 'cl_t11', 'cl_t12', 'cl_t13', 'cl_t14', 'cl_t16', 'cl_t17']
with open('colPred213', 'rb') as f:
    colPred_pr = pickle.load(f)
with open('thres_35_rf','rb') as f:
    th_rf35 = pickle.load(f )
### for banned me
with open('offline.d', 'rb') as f:
    test_rest = pickle.load(f)
rf = ComboModel('rforest/mrf_', th_rf35, colPred_pr, colClust_n)

with open('model_feat.pkl', 'rb') as f:
    model_features = pickle.load(f)

DEFAULT_USER = ['cl_t1',  'cl_t8', 'cl_t3']
Real_user = []
colShow = ['name', 'BAD', 'rating', 'location' , 'url']

app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index_2.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return the prediction."""
    data = request.json
    loc = data['user_loc']
    real_user = data['user_cl'].strip().split(' ')


    rests  = build_restoraunts(loc)
    if rests is None:
        return jsonify({'error': 'Cant get food'})
        #rests = test_rest
    else:
        X = cleaning_data(rests, model_features)
        if len(real_user) == 0:
            prediction = rf.predict(X, DEFAULT_USER)
        else:
            prediction = rf.predict(X, real_user)

        am_bad = int(prediction.sum())
        am_norm = int(len(prediction) - am_bad)
        rests['url'] = rests['url'].str.split('?').map(lambda x : x[0])
        rests['BAD'] = prediction
        bad = rests[prediction][colShow]
        good = rests[~prediction][colShow]
        return jsonify({"bad":bad.to_html(), "good":good.to_html()})
        #return jsonify({'bad restaurants': am_bad, 'norm restaurants': am_norm})

    #return jsonify({'cluster': real_user})
@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    """Return a random prediction."""
    return jsonify({'a':1, 'b':2})

@app.route('/clusters', methods=['GET', 'POST'])
def clusters():
    """Have to cluster user by review, now just take cluster nums"""
    data = request.json
    stroka = data['user_rev']
    l = stroka.split(',')
    user = []
    for x in l:
        user.append('cl_t'+str(int(x)))
    Real_user = list(set(user))
    s = ''
    for cl in Real_user:
        s += str(cl)+' '
    return s

@app.route('/clustering_yelp', methods=['GET', 'POST'])
def clusters_from_yelp():
    """Have to cluster user by review, now just take cluster nums"""
    url = request.json['user_yelp']
    rews = scrap_by_users(url)
    rs = extract_bad_revs(rews)
    if len(rs) == 0:
        return
    ###Here should be filtering, tokenize and clustering
    cl_med = ClustModel.transform(rs)

    return s
