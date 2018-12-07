import pandas as pd
import time
from scraping.yelp_api import search, API_KEY, DEFAULT_TERM
import pickle
import numpy as np

from trans_rat import FillPrice, CategoriesTransformScr, RatingNormalizer, FillPriceScr

### looking just on first 100
offset_lim = 100

def build_restoraunts(loc):
    """ collects offset_lim (100) of nearest restaurants  through YELP API"""

    offsets = range(0,offset_lim,50)
    l = []
    for offset in offsets:
        response = search(API_KEY, DEFAULT_TERM, loc, offset)
        data = response.get('businesses', None)
        if data is None:
            return None
        l+=data
    return pd.DataFrame(l)

### clean data

def cleaning_data( X, model_features, inplace = False):
    X = X.copy() if not inplace else X
    X.drop(columns= ['coordinates', 'display_phone',
                                        'distance','image_url', 'name','location',
                                       'is_closed','phone', 'transactions','url'], inplace = True)
    fpr  = FillPriceScr()
    ct = CategoriesTransformScr(model_features)
    rn = RatingNormalizer(starCol = 'rating')
    X = fpr.transform(X, inplace = True)
    X = ct.transform(X, inplace = True)
    X = rn.transform(X, inplace = True)
    return X

class ComboModel:
    def __init__(self, model_path, model_thres, col_pred, col_clust):
        self.clust_model = {}
        self.model_thres = model_thres
        self.col_pred = col_pred
        for col in col_clust:

            filename_rf = model_path+col+'.pkl'
            with open(filename_rf,'rb') as f:
                self.clust_model[col] =pickle.load( f)

    def predict(self, X, cl_user ):
        Y = pd.DataFrame()
        #Y['alias'] = X['alias']
        answer = np.array(len(X)*[False])
        for col in cl_user:
            Y[col] = self.clust_model[col].predict_proba(X[self.col_pred])[:,1]
            answer = answer|(Y[col]>self.model_thres[col])
        return answer

class ClusterReviews:

    """takes:
    X - list of reviews
    fitted models and idf-vector - returns cluster nums for reviews"""
    def __init__(self,countvectorizer, idf, clustering):
        self.cv = countvectorizer
        self.idf = idf
        self.clustering = clustering
    def transform(self,X):
        c = self.cv.transform(X)
        i = c.toarray()*self.idf
        cl = self.clustering.predict(i)
        return cl

def extract_bad_revs(rs):
    r_df = pd.DataFrame(rs)
    r_df['rating'] = r_df.rating.astype(float)
    l = r_df[r_df.rating <=3]['text']
    return l.tolist()
