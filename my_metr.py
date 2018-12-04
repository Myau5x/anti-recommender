import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score,  recall_score, precision_score


def accu(act, pred, thres):
    return ((act <thres) == (pred <thres)).mean()


def rating_to_prob(rating):
    '''convert rating to probability that it is bad rest'''
    """
    1 --- 1
    2 --- .9
    3 --- .8
    4 --- .2
    5 --- 0

    """
    prob = np.zeros(len(rating))
    prob[rating >= 5] = 0
    prob[rating <1] = 1
    prob[(rating<5)&(rating>=4)] = (5- rating[(rating<5)&(rating>=4)])*0.2
    prob[(rating<4)&(rating>=3)] = (4- rating[(rating<4)&(rating>=3)])*0.6 + 0.2
    prob[(rating<3)&(rating>=2)] = (3- rating[(rating<3)&(rating>=2)])*0.1 + 0.8
    prob[(rating<2)&(rating>=1)] = (2- rating[(rating<2)&(rating>=1)])*0.1 + 0.9
    return prob

def transform_to_score(data):
    """ transform review """
    test_df = data.copy()
    test_df['similar']  = (test_df.user_cl == test_df.biz_cl)
    test_gr_df =  test_df.groupby('review_id').agg({'similar':'sum', 'stars':'mean', 'rating':'mean'})
    test_gr_df['pred'] = (test_gr_df.similar > 0)
    test_gr_df['act'] = (test_gr_df.stars < 3)
    test_gr_df['base'] = (test_gr_df.rating < 3)
    test_gr_df['base_3.5'] = (test_gr_df.rating < 3.5)

    return test_gr_df

def transform_aggregated(data):
    """Transform aggregated data to dataFrame thats easy to calculate score"""
    test_gr_df =  data.copy()
    test_gr_df['pred'] = (test_gr_df['sum(similar)'] > 0)

    test_gr_df['act'] = (test_gr_df['avg(stars)'] < 3)
    test_gr_df['base'] = (test_gr_df['avg(rating)'] < 3)
    test_gr_df['base_3.5'] = (test_gr_df['avg(rating)'] < 3.5)

    return test_gr_df



def my_scorer(data, colTrue ='act', colPred = 'pred', colBase = 'base' , colBase35 = 'base_3.5'):
    acc ={}
    acc[colPred]  = accuracy_score(data[colTrue], data[colPred])
    acc[colBase] = accuracy_score(data[colTrue], data[colBase])
    acc[colBase35] = accuracy_score(data[colTrue], data[colBase35])
    acc['combo_base'] = accuracy_score(data[colTrue], data[colBase]|data[colPred])
    acc['combo_35'] = accuracy_score(data[colTrue], data[colBase35]|data[colPred])
    recall = {}
    recall[colPred]  = recall_score(data[colTrue], data[colPred])
    recall[colBase] = recall_score(data[colTrue], data[colBase])
    recall[colBase35] = recall_score(data[colTrue], data[colBase35])
    recall['combo_base'] = recall_score(data[colTrue], data[colBase]|data[colPred])
    recall['combo_35'] = recall_score(data[colTrue], data[colBase35]|data[colPred])
    prec ={}
    prec[colPred]  = precision_score(data[colTrue], data[colPred])
    prec[colBase] = precision_score(data[colTrue], data[colBase])
    prec[colBase35] = precision_score(data[colTrue], data[colBase35])
    prec['combo_base'] = precision_score(data[colTrue], data[colBase]|data[colPred])
    prec['combo_35'] = precision_score(data[colTrue], data[colBase35]|data[colPred])



    return pd.DataFrame([acc, recall, prec], index = ['accuracy', 'recall', 'prec'])


def plot_roc_curve(y_true, X, model, ax, label):
    """Plot roc curve, y_true & model.predict_proba(X), ax --- axes of subplot"""
    fpr, tpr, thr = roc_curve(y_true, model.predict_proba(X)[:,1])
    ax.plot(fpr, tpr, label = label)

def calc_thres(y_true, X, model,  thres = 0.7,label = ''):
    """Plot roc curve, y_true & model.predict_proba(X), ax --- axes of subplot"""
    fpr, tpr, thr = roc_curve(y_true, model.predict_proba(X)[:,1])
    idx = np.abs(tpr - thres).argmin()
    #print(label, ':  ', col,' ', tpr[idx], ' ', fpr[idx], ' ', thr[idx])
    return thr[idx]
