import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score, roc_curve


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

#pred[act <3.5].mean(), base[act<3.5].mean()
###(3.1792809133547317, 3.4462038511106057)
