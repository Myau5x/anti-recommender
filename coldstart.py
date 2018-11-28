import pandas as pd
import numpy as np

def expand_features(X, col='features', inplace=False, sep=None):
    X = X.copy() if not inplace else X
    if sep==None:
        for i in range(len(X[col].iloc[0])):
            X['feature_' + str(i)] = X[col].apply(lambda x: x[i])
    else:
        lbls = np.unique(sep.join(X[col].unique()).split(sep))
        for lbl in lbls:
            X[lbl] = X[col].apply(lambda x: lbl in x)
    X.drop(col, axis=1, inplace=True)
    return X

def similar_rest(r1, r2):
    ''' calculate similarity beetween 2 Restaurants
        based on category, price_range , mean_review
    '''
    sc =  similar_cat(r1['categories'], r2['categories'])
    sr = similar_review(r1['stars'], r2['stars'])
    sr_c = similar_review_count(r1['rat_to_rev'], r2['rat_to_rev'])
    p1 = r1['RestaurantsPriceRange2']
    p2 = r2['RestaurantsPriceRange2']
    sp = similar_price_range(p1,p2)
    return sc+sr+sr_c+sp

def similar_cat(c1,c2):
    set1 = set(c1)
    set2 = set(c2)
    if c1 is None or c2 is None:
        return 0
    else:
        return len(set1.intersection(set2))/len(set1.union(set2))

def similar_price_range(p1, p2):
    return abs(p2 -p1)/3

def similar_review(r1,r2):
    return abs(r1-r2)/4

def similar_review_count(c1,c2):
    return abs(c1-c2)/2
