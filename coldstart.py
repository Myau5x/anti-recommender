import pandas as pd
import numpy as np

def expand_factors(X, col='features', inplace=False):
    X = X.copy() if not inplace else X
    X[col] = X[col].map(lambda x: x[1:-1]).str.split(',')

    for i in range(len(X[col].iloc[0])):
        X['feature_' + str(i)] = X[col].apply(lambda x: float(x[i]))

    X.drop(col, axis=1, inplace=True)
    return X

def similar_rest(r1, r2):
    ''' calculate similarity beetween 2 Restaurants
        based on category, price_range , mean_review
    '''
    sc =  2*similar_cat(r1['categories'], r2['categories'])
    sr = similar_review(r1['stars'], r2['stars'])
    sr_c = similar_review_count(r1['rat_to_rev'], r2['rat_to_rev'])
    p1 = r1['RestaurantsPriceRange2']
    p2 = r2['RestaurantsPriceRange2']
    sp = similar_price_range(p1,p2)
    return sc+sr+sr_c+sp

def find_k_similar(train, row, k=98.5, rank = 6):
    x = train.apply(similar_rest, axis = 1,r2= row)
    feat_c = train.columns[-rank:]

    return train[feat_c][(x >np.percentile(x, k))].mean(axis = 0)

def features_for_new(X, train, rank = 6):
    X_f = pd.dataFrame()
    X_f[feat_c] = train.columns[-rank:]
    pass


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
