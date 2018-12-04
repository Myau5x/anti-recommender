import pandas as pd
import numpy as np

class FillPrice:
    def __init__(self, priceCol = 'RestaurantsPriceRange2'):
        self.priceCol = priceCol

    def transform(self, X, inplace = False):
        X = X.copy() if not inplace else X
        X['rest_isna'] = pd.isnull(X[priceCol])
        X[priceCol][pd.isnull(X[priceCol])] = 2
        return X


class CategoriesExpand:
    """transform categories to dummy variables"""

    def __init__(self, model_features):
        self.model_features = model_features

    def transform(self,X, col='categories', inplace=False):
        X = X.copy() if not inplace else X
        lbls = self.model_features
        for lbl in lbls:
            X[lbl] = X[col].apply(lambda x: lbl in x)
        X['num_of_categ'] = X[col].map(lambda x: len(x.split(',')))
        X.drop(col, axis=1, inplace=True)
        return X
#'categories': [{'alias': 'hawaiian', 'title': 'Hawaiian'},

class CategoriesTransformScr:
    """transform categories to dummy variables"""

    def __init__(self, model_features):
        self.model_features = model_features

    def transform(self,X, col='categories', inplace=False):
        X = X.copy() if not inplace else X
        X[col] = X[col].map(lambda x:  list(map(lambda y : y['title'],x)))
        lbls = self.model_features
        for lbl in lbls:
            X[lbl] = X[col].map(lambda x: lbl in x)
        X['num_of_categ'] = X[col].map(lambda x: len(x))
        X.drop(col, axis=1, inplace=True)
        return X


class RatingNormalizer:
    def __init__(self, revCol = 'review_count', starCol = 'stars', n =250):
        self.n = n
        self.revCol = revCol
        self.starCol = starCol

    def transform(self, X, inplace = False):
        X = X.copy() if not inplace else X
        rev_norm = X[self.revCol].map(lambda x: x if x < self.n else self.n)
        X['st_over_am'] = X[self.starCol]/rev_norm

#Pipeline = Pipeline([FillPrice, CategoriesExpand, RatingNormalizer ])
if (__name__ == "__main__"):
    print('it is ok')
