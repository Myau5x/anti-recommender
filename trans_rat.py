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
        lbls = self.model_feature
        for lbl in lbls:
            X[lbl] = X[col].apply(lambda x: lbl in x)
        X['num_of_categ'] = X[col].map(lambda x: len(x.split(',')))
        X.drop(col, axis=1, inplace=True)
        return X
