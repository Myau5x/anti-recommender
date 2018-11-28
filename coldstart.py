import pandas as pd


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
