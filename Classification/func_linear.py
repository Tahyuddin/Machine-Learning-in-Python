import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.6f} (std: {1:.6f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


class Dropper(BaseEstimator, TransformerMixin):

    def __init__(self, var):
        self.var = var

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop([self.var], axis=1)
        return X


class convert_to_numeric(BaseEstimator, TransformerMixin):

    def __init__(self, var):
        self.var = var

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.var] = pd.to_numeric(X[self.var], errors='coerce')
        return X


class create_dummy(BaseEstimator, TransformerMixin):

    def __init__(self, var, freq_cutoff=0):
        self.var = var
        self.freq_cutoff = freq_cutoff
        self.var_cat_dict = {}
        self.feature_names = []

    def fit(self, X, y=None):
        k = X[self.var].value_counts()
        if (k <= self.freq_cutoff).sum() == 0:
            cats = k.index[:-1]
        else:
            cats = k.index[k > self.freq_cutoff]
        self.var_cat_dict[self.var] = cats
        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                self.feature_names.append(col + '_' + str(cat))
        return self

    def transform(self, X, y=None):
        dummy_data = X.copy()
        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name = col + '_' + str(cat)
                dummy_data[name] = (dummy_data[col] == cat).astype(int)
            del dummy_data[col]
        return dummy_data


class Imputer_num(BaseEstimator, TransformerMixin):

    def __init__(self, var, mean=True):
        self.var = var
        self.value = 0
        self.mean = mean

    def fit(self, X, y=None):
        if self.mean:
            self.value = X[self.var].mean()
        else:
            self.value = X[self.var].median()
        return self

    def transform(self, X, y=None):
        X[self.var] = X[self.var].fillna(self.value)
        return X


class Imputer_cat(BaseEstimator, TransformerMixin):

    def __init__(self, var, value='missng'):
        self.var = var
        self.value = value
        self.mode = ''

    def fit(self, X, y=None):
        self.mode = self.value = X[self.var].mode()

    def transform(self, X, y=None):
        if self.value is None:
            X[self.var].fillna(self.mode, inplace=True)
        else:
            X[self.var].fillna(self.value, inplace=True)
        return X


class remove_pct(BaseEstimator, TransformerMixin):

    def __init__(self, var):
        self.var = var

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.var] = X[self.var].str.replace('%', '')
        return X


class Fico(BaseEstimator, TransformerMixin):
    def __init__(self, var):
        self.var = var

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        a = X[self.var].str.split('-', expand=True)
        a[2] = (a[0].astype('int') + a[1].astype('int')) / 2
        X['Fico'] = a[2]
        del X[self.var]
        return X


class custom(BaseEstimator, TransformerMixin):

    def __init__(self, var, pairs):
        self.var = var
        self.pairs = pairs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        x1 = []
        for i in X[self.var]:
            try:
                x1.append(self.pairs[i])
            except KeyError:
                x1.append(0)
        X['alpha'] = x1
        return X