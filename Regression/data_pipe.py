import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def top_models(results, n_top=3):
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


class FillMissing(BaseEstimator, TransformerMixin):

    def __init__(self, var, default_value_str='missing'):
        self.var = var
        self.default_value_str = default_value_str
        self.learner = {}

    def fit(self, X, y=None):
        if X[self.var].dtype == 'O':
            self.learner[self.var] = self.default_value_str
        else:
            self.learner[self.var] = X[self.var].median()
        return self

    def transform(self, X, y=None):
        X[self.var] = X[self.var].fillna(self.learner[self.var])
        return X


class Custom_func(BaseEstimator, TransformerMixin):

    def __init__(self, func, var):
        self.func = func
        self.var = var

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_X = self.func(X, self.var)
        return new_X


class date_components(BaseEstimator, TransformerMixin):

    def __init__(self, var):
        self.var = var

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.var] = pd.to_datetime(X[self.var])

        X[f'{self.var}_year'] = X[self.var].dt.year
        X[f'{self.var}_year'] = X[f'{self.var}_year'].fillna(X[f'{self.var}_year'].mean())

        X[f'{self.var}_month_sin'] = np.sin(2 * np.pi * X[self.var].dt.month / 12)
        X[f'{self.var}_month_sin'] = X[f'{self.var}_month_sin'].fillna(X[f'{self.var}_month_sin'].mean())

        X[f'{self.var}_month_cos'] = np.cos(2 * np.pi * X[self.var].dt.month / 12)
        X[f'{self.var}_month_cos'] = X[f'{self.var}_month_cos'].fillna(X[f'{self.var}_month_cos'].mean())

        X[f'{self.var}_day_sin'] = np.sin(2 * np.pi * X[self.var].dt.day / 31)
        X[f'{self.var}_day_sin'] = X[f'{self.var}_day_sin'].fillna(X[f'{self.var}_day_sin'].mean())

        X[f'{self.var}_day_cos'] = np.cos(2 * np.pi * X[self.var].dt.day / 31)
        X[f'{self.var}_day_cos'] = X[f'{self.var}_day_cos'].fillna(X[f'{self.var}_day_cos'].mean())

        X[f'{self.var}_weekday_sin'] = np.sin(2 * np.pi * X[self.var].dt.weekday / 7)
        X[f'{self.var}_weekday_sin'] = X[f'{self.var}_weekday_sin'].fillna(X[f'{self.var}_weekday_sin'].mean())

        X[f'{self.var}_weekday_cos'] = np.cos(2 * np.pi * X[self.var].dt.weekday / 7)
        X[f'{self.var}_weekday_cos'] = X[f'{self.var}_weekday_cos'].fillna(X[f'{self.var}_weekday_cos'].mean())

        X[f'{self.var}_hour_sin'] = np.sin(2 * np.pi * X[self.var].dt.hour / 24)
        X[f'{self.var}_hour_sin'] = X[f'{self.var}_hour_sin'].fillna(X[f'{self.var}_hour_sin'].mean())

        X[f'{self.var}_hour_cos'] = np.cos(2 * np.pi * X[self.var].dt.hour / 24)
        X[f'{self.var}_hour_cos'] = X[f'{self.var}_hour_cos'].fillna(X[f'{self.var}_hour_cos'].mean())

        X[f'{self.var}_minute_sin'] = np.sin(2 * np.pi * X[self.var].dt.minute / 60)
        X[f'{self.var}_minute_sin'] = X[f'{self.var}_minute_sin'].fillna(X[f'{self.var}_minute_sin'].mean())

        X[f'{self.var}_minute_cos'] = np.cos(2 * np.pi * X[self.var].dt.minute / 60)
        X[f'{self.var}_minute_cos'] = X[f'{self.var}_minute_cos'].fillna(X[f'{self.var}_minute_cos'].mean())

        X = X.drop([self.var], axis=1)
        return X


class Daydiffs(BaseEstimator, TransformerMixin):

    def __init__(self, early_date_var, recent_date_var):
        self.early_date_var = early_date_var
        self.recent_date_var = recent_date_var

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[f'date_diff_{self.recent_date_var}_and_{self.early_date_var}'] = (
                X[self.recent_date_var] - X[self.early_date_var]).dt.days
        return X


class TextVariable(BaseEstimator, TransformerMixin):

    def __init__(self, col, feature_count=100):
        self.col = col
        self.tfidfs = {}
        self.feature_count = feature_count

    def fit(self, X, y=None):
        self.tfidfs[self.col] = TfidfVectorizer(analyzer='word', stop_words='english',
                                                token_pattern=r'(?u)\b[A-Za-z]+\b', min_df=0.01, max_df=0.8,
                                                max_features=self.feature_count)
        self.tfidfs[self.col].fit(X[self.col])

        return self

    def transform(self, X):
        datasets = pd.DataFrame(data=self.tfidfs[self.col].transform(X[self.col]).toarray(),
                                columns=[self.col + '_' + _ for _ in
                                         list(self.tfidfs[self.col].get_feature_names_out())])

        return pd.concat([X, datasets], axis=1)
