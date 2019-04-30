import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler

from my_src import class_stats
from my_src import ratio_directions

class BestDirectionsSquares:
    '''
    Arrange directions in decreasing order of ratio of covariance target 1 to covariance target2 in
    that direction. The covariances are computed after removing the class means.
    '''
    def __init__(self, n_directions = 30):
        self.sub_models = {'scaler' : StandardScaler(),
                           'class_means' : class_stats.ClassMeans(),
                           'class_cov' : class_stats.ClassCovariances(),
                           'ratio_directions' : ratio_directions.RatioDirections(), 
                           'weight_predictor' : LogisticRegression(solver = 'lbfgs'),
                            }         
        self.sub_models['class_cov'].set_class_means(self.sub_models['class_means'])
        self.sub_models['ratio_directions'].set_class_covariances(self.sub_models['class_cov'])

        self._needs_fit = {sub_model : True for sub_model in self.sub_models.keys()}
        self.n_directions = n_directions

    def partial_fit(self, sub_model, X, y = None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if sub_model in ['scaler', 'class_means', 'class_cov']:
            self._partial_fit_predirections(sub_model, X, y)

        elif sub_model == 'ratio_directions':
            self.sub_models[sub_model].fit()
            self._needs_fit[sub_model] = False

        else:
            raise Exception(sub_model + ' isn\'t a valid sub-model of BestDirectionsModel, choices are ' + 
                            str(self.sub_models.keys()))

    def _partial_fit_predirections(self, sub_model, X, y):
        if y is None and sub_model != 'scaler':
            raise Exception('Training ' + sub_model + ' requires target class data.')

        # Check that previous transformers have trained at least once.
        for transformer in ['scaler', 'class_means', 'class_cov']:
            if transformer == sub_model:
                break
            elif self._needs_fit[transformer]:
                raise Exception('Need to do atleast one partial fit of ' + transformer +
                                ' before doing any training on ' + sub_model) 

        if sub_model != 'scaler':
            X = self.sub_models['scaler'].transform(X)

        self.sub_models[sub_model].partial_fit(X, y)
        self._needs_fit[sub_model] = False

    def fit_directions(self):
        if self._needs_fit['class_cov']: 
            raise Exception("Need to fit class_cov before ratio directions")

        self.sub_models['ratio_directions'].fit()
        self._needs_fit['ratio_directions'] = False

    def reduce_directions(self, X):
        for sub_model in ['scaler', 'class_means', 'ratio_directions']:
            if self._needs_fit[sub_model]:
                raise Exception('Need to fit ' + sub_model + ' before reducing dimensions of data features.')

        need_pandas = False
        if isinstance(X, pd.DataFrame):
            need_pandas = True
            X_ind = X.index
            X = X.values

        X = self.sub_models['scaler'].transform(X)
        X = self.sub_models['class_means'].transform(X)
        X_diff = X[:, 0].reshape(-1, 1)
        directions = self.sub_models['ratio_directions'].directions[:self.n_directions, :]
        X_coords = np.dot(X[:, 1:], directions.T)
        X_coords = np.concatenate([X_diff, X_coords], axis = 1)

        if need_pandas:
            X_coords = pd.DataFrame(X_coords, index = X_ind)

        return X_coords 

    def fit_squares(self, X_directions, y):
        if isinstance(X_directions, pd.DataFrame):
            X_directions = X_directions.values
        if isinstance(y, pd.Series):
            y = y.values

        X_squares = X_directions[:, 1:]**2 # Drop the mean direction square.
        self.sub_models['weight_predictor'].fit(X_squares, y) 
        self._needs_fit['weight_predictor'] = False

    def reduce_squares(self, X_directions):
        if self._needs_fit['weight_predictor']:
            raise Exception('Need to fit weight predictor before reducing to squares')

        if isinstance(X_directions, pd.DataFrame):
            X_ind = X_directions.index
            X_directions = X_directions.values
        else:
            X_ind = None

        X_squares = X_directions[:, 1:]**2 # Drop the mean direction square. 
        weights = self.sub_models['weight_predictor'].coef_
        X_squares = np.dot(X_squares, weights.T) #.reshape(-1, 1)
        X_squares = np.concatenate([X_directions[:, [0]], X_squares], axis = 1)

        if not X_ind is None:
            X_squares = pd.DataFrame(X_squares, index = X_ind)
        return X_squares 

    def _transform_ready(self, sub_models):
        for sub_model in sub_models:
            if self._needs_fit[sub_model]:
                return False

        return True

    def fit_chunks(self, file_name, index_col, chunk_size, verbose = True):

        # Train the preprocessing sub-models.

        for sub_model in ['scaler', 'class_means', 'class_cov']: 
            if verbose:
                print('Training ' + sub_model + '\'s')
            reader = pd.read_csv(file_name, index_col = index_col, chunksize = chunk_size) 

            for i, df in enumerate(reader):
                if verbose:
                    print('Chunk', i, end = ', ')
                X_train = df.drop('target', axis = 1)
                y_train = df['target']
                self.partial_fit(sub_model, X_train, y_train)

            if verbose:
                print(sub_model + '\'s finished training')

        if verbose:
            print('Fitting directions')
        self.fit_directions()

        # Do the reduction to best directions.

        if verbose:
            print('Reducing to directions')
        reader = pd.read_csv('data/train.csv', index_col = 'ID_code', chunksize = chunk_size)

        X_reduced = []
        y_train = []
        for i, df in enumerate(reader):
            if verbose:
                print('Chunk', i, end = '')
            X_train = df.drop('target', axis = 1)
            X_train = self.reduce_directions(X_train)
            X_reduced.append(X_train)
            y_train.append(df['target'])
               
            if verbose: 
                print(end = ', ')
        X_reduced = pd.concat(X_reduced)
        y_train = pd.concat(y_train)

        # Train the weights.
        self.fit_squares(X_reduced, y_train)

       
    def transform(self, X):
        X = self.reduce_directions(X)
        X = self.reduce_squares(X)
        return X

# List of classes that are models. Useful for JSON encoding/decoding.
models = [BestDirectionsSquares, StandardScaler, LogisticRegression]

