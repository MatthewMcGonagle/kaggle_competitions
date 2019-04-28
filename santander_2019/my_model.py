import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler

class BestDirectionsSquares:

    def __init__(self, n_directions = 30):
        self.sub_models = {'scaler' : StandardScaler(),
                           'target_means' : TargetMeans(),
                           'target_cov' : TargetCovariances(),
                           'ratio_directions' : None, # Ratio directions is initially None.
                           'weight_predictor' : LogisticRegression(solver = 'lbfgs'),
                            }         
        self.needs_fit = {sub_model : True for sub_model in self.sub_models.keys()}
        self.n_directions = n_directions

    def partial_fit(self, sub_model, X, y = None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if sub_model in ['scaler', 'target_means', 'target_cov']:
            self._partial_fit_predirections(sub_model, X, y)

        elif sub_model == 'ratio_directions':
            self.sub_models[sub_model] = RatioDirections(self.sub_models['target_cov'])
            self.needs_fit[sub_model] = False

        else:
            raise Exception(sub_model + ' isn\'t a valid sub-model of BestDirectionsModel, choices are ' + 
                            str(self.sub_models.keys()))

    def _partial_fit_predirections(self, sub_model, X, y):
        if y is None and sub_model != 'scaler':
            raise Exception('Training ' + sub_model + ' requires target class data.')

        if sub_model == 'scaler':
            transformers = []
        elif sub_model == 'target_means':
            transformers = ['scaler']
        elif sub_model == 'target_cov':
            transformers = ['scaler', 'target_means']

        if not self._transform_ready(transformers):
            raise Exception('Need to do atleast one partial fit on all of ' + transformers +
                            ' before doing any training on ' + sub_model) 

        for transformer in transformers: 
            X = self.sub_models[transformer].transform(X)

        if sub_model == 'target_cov':
            # Drop the mean difference direction.
            X = X[:, 1:]

        self.sub_models[sub_model].partial_fit(X, y)
        self.needs_fit[sub_model] = False

    def fit_directions(self):
        if self.needs_fit['target_means'] or self.needs_fit['target_cov']: 
            raise Exception("Need to fit target_means and target_cov before ratio directions")

        self.sub_models['ratio_directions'] = RatioDirections(self.sub_models['target_cov'])
        self.needs_fit['ratio_directions'] = False

    def reduce_directions(self, X):
        if self.needs_fit['scaler'] or self.needs_fit['target_means']:
            raise Exception("Need to fit scaler and target means before reducing dimensions of data")

        if isinstance(X, pd.DataFrame):
            X_ind = X.index
            X = X.values
        else:
            X_ind = None

        X = self.sub_models['scaler'].transform(X)
        X = self.sub_models['target_means'].transform(X)
        X_diff = X[:, 0].reshape(-1, 1)
        directions = self.sub_models['ratio_directions'].directions[:self.n_directions, :]
        X_coords = np.dot(X[:, 1:], directions.T)
        X_coords = np.concatenate([X_diff, X_coords], axis = 1)

        if not X_ind is None:
            X_coords = pd.DataFrame(X_coords, index = X_ind)

        return X_coords 

    def fit_squares(self, X_directions, y):
        if isinstance(X_directions, pd.DataFrame):
            X_directions = X_directions.values
        if isinstance(y, pd.Series):
            y = y.values

        X_squares = X_directions[:, 1:]**2 # Drop the mean direction square.
        self.sub_models['weight_predictor'].fit(X_squares, y) 
        self.needs_fit['weight_predictor'] = False

    def reduce_squares(self, X_directions):
        if self.needs_fit['weight_predictor']:
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
            if self.needs_fit[sub_model]:
                return False

        return True

    def fit_chunks(self, file_name, index_col, chunk_size, verbose = True):

        # Train the preprocessing sub-models.

        for sub_model in ['scaler', 'target_means', 'target_cov']: 
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

class TargetAccumulator:
    def __init__(self):
        self.sums = None
        self.counts = np.zeros(2)
        self.values = None
        self.calculations_updated = False

    def do_calculations(self):
        if self.sums is None:
            raise Exception("Can't do calculations if the transformer hasn't been fitted to any data.")
        self.values = self.sums / self.counts.reshape(-1, 1)
        self.calculations_updated = True

class TargetMeans(TargetAccumulator):

    def __init__(self):
        TargetAccumulator.__init__(self)
        self.direction = None

    def partial_fit(self, X, y):
        self.calculations_updated = False
        if self.sums is None:
            _, nFeatures = X.shape
            self.sums = np.zeros((2, nFeatures))

        for target in [0, 1]: 
            X_target = X[y == target]
            self.sums[target] += X_target.sum(axis = 0)
            self.counts[target] += len(X_target)

    def do_calculations(self):
        TargetAccumulator.do_calculations(self)
        self.direction = self.values[1] - self.values[0]
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.direction = self.direction.reshape(1, -1)

    def transform(self, X, y = None):
        if not self.calculations_updated:
            self.do_calculations()

        if isinstance(X, pd.DataFrame):
            X_ind = X.index
            X = X.values
        else:
            X_ind = None

        X = X - self.get_offset() 
        X_proj_coords = np.dot(X, self.direction.T)
        X_proj = np.dot(X_proj_coords, self.direction) 
        X_orthog = X - X_proj
        X_new = np.concatenate([X_proj_coords, X_orthog], axis = -1)

        if not X_ind is None:
            X_new = pd.DataFrame(X_new, index = X_ind)
        return X_new 

    def get_offset(self):
        return self.values[0].reshape(1, -1)

    def get_difference_direction(self):
        if not self.calculations_updated:
            self.do_calculations()
        return self.direction

class TargetCovariances(TargetAccumulator):

    def __init__(self):
        TargetAccumulator.__init__(self)
        #self.sums = None 
        #self.counts = np.zeros(2)
        #self.values = None
        #self.calculations_updated = False

    def partial_fit(self, X, y):
        '''
        Parameters:
        X : Numpy array of shape (n_samples, n_features + 1)
            The first features X[:, 0] should be the projection along the direction between class means after offsetting
            by a choice of class mean. The rest of the features X[:, 1 : n_features + 1] should be the result of removing the
            previous offset and then removing the projection along the direction of the difference in class means. 
            This matches the output of TargetMeans.transform().
        '''
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.calculations_updated = False 

        if self.sums is None:
            _, nFeatures = X.shape # First column is projection to mean target differences. We skip this.
            self.sums = np.zeros((2, nFeatures, nFeatures)) 

        for target in [0, 1]:
            on_target = (y == target)
            X_target = X[on_target, :] 
            X_products = np.dot(X_target.T, X_target)
            self.sums[target] += X_products
            self.counts[target] += on_target.sum() 

    def do_calculations(self):
        if self.sums is None:
            raise Exception("Need to fit transformer on data before doing calculations.")

        self.values = self.sums / self.counts.reshape(-1, 1, 1)
        self.calculations_updated = True

    def get_class_covariances(self):
        if not self.calculations_updated:
            self.do_calculations()
            self.calculations_updated = True

        return self.values

    def get_variances(self, directions):
        class_cov = self.get_class_covariances()

        variances = np.zeros((2, len(directions)))
        for target in [0, 1]:
            squares = np.dot(directions, class_cov[target]) 
            squares = [np.dot(sq, direction) for sq, direction in zip(squares, directions)]
            variances[target] = np.array(squares)
        return variances

def find_inverse_square_root(matrix, regularity = 1e-6):
    eig_vals, eig_vecs = np.linalg.eigh(matrix)
    eig_vecs = eig_vecs.T # Switch to row vectors.
   
    good_vals = eig_vals > regularity 
    eig_vals = eig_vals[good_vals]
    root_eig_vals = np.sqrt(eig_vals) 
    inverse_roots = 1 / root_eig_vals
    eig_vecs = eig_vecs[good_vals] 
   
    inverse_square_root = np.dot(np.diag(inverse_roots), eig_vecs) 
    inverse_square_root = np.dot(eig_vecs.T, inverse_square_root)
    return inverse_square_root 

class RatioDirections:
    '''
    Find the principal directions for ratios of in-class variances between classes.

    Members
    -------
    n_features : Int
    directions : Numpy Array of shape (n_features, n_features)
        Each row is a unit direction vector.
    ratios : Numpy Array of shape (n_features)
    rankings : Numpy Array of shape (n_features)
        Rankings for the directions based on the ratios.
    variances : Numpy Array of shape (2, n_features)
        The variances of each class along each direction.
    '''

    def __init__(self, target_cov = None, eig_values_cut = 0.02):
  
        self.eig_values_cut = eig_values_cut
        if target_cov is None: 
            self.n_features = None 
            self.ratios = None
            self.directions = None
            self.rankings = None
            self.variances = None
        else:
            self.fit(target_cov)

    def fit(self, target_cov):
        if not target_cov.calculations_updated:
            target_cov.do_calculations()
     
        self.n_features, _, _ = target_cov.values.shape

        inv_square_root0 = find_inverse_square_root(target_cov.values[0]) 
        ratio_matrix = np.dot(inv_square_root0.T, target_cov.values[1])
        ratio_matrix = np.dot(ratio_matrix, inv_square_root0)
        self.ratios, self.directions = np.linalg.eigh(ratio_matrix) 
        self.directions = self.directions.T

        self.rankings = RatioDirections._get_rankings(self.ratios, self.directions, target_cov) 
        order_ind = np.argsort(self.rankings)[::-1] # Make sure to put in descending order.

        # Reorder according to descending order of self.rankings. Remove the first as it is the
        # mean direction.
        self.rankings = self.rankings[order_ind][1:]
        self.ratios = self.ratios[order_ind][1:]
        self.directions = self.directions[order_ind, :][1:, :]        

        self.variances = target_cov.get_variances(self.directions) # Get the variance of each target 
                                                                   # along eigen directions.

    def _get_rankings(eig_values, eig_vecs, target_cov, regularity = 1e-6):
        '''
        Rankings are determined by size of eigenvalues relative to the MINIMUM of
        the in-class variances in the eigendirection (minimum taken over classes).
        '''
        rankings = np.abs(np.log(eig_values + regularity))
        return rankings 

    def transform(self, X, use_scalings = False):
   
        if isinstance(X, pd.DataFrame):
            X = X.values 
        X_coords = np.dot(X, self.directions.T) 

        # Recall that one of our basis vectors isn't an eigen vector.
        if use_scalings:
            scalings = 1 / np.sqrt(self.variances[0]).reshape(1, -1) 
            X_coords /= scalings
        return X_coords 

def quick_tune_reduced(X, n_components):

    nPoints, nFeatures = X.shape
    if n_components > nFeatures or n_components == 0:
        raise Exception

    n_remove = nFeatures - n_components
    if n_remove == 0:
        return X

    X_keep = X[:, :n_components - 1].copy()
    X_removed = X[:, n_components - 1 : nFeatures - 1]
    X_size = np.linalg.norm(X_removed, axis = 1).reshape(-1, 1)**2
    X_size = X[:, -1].reshape(-1, 1)**2 + X_size
    X_size = np.sqrt(X_size)

    X_new = np.concatenate([X_keep, X_size], axis = -1)
    return X_new 

