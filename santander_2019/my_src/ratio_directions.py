import numpy as np
import pandas as pd

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

    def __init__(self, class_cov = None, eig_values_cut = 0.02, use_scalings = True):
  
        self.eig_values_cut = eig_values_cut
        self.use_scalings = use_scalings
        self._class_cov = class_cov

        self.n_features = None 
        self.ratios = None
        self.directions = None
        self.rankings = None
        self.variances = None
        self._fitted = False

    def set_class_covariances(self, class_cov):
        self._class_cov = class_cov
        self._fitted = False

    def fit(self):
        if self._class_cov is None:
            raise Exception('Call set_class_covariances() before fitting an instance of' + 
                            ' RatioDirections if class_cov is None.')
        class_cov = self._class_cov.get_values()
     
        self.n_features, _, _ = class_cov.shape

        inv_square_root0 = find_inverse_square_root(class_cov[0]) 
        ratio_matrix = np.dot(inv_square_root0.T, class_cov[1])
        ratio_matrix = np.dot(ratio_matrix, inv_square_root0)
        self.ratios, self.directions = np.linalg.eigh(ratio_matrix) 
        self.directions = self.directions.T

        self.rankings = RatioDirections._get_rankings(self.ratios) 
        order_ind = np.argsort(self.rankings)[::-1] # Make sure to put in descending order.

        # Reorder according to descending order of self.rankings. Remove the first as it is the
        # mean direction.
        self.rankings = self.rankings[order_ind][1:]
        self.ratios = self.ratios[order_ind][1:]
        self.directions = self.directions[order_ind, :][1:, :]        

        self.variances = self._class_cov.get_variances(self.directions) # Get the variance of each target 
                                                                   # along eigen directions.

        self._fitted = True

    @staticmethod
    def _get_rankings(ratios, regularity = 1e-6):
        '''
        Rankings are determined by size of eigenvalues relative to the MINIMUM of
        the in-class variances in the eigendirection (minimum taken over classes).
        '''
        rankings = np.abs(np.log(ratios + regularity))
        return rankings 

    def transform(self, X, use_scalings = False):
        if not self._fitted:
            self.fit()
 
        need_pandas = False
        if isinstance(X, pd.DataFrame):
            need_pandas = True
            X = X.values 
            X_ind = X.index

        X_coords = np.dot(X, self.directions.T) 

        # Recall that one of our basis vectors isn't an eigen vector.
        if self.use_scalings:
            scalings = 1 / np.sqrt(self.variances[0]).reshape(1, -1) 
            X_coords /= scalings

        if need_pandas:
            X_coords = pd.DataFrame(X_coords, index = X_ind)

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

# Global list of models.
models = [RatioDirections]

