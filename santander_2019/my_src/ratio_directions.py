'''
Classes for determining the optimal directions based on the ratios of the variances of each class.
'''

import numpy as np
import pandas as pd

def find_inverse_square_root(matrix, regularity = 1e-6):
    '''
    Find the inverse square root of a positive definite symmetric square matrix. Adds in a
    regularity term for numerical stability.

    Parameters
    ----------
    matrix : Numpy matrix of shape (n_features, n_features)
        Matrix to be inverted and find square rooted; should be postive definite and symmetric.

    regularity : scalar
        For numerical stability, actually find the inverse square root of
            matrix + regularity * I
        where I is the identity matrix of shape (n_features, n_features).

    Returns
    -------
    inverse_square_root : Numpy matrix of shape (n_features, n_features)
        The inverse square root.
    '''

    # We use an eigenvalue, eigenvector decomposition to find the inverse square root.
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
    Transformer that finds the optimal directions for the ratios of the in-class variances.

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

    def __init__(self, class_cov = None, use_scalings = True):
        '''
        Initialize parameters of the transformer.

        Parameters
        ----------
        class_cov : class_stats.ClassCovariances
            The in-class covariances to use to find the optimal directions.

        use_scalings : Boolean
            When transforming, should we scale each direction to make the variance of class 0 to
            be uniform, i.e. 1.
        '''
        self.use_scalings = use_scalings
        self._class_cov = class_cov

        self.n_features = None
        self.ratios = None
        self.directions = None
        self.rankings = None
        self.variances = None
        self._fitted = False

    def set_class_covariances(self, class_cov):
        '''
        Set the in-class covariances that are used for fits.

        Parameters
        ----------
        class_cov : class_stats.ClassCovariances
            The in-class covariances to use for the ratio directions fit.
        '''
        self._class_cov = class_cov
        self._fitted = False

    def fit(self):
        '''
        Fit the directions that are optimal for the ratio of in-class variances.
        '''
        if self._class_cov is None:
            raise Exception('Call set_class_covariances() before fitting an instance of' +
                            ' RatioDirections if class_cov is None.')
        class_cov = self._class_cov.get_values()

        self.n_features, _, _ = class_cov.shape

        # We look at the ratio of quadratic forms
        # (direction.T * covariance_class_1 * direction) / (direction.T * covariance_class_0 * direction).
        # This amounts to finding the inverse square root inv_sqrt(covariance_class_0), and then
        # studying the eigenvalue decomposition of
        # M = inv_sqrt(covariance_class_0).T * covariance_class_1 * inv_sqrt(covariance_class_0).
        # Our optimal directions are the eigenvectors of M.

        inv_square_root0 = find_inverse_square_root(class_cov[0])
        ratio_matrix = np.dot(inv_square_root0.T, class_cov[1])
        ratio_matrix = np.dot(ratio_matrix, inv_square_root0)
        self.ratios, self.directions = np.linalg.eigh(ratio_matrix)
        self.directions = self.directions.T

        # Rank the ratios according to importance.
        self.rankings = RatioDirections._get_rankings(self.ratios)
        order_ind = np.argsort(self.rankings)[::-1] # Make sure to put in descending order.

        # Reorder according to descending order of self.rankings. Remove the first as it is the
        # mean direction.
        self.rankings = self.rankings[order_ind][1:]
        self.ratios = self.ratios[order_ind][1:]
        self.directions = self.directions[order_ind, :][1:, :]

        self.variances = self._class_cov.get_variances(self.directions) # Get the variance of each
                                                                        # target along eigen
                                                                        # directions.

        self._fitted = True

    @staticmethod
    def _get_rankings(ratios, regularity = 1e-6):
        '''
        Rankings are determined by how far the ratio differs from 1.0. That is we look at the
        absolute value of the log of the ratio. The larger this value (and so further from 1.0)
        the more important the ratio is.

        Returns
        -------
        rankings : Numpy array of size n_features
            The unsorted rankings of each ratio.
        '''
        rankings = np.abs(np.log(ratios + regularity))
        return rankings

    def transform(self, X):
        '''
        Find the coordinates along the optimal directions.

        Parameters
        ----------
        X : Numpy array or pandas.DataFrame of shape (n_samples, n_features)
            The sample features to transform.

        Returns
        -------
        direction_coordinates : Numpy array or pandas.DataFrame of shape (n_samples, n_features)
            The coordinates along each optimal ratio direction. If the original data is of
            instance pandas.DataFrame, then this will be as well (with the same index as X).
        '''
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

# Global list of models.
MODELS = [RatioDirections]
