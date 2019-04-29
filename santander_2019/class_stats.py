import pandas as pd
import numpy as np

class ClassAccumulator:
    def __init__(self):
        self._sums = None
        self._counts = np.zeros(2)
        self._values = None
        self._calculations_updated = False

    def _update_calculations(self):
        if self._sums is None:
            raise Exception("Can't do calculations if the transformer " + self.__class__.__name__ +
                            " hasn't been fitted to any data.")
        counts_shape = (-1,) + tuple(1 for _ in self._sums.shape)[1:]
        self._values = self._sums / self._counts.reshape(counts_shape)
        self._calculations_updated = True

    def get_values(self):
        if not self._calculations_updated:
            self._update_calculations()
        return self._values

class ClassMeans(ClassAccumulator):
    '''
    Do a rolling calculation of class means using partial fits. Transform by shifting by an offset
    given by one of the class means, then find the projection coordinate along the difference of the
    means direction and the original coordinates of the part orthogonal to the mean direction.

    Class members should be accessed with the interface, because we want to ensure that the calculations
    are up to date before accessing them.
    '''
    def __init__(self):
        ClassAccumulator.__init__(self)
        self._direction = None

    def partial_fit(self, X, y):
        self._calculations_updated = False
        if self._sums is None:
            _, n_features = X.shape
            self._sums = np.zeros((2, n_features))

        for target in [0, 1]:
            X_target = X[y == target]
            self._sums[target] += X_target.sum(axis = 0)
            self._counts[target] += len(X_target)

    def _update_calculations(self):
        ClassAccumulator._update_calculations(self)
        self._direction = self._values[1] - self._values[0]
        self._direction = self._direction / np.linalg.norm(self._direction)
        self._direction = self._direction.reshape(1, -1)

    def transform(self, X, y = None):
        # Parent class will raise exception if not fits have been done yet.
        if not self._calculations_updated:
            self._update_calculations()

        # Handle pandas.DataFrame.
        if isinstance(X, pd.DataFrame):
            X_ind = X.index
            X = X.values
            need_dataframe = True
        else:
            X_ind = None
            need_dataframe = False

        # Subtract by offset and project onto mean direction.
        X = X - self._unsafe_get_offset()
        X_proj_coords = np.dot(X, self._direction.T)
        X_proj = np.dot(X_proj_coords, self._direction)
        X_orthog = X - X_proj

        # Final features are the projection coordinates and the original
        # coordinates of the part orthogonal to the mean direction.
        X_new = np.concatenate([X_proj_coords, X_orthog], axis = -1)

        if need_dataframe:
            X_new = pd.DataFrame(X_new, index = X_ind)
        return X_new

    def get_offset(self):
        if not self._calculations_updated:
            self._update_calculations()
        return self._unsafe_get_offset()

    def _unsafe_get_offset(self):
        return self._values[0].reshape(1, -1)

    def get_difference_direction(self):
        if not self._calculations_updated:
            self._update_calculations()
        return self._direction

class ClassCovariances(ClassAccumulator):
    def __init__(self, class_means = None):
        ClassAccumulator.__init__(self)
        self._class_means = class_means

    def set_class_means(self, class_means):
        self._class_means = class_means

    def partial_fit(self, X, y, do_class_means_transform = True):
        '''
        Parameters:
        X : Numpy array of shape (n_samples, n_features)
            The features to fit.
        '''

        if self._class_means is None:
            raise Exception("Need to call set_class_means() before fitting since self.class_means is None")

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Make sure to get n_features of original data if this is the first time fitting. Make sure
        # to do this before transforming using self.class_means if it is not None.
        if self._sums is None:
            _, n_features = X.shape # First column is projection to mean target differences. We skip this.
            self._sums = np.zeros((2, n_features, n_features))

        if do_class_means_transform:
            X = self._class_means.transform(X)
            X = X[:, 1:] # Drop the class difference direction projection coordinate.

        self._calculations_updated = False

        for target in [0, 1]:
            on_target = (y == target)
            X_target = X[on_target, :]
            X_products = np.dot(X_target.T, X_target)
            self._sums[target] += X_products
            self._counts[target] += on_target.sum()

    def get_class_covariances(self):
        if not self._calculations_updated:
            self._update_calculations()

        return self._values

    def get_variances(self, directions):

        class_cov = self.get_class_covariances()

        variances = np.zeros((2, len(directions)))
        for target in [0, 1]:
            squares = np.dot(directions, class_cov[target])
            squares = [np.dot(sq, direction) for sq, direction in zip(squares, directions)]
            variances[target] = np.array(squares)
        return variances
