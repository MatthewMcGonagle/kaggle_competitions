'''
Classes that allow us to do kfold validation and other test/train splits of data that is split
into chunks.

It takes a while to read each chunk of data from disk, so it makes sense to do partial fits on
splits for each chunk (as opposed to iterating over each split and then each chunk). These
methods allow us to consistently across different training sessions of transformers in
pipeline, even if we use shuffling in our splits.
'''
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

class SplitReaderIndices:
    '''
    Keep track of indices of splits of file reader chunks (chunks of data).
    '''
    def __init__(self, filename, index_col, chunk_size, target_name = 'target',
                 splitter = StratifiedKFold(5), verbose = True):
        '''
        Get the indices of splits for chunks of data from CSV file.

        Parameters
        ----------
        filename : String
            Name of Comma Separated Values that data chunks are read from.

        index_col : String
            The name of the column that will be used for the indices of the data.

        chunk_size : Int
            The (maximum) number of lines to read for each chunk.

        target_name : String
            The name of the column that be used for the target.

        splitter : sklearn.model_selection splitter
            The splitter to use to split each data chunk.

        verbose : Boolean, default = True
            Whether to print information as indices are made.
        '''
        self.filename = filename
        self.index_col = index_col
        self.chunk_size = chunk_size
        self.target_name = target_name
        self.splitter = splitter

        # Get the indices of the chunk splits.

        reader = pd.read_csv(filename, index_col = index_col, chunksize = chunk_size)
        chunk_train = []
        chunk_test = []

        if verbose:
            print('Splitting chunks: ', end = '')

        # For each chunk, get the indices of the training and testing data for each split of
        # the chunk.
        for chunk_i, chunk in enumerate(reader):
            if verbose:
                print(chunk_i, end = ', ')

            x = chunk.index
            y = chunk[target_name]
            new_train_kfold = []
            new_test_kfold = []
            for train_i, test_i in splitter.split(x, y):
                new_train_kfold.append(x[train_i])
                new_test_kfold.append(x[test_i])
            chunk_train.append(new_train_kfold)
            chunk_test.append(new_test_kfold)

        self.chunk_train = chunk_train
        self.chunk_test = chunk_test

    def __iter__(self):
        '''
        Get an iterator to iterate over splits of each chunk.

        Returns
        -------
        split_chunks : SplitReader
            Iterator of splits for each chunk; the splits themselves are also iterable.
        '''
        return SplitReader(self)

    def get_train(self):
        '''
        Returns
        -------
        chunk_training_indices : List of Lists of Indices
            List of training split indices for each chunk; i.e. chunk_training_indices[0] is a
            list of the split training indices for chunk 0.
        '''
        return self.chunk_train

    def get_test(self):
        '''
        Returns
        -------
        chunk_testing_indices : List of Lists of Indices
            List of testing split indices for each chunk; i.e. chunk_testing_indices[0] is a
            list of the split testing indices for chunk 0.
        '''
        return self.chunk_test

class SplitReader:
    '''
    Read in splits of data chunks.
    '''
    def __init__(self, split_indices):
        '''
        Set up data reader based on previously found indices of splits of chunks.

        Parameters
        ----------
        split_indices : SplitReaderIndices
            The indices to use for the training/test splits of each chunk.
        '''
        self.reader = pd.read_csv(split_indices.filename, index_col = split_indices.index_col,
                                  chunksize = split_indices.chunk_size)
        self.split_indices = split_indices
        self.chunk_i = 0

    def __iter__(self):
        '''
        Returns self.
        '''
        return self

    def __next__(self):
        '''
        Get the splits of the next data chunk.

        Returns
        -------
        chunk_splits : ChunkSplits
            The splits of the training and test sets for the X features and the target classes.
        '''
        target_name = self.split_indices.target_name
        chunk = next(self.reader)

        chunk_train = self.split_indices.get_train()[self.chunk_i]
        chunk_test = self.split_indices.get_test()[self.chunk_i]
        self.chunk_i += 1

        return ChunkSplits(chunk, chunk_train, chunk_test, target_name)

class ChunkSplits:
    '''
    The list of splits for a particular data chunk.
    '''
    def __init__(self, chunk, chunk_train, chunk_test, target_name):
        '''
        Form a list of the training/test split of explanatory X data and target y data for a
        particular data chunk.

        Parameters
        ----------
        chunk : pandas.DataFrame
            The chunk of data.

        chunk_train : List of indices
            List of training indices for the training splits of this chunk.

        chunk_test : List of indices
            List of testing indices for the testing splits of this chunk.

        target_name : String
            The column name of the target y data.
        '''
        self.X_chunk = chunk.drop(target_name, axis = 1)
        self.y_chunk = chunk[target_name]
        self.train_ind = iter(chunk_train)
        self.test_ind = iter(chunk_test)

    def __iter__(self):
        '''
        Returns self.
        '''
        return self

    def __next__(self):
        '''
        Get the next training/test split of this chunk.

        Returns
        -------
        X_train, X_test, y_train, y_test : (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
            Tuple of the split training and testing data.
        '''
        train = next(self.train_ind)
        test = next(self.test_ind)
        X_train = self.X_chunk.loc[train, :]
        y_train = self.y_chunk.loc[train]
        X_test = self.X_chunk.loc[test, :]
        y_test = self.y_chunk.loc[test]

        return X_train, X_test, y_train, y_test

def do_kfold_cv(kfold_X, y, model, score, return_models = False):
    '''
    kfold_X : List of Dictionaries of pandas.DataFrame
        Keys of dictionaries should be 'train' and 'test'.
    y : pandas.Series
        All y values, both train and test.
    model : An sklearn model.
        Model implements fit() and predict_proba().
    '''
    scores = []
    if return_models:
        models = []
    for i, X in enumerate(kfold_X):
        print('fold', i, end = ', ')
        y_train = y.loc[X['train'].index]
        new_model = clone(model)
        new_model.fit(X['train'], y_train)

        y_test = y.loc[X['test'].index]
        y_model = new_model.predict_proba(X['test'])[:, 1]
        scores.append(score(y_test, y_model))
        if return_models:
            models.append(new_model)

    if return_models:
        return np.array(scores), models

    return np.array(scores)
