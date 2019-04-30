import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

class StratifiedKFoldReaderIndices:

    def __init__(self, filename, index_col, chunk_size, target_name = 'target', 
                 kfold = 5, random_state = np.random.RandomState(), verbose = True):
        self.filename = filename
        self.index_col = index_col
        self.chunk_size = chunk_size
        self.target_name = target_name
        self.kfold = kfold

        reader = pd.read_csv(filename, index_col = index_col, chunksize = chunk_size) 
        stratified_kfold = StratifiedKFold(kfold, random_state = random_state)
        chunk_train = []
        chunk_test = []

        if verbose:
            print('Splitting chunks: ', end = '')

        for chunk_i, chunk in enumerate(reader):
            if verbose:
                print(chunk_i, end = ', ')

            x = chunk.index
            y = chunk[target_name] 
            new_train_kfold = []
            new_test_kfold = []
            for train_i, test_i in stratified_kfold.split(x, y):
                new_train_kfold.append(x[train_i])
                new_test_kfold.append(x[test_i])
            chunk_train.append(new_train_kfold)
            chunk_test.append(new_test_kfold)

        self.chunk_train = chunk_train
        self.chunk_test = chunk_test

    def __iter__(self):
        return StratifiedKFoldReader(self)
            
    def get_train(self):
        return self.chunk_train

    def get_test(self):
        return self.chunk_test  

    def concatenate_chunks(self):
        train_totals = [[] for _ in range(self.kfold)]
        test_totals = [[] for _ in range(self.kfold)]
        for folds in chunk_train:
            for new_indices, total in zip(folds, train_totals):
                total.extend(new_indices) 

        for folds in chunk_test:
            for new_indices, total in zip(folds, test_totals):
                total.extend(new_indices)

        return train_totals, test_totals
             
class StratifiedKFoldReader:

    def __init__(self, split_indices): 
        self.reader = pd.read_csv(split_indices.filename, index_col = split_indices.index_col, 
                                  chunksize = split_indices.chunk_size) 
        self.split_indices = split_indices
        self.chunk_i = 0 

    def __iter__(self):
        return self

    def __next__(self):

        target_name = self.split_indices.target_name
        chunk = next(self.reader)

        chunk_train = self.split_indices.get_train()[self.chunk_i]
        chunk_test = self.split_indices.get_test()[self.chunk_i]
        self.chunk_i += 1

        return StratifiedKFoldChunk(chunk, chunk_train, chunk_test, target_name) 
         
class StratifiedKFoldChunk:

    def __init__(self, chunk, chunk_train, chunk_test, target_name): 

        self.X_chunk = chunk.drop(target_name, axis = 1)
        self.y_chunk = chunk[target_name]
        self.train_ind = iter(chunk_train)
        self.test_ind = iter(chunk_test)

    def __iter__(self):
        return self

    def __next__(self):

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
    else:
        return np.array(scores)

