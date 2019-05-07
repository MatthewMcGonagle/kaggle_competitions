'''
Encoders and decoders for JSON representation of model, sub-models, and transformers. This is
useful for model persistence.
'''

import json
import numpy as np

from sklearn.base import BaseEstimator

from my_src import (my_model, class_stats, ratio_directions)

# What is recognized as being a model by the model encoder/decoder.
MODELS = my_model.models + class_stats.MODELS + ratio_directions.models + [BaseEstimator]
MODELS = {x.__name__ : x for x in MODELS}

# Remember that custom JSON encoders return objects that are JSON serializable
# by the default encoders in json.py.

class MyNumpyEncoder(json.JSONEncoder):
    '''
    Encodes the numpy objects necessary for the model.
    '''
    def default(self, obj):
        '''
        Encodes a numpy.ndarray or numpy.int64 as a dictionary with the data and the
        numpy class.
        Parameters
        ----------
        obj : numpy.ndarray or numpy.int64

        Returns
        -------
        encoding : dictionary
            '__class__' : Identifies ndarray or int64.
            'data' : The data of the numpy object.
        '''
        encoding = {'__class__' : obj.__class__.__name__}
        if isinstance(obj, np.ndarray):
            encoding['data'] = obj.tolist()
            return encoding

        if isinstance(obj, np.int64):
            encoding['data'] = obj.item()
            return encoding

        # Let the base class raise a type error.
        return json.JSONEncoder.default(self, obj)

class MyModelEncoder(json.JSONEncoder):
    '''
    Encodes the models, sub-models, and transformers of our model. Will encode objects of type
    listed in my_json.MODELS, a dictionary, or a numpy class used by our model.
    '''
    def default(self, obj):
        '''
        Does the encoding.

        Parameters
        ----------
        obj : dictionary, numpy.ndarray, numpy.int64, or is instance of class in my_json.MODELS.

        Returns
        -------
        encoding : dictionary
            Returns dictionary with objects that can be encoded by the default encoding scheme
            of json.py.
        '''

        if isinstance(obj, (np.ndarray, np.int64)):
            return MyNumpyEncoder().default(obj)

        if MyModelEncoder._is_model(obj) or MyModelEncoder._is_dict(obj):
            if MyModelEncoder._is_dict(obj):
                encoding = {}
                to_encode = obj
            else:
                encoding = {'__class__' : obj.__class__.__name__}
                to_encode = obj.__dict__

            for key, item in to_encode.items():
                if MyModelEncoder._is_model(item) or MyModelEncoder._is_dict(item):
                    item = MyModelEncoder().default(item)
                encoding[key] = item
            drop_duplicate_encoding(obj, encoding)
            return encoding

        # Use the default to raise a type error.
        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def _is_model(obj):
        '''
        Whether an object is an instance of a model type recognized by the encoder.
        Returns boolean.
        '''
        return any(isinstance(obj, model) for _, model in  MODELS.items())

    @staticmethod
    def _is_dict(obj):
        '''
        Whether an object is a dictionary. Returns a boolean.
        '''
        return type(obj) == type({})

def drop_duplicate_encoding(obj, encoding):
    '''
    Drop encoding of cross-references from one sub-model to another. By default, these cross
    references are encoded as exact copies of the encoding of the referenced object. These
    cross-references should be resolved when decoding.

    Parameters
    ----------
    encoding : dictionary of objects encodable by json.py
        The encoding of the sub-model with copies of encodings to be removed.

    Returns
    -------
    duplicate_free_encoding : dictionary of objects encodable my json.py
        The duplicate encodings of cross-references have now been replaced by
        key pairs of type (key : None).
    '''

    if isinstance(obj, my_model.BestDirectionsSquares):
        # These are already encoded as members of BestDirectionsSquares, so can
        # set references to None inside sub-model. Just remember to reset
        # references when decoding.
        if not '_class_means' in encoding['sub_models']['class_cov'].keys():
            raise Exception('_class_means not in sub_models[\'class_cov\'] keys')
        encoding['sub_models']['class_cov']['_class_means'] = None

        if not '_class_cov' in encoding['sub_models']['ratio_directions'].keys():
            raise Exception('_class_cov not in sub_models[\'ratio_directions\'] keys')
        encoding['sub_models']['ratio_directions']['_class_cov'] = None

# Remeber that object hook function is called as the decoder for every nested dictionary in the
# JSON encoding.

def as_full_model(estimator_type):
    '''
    Gives an object hook for the full model based on the type of the final predictor/estimator.

    Parameters
    ----------
    estimator_type : class
        Class of the final estimator/predictor, e.g. sklearn.linear_model.LogisticRegression.

    Returns
    -------
    object_hook_full_model : function
        The object hook function to use for decoding the JSON of the full model.
    '''
    def new_obj_hook(dct):
        return as_model(dct, estimator_type)
    return new_obj_hook

def as_model(dct, estimator_type):
    '''
    Given an estimator type, this is method gives the object hook for decoding the JSON
    representation of the model.

    Parameters
    ----------
    dct : dictionary
        Dictionary of objects where JSON sub-encodings have already been decoded.
    estimator_type : class of estimator/predictor
        The type of the final estimator/predictor of the model. This type will be used for
        the decoding of the final estimator/predictor.
    '''
    if '__class__' not in dct.keys():
        return dct

    if dct['__class__'] in MODELS.keys() or dct['__class__'] == estimator_type.__name__:
        if dct['__class__'] == estimator_type.__name__:
            model = estimator_type()
        else:
            model = MODELS[dct['__class__']]()
        dct.pop('__class__')

        # For sklearners, some attributes of a fitted model are not found in an initial instance
        # (i.e. before fitting). So we need to iterate over dct.keys() after popping the class.
        for key in dct.keys():
            model.__dict__[key] = dct[key]
        if isinstance(model, my_model.BestDirectionsSquares):
            make_sub_model_cross_references(model)
        return model

    if dct['__class__'] == np.ndarray.__name__:
        return np.array(dct['data'])

    if dct['__class__'] == np.int64.__name__:
        return np.int64(dct['data'])

    raise Exception('Don\'t know how to decode ', dct['__class__'])

def make_sub_model_cross_references(model):
    '''
    Make model cross-references. These were set to None during the encoding process as
    the references can't be encoded in JSON.

    Parameters
    ----------
    model : my_model.BestDirectionsSquares
        The decoded model.
    '''
    model.sub_models['class_cov'].set_class_means(model.sub_models['class_means'])
    model.sub_models['ratio_directions'].set_class_covariances(model.sub_models['class_cov'])
