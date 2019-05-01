import json
import numpy as np

from sklearn.base import BaseEstimator

from my_src import (my_model, class_stats, ratio_directions)

models = my_model.models + class_stats.models + ratio_directions.models + [BaseEstimator]
models = {x.__name__ : x for x in models} 

# Remember that custom JSON encoders return objects that are JSON serializable
# by the default encoders in json.py.

class MyNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        encoding = {'__class__' : obj.__class__.__name__}
        if isinstance(obj, np.ndarray):
            encoding['data'] = obj.tolist()
            return encoding 
        elif isinstance(obj, np.int64):
            encoding['data'] = obj.item()
        else: # Let the base class raise a type error.
            json.JSONEncoder.default(self, obj)
           
class MyModelEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.int64)):
           return MyNumpyEncoder().default(obj)

        elif MyModelEncoder._is_model(obj) or MyModelEncoder._is_dict(obj):
            encoding = {'__class__' : obj.__class__.__name__}
            if MyModelEncoder._is_dict(obj):
                to_encode = obj
            else:
                to_encode = obj.__dict__

            for key, item in to_encode.items():
                 if MyModelEncoder._is_model(item) or MyModelEncoder._is_dict(item):
                     item = MyModelEncoder().default(item)
                 encoding[key] = item
            drop_duplicate_encoding(obj, encoding)
            return encoding

        else: # Use the default to raise a type error.
           json.JSONEncoder.default(self, obj)
           
    def _is_model(obj):
        return any(isinstance(obj, model) for _, model in  models.items()) 

    def _is_dict(obj):
        return isinstance(obj, {}.__class__)#obj.__class__ == {}.__class__

def drop_duplicate_encoding(obj, encoding):
    
    if isinstance(obj, my_model.BestDirectionsSquares):
        # These are already encoded as members of BestDirectionsSquares, so can
        # set references to None inside sub-model. Just remember to reset
        # references when decoding.
        if not '_class_means' in encoding['sub_models']['class_cov'].keys():
            raise Exception('_class_means not in sub_models[\'class_cov\'] keys')
        else:
            encoding['sub_models']['class_cov']['_class_means'] = None

        if not '_class_cov' in encoding['sub_models']['ratio_directions'].keys():
            raise Exception('_class_cov not in sub_models[\'ratio_directions\'] keys')
        else:
            encoding['sub_models']['ratio_directions']['_class_cov'] = None

# Remeber that object hook function is called as the decoder for every nested dictionary in the
# JSON encoding.

def as_full_model(estimator_type):
    def new_obj_hook(dct):
        return as_model(dct, estimator_type)
    return new_obj_hook
 
def as_model(dct, estimator_type = None.__class__):

    if '__class__' not in dct.keys() or dct['__class__'] == {}.__class__.__name__:
        return dct

    elif dct['__class__'] in models.keys() or dct['__class__'] == estimator_type.__name__:
        if dct['__class__'] == estimator_type.__name__:
            model = estimator_type()
        else: 
            model = models[dct['__class__']]() 
        dct.pop('__class__')
        # For sklearners, some attributes of a fitted model are not found in an initial instance
        # (i.e. before fitting). So we need to iterate over dct.keys() after popping the class.
        for key in dct.keys():
            model.__dict__[key] = dct[key]
        if isinstance(model, my_model.BestDirectionsSquares):
            make_sub_model_cross_references(model)
        return model 

    elif dct['__class__'] == np.ndarray.__name__:
        return np.array(dct['data'])

    elif dct['__class__'] == np.int64.__name__:
        return np.int64(dct['data'])

    else:
        print(dct['__class__'], estimator_type.__name__)
        raise Exception('Don\'t know how to decode ', dct['__class__'])

def make_sub_model_cross_references(model):

    model.sub_models['class_cov'].set_class_means(model.sub_models['class_means'])
    model.sub_models['ratio_directions'].set_class_covariances(model.sub_models['class_cov'])
