import json
import my_model
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Remember that custom JSON encoders return objects that are JSON serializable
# by the default encoders in json.py.

class TargetAccumulatorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, my_model.TargetAccumulator):
            if not obj.calculations_updated:
                obj.do_calculations() 
            return MyNumpyEncoder().default(obj)  # Need to encode numpy arrays.
        else: # Let base class raise type error.
           json.JSONEncoder.default(self, obj)

class MyNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        new_dict = {'__class__' : obj.__class__.__name__} 
        for key in obj.__dict__.keys():
            if isinstance(obj.__dict__[key], np.ndarray):
                new_dict['numpy__' + key] = obj.__dict__[key].tolist() 
            elif isinstance(obj.__dict__[key], np.int64):
                new_dict['numpy_int64__' + key] = obj.__dict__[key].item()
            else:
                new_dict[key] = obj.__dict__[key]
        return new_dict

    def is_numpy_array(key):
        id_length = len('numpy__')
        if len(key) < id_length:
            return False
        return key[:id_length] == 'numpy__'

    def is_numpy_int64(key):
        id_length = len('numpy_int64__')
        if len(key) < id_length:
            return False
        return key[:id_length] == 'numpy_int64__'

    def shorten(key, key_type):
        if key_type == 'array':
            return key[len('numpy__'):] 
        elif key_type == 'int64':
            return key[len('numpy_int64__'):]
        else:
            raise Exception('Not able to shorten key')

def as_numpy_items(dct):
    class_conversions = [my_model.TargetMeans, my_model.TargetCovariances, my_model.RatioDirections, 
                         my_model.BestDirectionsSquares, StandardScaler, LogisticRegression]
    class_conversions = {x.__name__ : x for x in class_conversions} 
    if '__class__' not in dct.keys():
        raise Exception('\'__class__\' not in keys ', dct.keys())
    my_class = dct.pop('__class__')
    obj = class_conversions[my_class]()
    for key in dct.keys():
        if MyNumpyEncoder.is_numpy_array(key):
            new_key = MyNumpyEncoder.shorten(key, 'array')
            obj.__dict__[new_key] = np.array(dct[key])
        elif MyNumpyEncoder.is_numpy_int64(key):
            new_key = MyNumpyEncoder.shorten(key, 'int64')
            obj.__dict__[new_key] = np.int64(dct[key])
        else:
            obj.__dict__[key] = dct[key]
    return obj

class BestDirectionsSquaresEncoder(json.JSONEncoder):
   def default(self, obj):
        if isinstance(obj, my_model.BestDirectionsSquares):
            new_dict = {'__class__' : obj.__class__.__name__}
            for key in obj.__dict__.keys():
                if key != 'sub_models':
                   new_dict[key] = obj.__dict__[key] 
            # Need to add __class__ to new_dict['needs_fit'] since it is a dictionary.
            new_dict['needs_fit']['__class__'] = 'needs_fit'

            # Get the correct encoder classes for each sub-model.
            sub_models = { key : MyNumpyEncoder() for key in ['scaler', 'ratio_directions', 'weight_predictor']}
            sub_models.update({ key : TargetAccumulatorEncoder() for key in ['target_means', 'target_cov']})
            # Do the encoding.
            for key in sub_models.keys():
                sub_models[key] = sub_models[key].default(obj.sub_models[key]) 
            sub_models['__class__'] = 'sub_models'
            new_dict['sub_models'] = sub_models
            return new_dict

        else: # Let the base class raise a type error.
            json.JSONEncoder.default(self, obj)

# Rememer that JSON decoding object hook functions are called on every dictionary (i.e. nested) of
# the JSON object.

def as_best_directions_squares(dct):

    if '__class__' not in dct.keys():
        raise Exception('Did not find \'__class__\' in keys', dct.keys())

    if dct['__class__'] == 'sub_models' or dct['__class__'] == 'needs_fit':
        return dct

    else:
        return as_numpy_items(dct)
