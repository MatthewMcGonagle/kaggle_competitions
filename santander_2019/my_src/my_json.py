import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from my_src import (my_model, class_stats, ratio_directions)

models = my_model.models + class_stats.models + ratio_directions.models
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
            if not MyModelEncoder._is_dict(obj):
                to_encode = obj.__dict__
            else:
                to_encode = obj

            for key in to_encode.keys():
                 item = to_encode[key]
                 if MyModelEncoder._is_model(item) or MyModelEncoder._is_dict(item):
                     item = MyModelEncoder().default(item)
                 encoding[key] = item
            return encoding
        else: # Use the default to raise a type error.
           json.JSONEncoder.default(self, obj)
           
    def _is_model(obj):
        return obj.__class__.__name__ in models.keys()

    def _is_dict(obj):
        return obj.__class__ == {}.__class__

# Remeber that object hook function is called as the decoder for every nested dictionary in the
# JSON encoding.

def as_model(dct):

    if '__class__' not in dct.keys():
        raise Exception('__class__ not in dictionary keys:', dct.keys())

    elif dct['__class__'] == {}.__class__.__name__:
        return dct

    elif dct['__class__'] in models.keys():
        model = models[dct['__class__']]() 
        dct.pop('__class__')
        # For sklearners, some attributes of a fitted model are not found in an initial instance
        # (i.e. before fitting). So we need to iterate over dct.keys() after popping the class.
        for key in dct.keys():
            model.__dict__[key] = dct[key]
        return model 

    elif dct['__class__'] == np.ndarray.__name__:
        return np.array(dct['data'])

    elif dct['__class__'] == np.int64.__name__:
        return np.int64(dct['data'])

    else:
        raise Exception('Don\'t know how to decode ', dct['__class__'])

