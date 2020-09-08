import tensorflow as tf
import numpy as np
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K
import skimage.morphology as skm
import scipy.ndimage.morphology as snm



"""
===============
GLOBAL VARIABLE
===============
"""
MIN_LATT=-1
MAX_LATT=1

class NonPositive(Constraint):
    """
    Constraint to NonPositive Values
    """ 
    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = MAX_LATT

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

class NonPositiveIncreasing(Constraint):
    """
    Constraint to NonPositive and Center equal to zero
    """
    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = MAX_LATT

    def __call__(self, w):
        w = K.clip(w, self.min_value, self.max_value)
        data = np.ones(w.shape)
        data[int(w.shape[0]/2),int(w.shape[1]/2),:,:]=0
        #data_tf = tf.convert_to_tensor(data, np.float32)
        w = tf.multiply(w,tf.convert_to_tensor(data, np.float32))
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}


class Lattice(Constraint):
    """
    Contraint to Value Lattice Value
    """ 
    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = -MIN_LATT

    def __call__(self, w):
        w = K.clip(w, self.min_value, self.max_value)
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

class SEconstraint(Constraint):
    """
    Constraint any SE Shape
    Only for square filters.
    """
    def __init__(self,SE=skm.disk(1)):
        self.min_value = MIN_LATT
        self.max_value = -MIN_LATT
        self.data=SE

    def __call__(self, w):
        data = self.data
        data = np.repeat(data[:, :, np.newaxis], w.shape[2], axis=2)
        data = np.repeat(data[:, :, :,np.newaxis], w.shape[3], axis=3)
        w = w+(tf.convert_to_tensor(data, np.float32)+self.min_value)
        w = K.clip(w, self.min_value, self.max_value)
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value,
                'SE': self.data}



class Disk(Constraint):
    """
    Constraint to Disk Shape
    Only for square filters.
    """
    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = -MIN_LATT

    def __call__(self, w):
        #print('DISK CONSTRAINT',w.shape)
        data = skm.disk(int(w.shape[0]/2))
        data = np.repeat(data[:, :, np.newaxis], w.shape[2], axis=2)
        data = np.repeat(data[:, :, :,np.newaxis], w.shape[3], axis=3)
        w = w+(tf.convert_to_tensor(data, np.float32)+self.min_value)
        w = K.clip(w, self.min_value, self.max_value)
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}


class Diamond(Constraint):
    """
    Constraint to Diamond Shape
    Only for square filters.
    """
    def __init__(self):
        self.min_value = MIN_LATT
        self.max_value = -MIN_LATT

    def __call__(self, w):
        #print('DIAMOND CONSTRAINT',w.shape)
        data = skm.diamond(int(w.shape[0]/2))
        data = np.repeat(data[:, :, np.newaxis], w.shape[2], axis=2)
        data = np.repeat(data[:, :, :,np.newaxis], w.shape[3], axis=3)
        w = w+(tf.convert_to_tensor(data, np.float32)+self.min_value)
        w = K.clip(w, self.min_value, self.max_value)
        return w

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}
