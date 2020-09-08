"""
==========
References
==========
Implementation of Morphological Layers  [1]_, [2]_, [3]_, [4]_

.. [1] Serra, J. (1983) Image Analysis and Mathematical Morphology. 
       Academic Press, Inc. Orlando, FL, USA
.. [2] Soille, P. (1999). Morphological Image Analysis. Springer-Verlag
.. [3] Najman, L., & Couprie, M. (2006). Building the component tree in
       quasi-linear time. IEEE Transactions on Image Processing, 15(11),
       3531-3539.
       :DOI:`10.1109/TIP.2006.877518`
.. [4] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
       Component Tree Computation Algorithms. IEEE Transactions on Image
       Processing, 23(9), 3885-3895.
       :DOI:`10.1109/TIP.2014.2336551`

"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from morpholayers.constraints import SEconstraint
import skimage.morphology as skm
import scipy.ndimage.morphology as snm

"""
===============
GLOBAL VARIABLE
===============
"""

NUM_ITER_REC=21 #Default value for number of iterations in  reconstruction operator.

"""
===================
Classical Operators
===================
"""

def dilation2d(x, st_element, strides, padding,rates=(1, 1)):
    """
    Basic Dilation Operator
    :param st_element: Nonflat structuring element
    :strides: strides as classical convolutional layers
    :padding: padding as classical convolutional layers
    :rates: rates as classical convolutional layers
    """
    x = tf.nn.dilation2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
    return x


def erosion2d(x, st_element, strides, padding,rates=(1, 1)):
    """
    Basic Erosion Operator
    :param st_element: Nonflat structuring element
    c """
    x = tf.nn.erosion2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
    return x


def opening2d(x, st_element, strides, padding,rates=(1, 1)):
    """
    Basic Opening Operator
    :param st_element: Nonflat structuring element
    :strides: strides are only applied in second operator (dilation)
    :padding: padding as classical convolutional layers
    :rates: rates are only applied in second operator (dilation)
    """
    x = tf.nn.erosion2d(x, st_element, (1, ) + (1,1) + (1, ),padding.upper(),"NHWC",(1,) + (1,1) + (1,))
    x = tf.nn.dilation2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,) + rates + (1,))
    return x

def closing2d(x, st_element, strides, padding,rates=(1, 1)):
    """
    Basic Closing Operator
    :param st_element: Nonflat structuring element
    :strides: strides are only applied in second operator (erosion)
    :padding: padding as classical convolutional layers
    :rates: rates are only applied in second operator (erosion)
    """
    x = tf.nn.dilation2d(x, st_element, (1, ) + (1,1) + (1, ),padding.upper(),"NHWC",(1,) + (1,1) + (1,))
    x = tf.nn.erosion2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,) + rates + (1,))
    return x

def gradient2d(x, st_element, strides, padding,rates=(1, 1)):
    """
    Gradient Operator
    :param st_element: Nonflat structuring element
    :strides: strides are only applied in second operator (erosion)
    :padding: padding as classical convolutional layers
    :rates: rates are only applied in second operator (erosion)
    """
    x = tf.nn.dilation2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))-tf.nn.erosion2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
    return x

def internalgradient2d(x, st_element, strides, padding,rates=(1, 1)):
    """
    Internal Gradient Operator
    :param st_element: Nonflat structuring element
    :strides: strides are only applied in second operator (erosion)
    :padding: padding as classical convolutional layers
    :rates: rates are only applied in second operator (erosion)
    """
    x = x-tf.nn.erosion2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
    return x

def togglemapping2d(x, st_element, strides=(1,1), padding='same',rates=(1, 1, 1, 1),steps=5):
    """
    Toggle Mapping Operator
    :param st_element: Nonflat structuring element
    :strides: strides are only applied in second operator (erosion)
    :padding: padding as classical convolutional layers
    :rates: rates are only applied in second operator (erosion)
    """
    for _ in range(steps):
        d=tf.nn.dilation2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
        e=tf.nn.erosion2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
        Delta=tf.keras.layers.Minimum()([tf.abs(d-x),tf.abs(x-e)])
        Mask=tf.cast(tf.less_equal(d-x,x-e),'float32')
        x=x+(Mask*Delta)
    return x



def togglemapping(X,steps=5):
    """
    K steps of toggle mapping operator
    :X is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(reconstruction_dilation, name="reconstruction")([Mask,Image])
    """
    for _ in range(steps):
        d=tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(X)
        e=MinPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(X)
        Delta=tf.keras.layers.Minimum()([d-X,X-e])
        Mask=tf.cast(tf.less_equal(d-X,X-e),'float32')
        X=X+(Mask*Delta)
    return X

def antidilation2d(x, st_element, strides, padding,rates=(1, 1)):
    """
    Basic Dilation Operator of the negative of the input image 
    :param st_element: Nonflat structuring element
    :strides: strides as classical convolutional layers
    :padding: padding as classical convolutional layers
    :rates: rates as classical convolutional layers
    """
    x = tf.nn.dilation2d(-x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
    return x

def antierosion2d(x, st_element, strides, padding,rates=(1, 1)):
    """
    Basic Erosion Operator of the negative of the input image
    :param st_element: Nonflat structuring element
    c """
    x = tf.nn.erosion2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
    return x

"""
==========================
Operator by Reconstruction
==========================
"""

def reconstruction_dilation(X,steps=NUM_ITER_REC):
    """
    K steps of reconstruction by dilation
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(reconstruction_dilation, name="reconstruction")([Mask,Image])
    """
    for i in range(steps):
        X[0]=tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(X[0])
        #print(X[0].shape)
        #print(X[1].shape)
        X[0]=tf.keras.layers.Minimum()([X[0],X[1]])
    return X[0]

def reconstruction_erosion(X,steps=NUM_ITER_REC):
    """
    K steps of reconstruction by dilation
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(reconstruction_dilation, name="reconstruction")([Mask,Image])
    """
    #Use in Keras: Lambda(reconstruction_erosion, name="reconstruction")([Mask,Image])
    for i in range(steps):
        X[0]=MinPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(X[0])
        #print(X[0].shape)
        #print(X[1].shape)
        X[0]=tf.keras.layers.Maximum()([X[0],X[1]])
    return X[0]

"""
==============
POOLING LAYERS
==============
"""

class MinPooling2D(Layer):
  """
  Min Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
  Arguments:
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self,  pool_size, strides,
               padding='valid', data_format=None,
               name=None, **kwargs):
    super(MinPooling2D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs):
    if self.data_format == 'channels_last':
      pool_shape = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)
    else:
      pool_shape = (1, 1) + self.pool_size
      strides = (1, 1) + self.strides
    outputs = -nn.max_pool(
        -inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format, 4))
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]
    rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                         self.strides[0])
    cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                         self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(MinPooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GradPooling2D(Layer):
  """
  Grad Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).
  Arguments:
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self,  pool_size, strides,
               padding='valid', data_format=None,
               name=None, **kwargs):
    super(GradPooling2D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.input_spec = InputSpec(ndim=4)
    
  def call(self, inputs):
    if self.data_format == 'channels_last':
      pool_shape = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)
    else:
      pool_shape = (1, 1) + self.pool_size
      strides = (1, 1) + self.strides
    outputs = nn.max_pool(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format, 4))+nn.max_pool(
        -inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format, 4))
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]
    rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                         self.strides[0])
    cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                         self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format
    }
    base_config = super(GradPooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))



"""
====================
Max/Min of Operators
====================
"""

class MaxofErosions2D(Layer):
    """
    Maximum of Erosion 2D Layer
    for now assuming channel last
    [1]_, [2]_, [3]_, [4]_

    :param num_filters: the number of filters
    :param kernel_size: kernel size used

    :Example:

    >>>from keras.models import Sequential,Model
    >>>from keras.layers import Input
    >>>xin=Input(shape=(28,28,3))
    >>>x=MaxofErosion2D(num_filters=7,kernel_size=(5,5)))(xin)
    >>>model = Model(xin,x)

    """
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), kernel_initializer='glorot_uniform',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(MaxofErosions2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(MaxofErosions2D, self).build(input_shape)

    def call(self, x):
        outputs = K.placeholder()
        for i in range(self.num_filters):
            # erosion2d returns image of same size as x
            # so taking min over channel_axis
            out = K.max(
                erosion2d(x, self.kernel[..., i],self.strides, self.padding),
                axis=self.channel_axis, keepdims=True)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])  # self.erosion_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config


class MinofDilations2D(Layer):
    """
    Minimum of Dilations 2D Layer assuming channel last
    
    :param num_filters: the number of filters
    :param kernel_size: kernel size used
    
    """
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), kernel_initializer='glorot_uniform',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(MinofDilations2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)

        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(MinofDilations2D, self).build(input_shape)

    def call(self, x):
        #outputs = K.placeholder()
        for i in range(self.num_filters):
            # dilation2d returns image of same size as x
            # so taking max over channel_axis
            out = K.min(dilation2d(x, self.kernel[..., i],self.strides, self.padding),axis=self.channel_axis, keepdims=True)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

"""
===================
Classical Layers
===================
"""

class Erosion2D(Layer):
    '''
    Erosion 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(Erosion2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(Erosion2D, self).build(input_shape)

    def call(self, x):
        outputs = K.placeholder()
        for i in range(self.num_filters):
            out = erosion2d(x, self.kernel[..., i],self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config


class DilationSE2D(Layer):
    '''
    Dilation SE 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, structuring_element=skm.disk(1),strides=(1, 1),
                 padding='same', dilation_rate=(1,1),kernel_initializer='Zeros',kernel_constraint=SEconstraint(SE=structuring_element),kernel_regularization=None,
                 **kwargs):
        super(DilationSE2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = structuring_element.shape
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(DilationSE2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = dilation2d(x, self.kernel[..., i],self.strides, self.padding,self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i]) 
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config




class Dilation2D(Layer):
    '''
    Dilation 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1),kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(Dilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(Dilation2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = dilation2d(x, self.kernel[..., i],self.strides, self.padding,self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i]) 
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config


class Antierosion2D(Layer):
    '''
    AntiErosion 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1), kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(Antierosion2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(Antierosion2D, self).build(input_shape)

    def call(self, x):
        outputs = K.placeholder()
        for i in range(self.num_filters):
            out = antierosion2d(x, self.kernel[..., i],self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config



class Antidilation2D(Layer):
    '''
    Antidilation 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1),kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(Antidilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(Antidilation2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = antidilation2d(x, self.kernel[..., i],self.strides, self.padding,self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i]) 
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

"""
=================================
Quadratic Morphological Operators
=================================
"""
class QuadraticDilation2D(Layer):
    '''
    Quadratic Dilation 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1),bias_initializer='Ones',bias_constraint=None,bias_regularization=None,scale=1,**kwargs):
        super(QuadraticDilation2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.scale=scale
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.bias_regularization = tf.keras.regularizers.get(bias_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]

        data=np.ones(self.kernel_size)
        data[int(data.shape[0]/2),int(data.shape[1]/2)]=0
        #data=snm.distance_transform_edt(data)**2
        data=snm.distance_transform_edt(data)
        data=(-(data/(4*self.scale))**2)
        data = np.repeat(data[:, :, np.newaxis], input_dim, axis=2)
        data = np.repeat(data[:, :, :,np.newaxis], self.num_filters, axis=3)
        self.data=tf.convert_to_tensor(data, np.float32)
        self.bias=self.add_weight(shape=(input_dim,self.num_filters),initializer=self.bias_initializer,
        name='bias',constraint =self.bias_constraint,regularizer=self.bias_regularization)
        super(QuadraticDilation2D, self).build(input_shape)

    def call(self, x):
        kernel=tf.math.multiply(self.data,self.bias)
        for i in range(self.num_filters):
            out = dilation2d(x, kernel[..., i],self.strides, self.padding,self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i]) 
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

"""
=================
Top Hat Operators
=================
"""
class TopHatOpening2D(Layer):
    '''
    TopHat from Opening 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1),kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(TopHatOpening2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(TopHatOpening2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = self.tophatopening2d(x, self.kernel[..., i],self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def tophatopening2d(self, x, st_element, strides, padding,
                     rates=(1, 1)):
                     
        z = tf.nn.erosion2d(x, st_element, (1, ) + (1,1) + (1, ),padding.upper(),"NHWC",(1,) + rates + (1,))
        z = tf.nn.dilation2d(z, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,) + rates + (1,))
        return x-z

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

class TopHatClosing2D(Layer):
    '''
    TopHat from Closing 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1),kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(TopHatClosing2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1
        #TODO strides not working.

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(TopHatClosing2D, self).build(input_shape)

    def call(self, x):
        #outputs = K.placeholder()
        for i in range(self.num_filters):
            out = self.tophatclosing2d(x, self.kernel[..., i],self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])

        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def tophatclosing2d(self, x, st_element, strides, padding,
                     rates=(1, 1)):
        z = tf.nn.dilation2d(x, st_element, (1, ) + (1,1) + (1, ),padding.upper(),"NHWC",(1,) + rates + (1,))
        z = tf.nn.erosion2d(z, st_element, (1, ) + (1,1) + (1, ),padding.upper(),"NHWC",(1,) + rates + (1,))
        return z-x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config
"""
===================
Opening and Closing
===================
"""
class Opening2D(Layer):
    '''
    Opening 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1),kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(Opening2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(Opening2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = opening2d(x, self.kernel[..., i],self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

class Closing2D(Layer):
    '''
    Closing 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same',dilation_rate=(1,1), kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(Closing2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(Closing2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = closing2d(x, self.kernel[..., i],self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=1) 
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

"""
==========================================
Morphological Empirical Mode Decomposition
==========================================
"""
class MorphoEMD2D(Layer):
    '''
    Morphological Empirical Mode Decomposition 2D Layer
    for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same',dilation_rate=(1,1), kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(MorphoEMD2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        # Be sure to call this at the end
        super(MorphoEMD2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = self.emd2d(x, self.kernel[..., i],self.strides, self.padding)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=1) 
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def emd2d(self, x, st_element, strides, padding,
                     rates=(1, 1, 1, 1)):
        x1 = tf.nn.dilation2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",rates)
        x2 = tf.nn.erosion2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",rates)
        return x-(x1+x2)/2

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

"""
=======
Probing
=======
"""

class Probing2D(Layer):
    '''
    Morphological Probing 2D Layer for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1),kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(Probing2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)+ (2,)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        super(Probing2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = self.probing2d(x, self.kernel[..., i,:],self.strides, self.padding,self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def probing2d(self, x, st_element, strides, padding,
                     rates=(1, 1)):
        x = tf.nn.dilation2d(x, st_element[...,0], (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))-tf.nn.erosion2d(x, st_element[...,1], (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config
"""
========
Gradient
========
"""

class Gradient2D(Layer):
    '''
    Morphological Gradient 2D Layer for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1),kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(Gradient2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        super(Gradient2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = gradient2d(x, self.kernel[..., i],self.strides, self.padding,self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i]) 
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config

class InternalGradient2D(Layer):
    '''
    Internal Morphological Gradient 2D Layer for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, strides=(1, 1),
                 padding='same', dilation_rate=(1,1),kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(InternalGradient2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        super(InternalGradient2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = internalgradient2d(x, self.kernel[..., i],self.strides, self.padding,self.rates)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i]) 
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config


class ToggleMapping2D(Layer):
    '''
    ToggleMapping 2D Layer for now assuming channel last
    '''
    def __init__(self, num_filters, kernel_size, steps=5,kernel_initializer='Zeros',kernel_constraint=None,kernel_regularization=None,
                 **kwargs):
        super(ToggleMapping2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = (1,1)
        self.padding = 'same'
        self.rates=(1,1)
        self.steps=steps

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        self.channel_axis = -1

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.num_filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)

        super(ToggleMapping2D, self).build(input_shape)

    def call(self, x):
        for i in range(self.num_filters):
            out = togglemapping2d(x, self.kernel[..., i],self.strides, self.padding,self.rates,self.steps)
            if i == 0:
                outputs = out
            else:
                outputs = K.concatenate([outputs, out])
        return outputs

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i]) 
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_filters*input_shape[self.channel_axis],)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config



