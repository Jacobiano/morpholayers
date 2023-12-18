"""
==========
References
==========
Implementation of Morphological Layers  [1]_, [2]_, [3]_, [4]_

.. [1] Serra, J. (1983) Image Analysis and Mathematical Morphology. 
       Academic Press, Inc. Orlando, FL, USA
.. [2] Soille, P. (1999). Morphological Image Analysis. Springer-Verlag
"""



from keras.layers import Layer

##Preliminary function

def condition_equal(last,new,image):
    return keras.ops.logical_not(keras.ops.all(keras.ops.equal(last, new)))

def update_dilation(last,new,mask):
     return [new, geodesic_dilation_step([new, mask]), mask]

def update_leveling(last,new,mask):
    return new,leveling_iteration([new, mask]),mask

def geodesic_dilation_step(X):
    """
    1 step of reconstruction by dilation
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(geodesic_dilation_step, name="reconstruction")([Mask,Image])
    """
    # perform a geodesic dilation with X[0] as marker, and X[1] as mask
    return keras.ops.minimum(keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(X[0]),X[1])

def leveling_iteration(X):
    """
    :Leveling iteration
    :X tensor: X[0] is the Mask and X[1] is the Image
    """
    return keras.ops.maximum(-keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(-X[0]),
                             keras.ops.minimum(keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(X[0]),X[1]))



##Dilation Layers: This implementation compute image to neighborhood which is not involving tf.dilation2d as last version of morpholayers. This allow us to use different backends: jax,pytorch or tensoflow.

##SE in 3D

@keras.saving.register_keras_serializable()
class DilationLayer(Layer):
    def __init__(self, filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_length = kernel_size[0]*kernel_size[1]
        self.strides = strides
        self.padding = padding

    # Create the state of the layer (weights)
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(1, 1, 1, self.kernel_length*input_shape[-1], self.filters),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
            )
    # Defines the computation
    def call(self, inputs):
      patches = keras.ops.image.extract_patches(inputs, self.kernel_size,padding=self.padding,strides=self.strides)
      return keras.ops.max(keras.ops.expand_dims(patches,axis=-1) + self.kernel,axis=-2)

    def get_config(self):
        return {"filters": self.filters}
        
        
@keras.saving.register_keras_serializable()
class ErosionLayer(Layer):
    def __init__(self, filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_length = kernel_size[0]*kernel_size[1]
        self.strides = strides
        self.padding = padding

    # Create the state of the layer (weights)
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(1, 1, 1, self.kernel_length*input_shape[-1], self.filters),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
            )
    # Defines the computation
    def call(self, inputs):
      patches = keras.ops.image.extract_patches(inputs, self.kernel_size,padding=self.padding,strides=self.strides)
      return keras.ops.min(keras.ops.expand_dims(patches,axis=-1) - self.kernel,axis=-2)

    def get_config(self):
        return {"filters": self.filters}
        
        
        
## MARGINAL OPERATOR


@keras.saving.register_keras_serializable()
class MarginalDilationLayer(Layer):
    def __init__(self, filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',kernel_initializer='Zeros'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_length = kernel_size[0]*kernel_size[1]
        self.strides = strides
        self.padding = padding
        if kernel_initializer=='Zeros':
            self.initializer=keras.initializers.Zeros()
        elif kernel_initializer=='Uniform':
            self.initializer=keras.initializers.RandomUniform(-1,0)
        else:
            self.initializer=kernel_initializer

    # Create the state of the layer (weights)
    def build(self, input_shape):
        self.channels=input_shape[-1]
        self.kernel = self.add_weight(
            shape=(1, 1, 1, self.kernel_length,input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name="kernel",
            )
    # Defines the computation
    def call(self, inputs):
        patches = keras.ops.image.extract_patches(inputs, self.kernel_size,padding=self.padding,strides=self.strides)
        b, h, w, c = keras.ops.shape(patches)
        patches = keras.ops.reshape(patches, (b, h, w, self.kernel_length,self.channels))
        return keras.ops.max(keras.ops.expand_dims(patches,axis=-1) + self.kernel,axis=3)
  
    def get_config(self):
        return {"filters": self.filters}


@keras.saving.register_keras_serializable()
class MarginalErosionLayer(Layer):
    def __init__(self, filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',kernel_initializer='Zeros'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_length = kernel_size[0]*kernel_size[1]
        self.strides = strides
        self.padding = padding
        if kernel_initializer=='Zeros':
            self.initializer=keras.initializers.Zeros()
        elif kernel_initializer=='Uniform':
            self.initializer=keras.initializers.RandomUniform(-1,0)
        else:
            self.initializer=kernel_initializer

    # Create the state of the layer (weights)
    def build(self, input_shape):
        self.channels=input_shape[-1]
        self.kernel = self.add_weight(
            shape=(1, 1, 1, self.kernel_length,input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name="kernel",
            )
    # Defines the computation
    def call(self, inputs):
        patches = keras.ops.image.extract_patches(inputs, self.kernel_size,strides=self.strides)
        b, h, w, c = keras.ops.shape(patches)
        patches = keras.ops.reshape(patches, (b, h, w, self.kernel_length,self.channels))
        return keras.ops.min(keras.ops.expand_dims(patches,axis=-1) - keras.ops.flip(self.kernel,axis=3),axis=3)

    def get_config(self):
        return {"filters": self.filters}

@keras.saving.register_keras_serializable()
class MarginalMEMLayer(Layer):
    "Marginal Morphological Empirical Mode
    "

    def __init__(self, filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',kernel_initializer='Zeros'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_length = kernel_size[0]*kernel_size[1]
        self.strides = strides
        if kernel_initializer=='Zeros':
            self.initializer=keras.initializers.Zeros()
        elif kernel_initializer=='Uniform':
            self.initializer=keras.initializers.RandomUniform(-1,0)
        else:
            self.initializer=kernel_initializer

    # Create the state of the layer (weights)
    def build(self, input_shape):
        self.channels=input_shape[-1]
        self.kernel = self.add_weight(
            shape=(1, 1, 1, self.kernel_length,input_shape[-1], self.filters),
            initializer=self.initializer,
            trainable=True,
            name="kernel",
            )
    # Defines the computation
    def call(self, inputs):
        patches = keras.ops.image.extract_patches(inputs, self.kernel_size,strides=self.strides)
        b, h, w, c = keras.ops.shape(patches)
        patches = keras.ops.reshape(patches, (b, h, w, self.kernel_length,self.channels))
        return keras.ops.max(keras.ops.expand_dims(patches,axis=-1) + self.kernel,axis=3) + keras.ops.min(keras.ops.expand_dims(patches,axis=-1) - keras.ops.flip(self.kernel,axis=3),axis=3)

    def get_config(self):
        return {"filters": self.filters}


#Reconstruction


@keras.saving.register_keras_serializable()
class GeodesicalReconstructionLayer(Layer):
    def __init__(self,steps=None):
        super().__init__()
        self.steps = steps

    def call(self, inputs):
        rec = inputs[0]
        rec = geodesic_dilation_step([rec, inputs[1]])
        _, rec,_=keras.ops.while_loop(condition_equal, update_dilation, [inputs[0], rec, inputs[1]], maximum_iterations=self.steps)
        return rec
        
@keras.saving.register_keras_serializable()
class LevelingLayer(Layer):
    def __init__(self,steps=None):
        super().__init__()
        self.steps = steps

    def call(self, inputs):
        lev = leveling_iteration([inputs[0],inputs[1]])
        _, rec,_=keras.ops.while_loop(condition_equal, update_leveling, [inputs[0], lev, inputs[1]], maximum_iterations=self.steps)
        return rec
