#!/usr/bin/env python
# coding: utf-8

"""
Created on Sat Mar 21 11:35:36 2020

@author: abramo
"""

import numpy as np
import os
import random
import skimage.io as io
import skimage.transform as trans
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K
import tensorflow as tf
from keras.activations import softmax
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Concatenate, Lambda, ZeroPadding2D, Activation, Reshape, Add
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback 
from keras.utils import plot_model, Sequence





def unet():
    inputs=Input(shape=(432,432,2))
    weight_matrix=Lambda(lambda z: z[:,:,:,1])(inputs)
    weight_matrix=Reshape((432,432,1))(weight_matrix)
    reshape=Lambda(lambda z : z[:,:,:,0])(inputs)
    reshape=Reshape((432,432,1))(reshape)

    reg=0.01
    
    #reshape=Dropout(0.2)(reshape)   ## Hyperparameter optimization only on visible layer
    Level1_l=Conv2D(filters=32,kernel_size=(1,1),strides=1,kernel_regularizer=regularizers.l2(reg))(reshape)
    Level1_l=BatchNormalization(axis=-1)(Level1_l)
    Level1_l_shortcut=Level1_l#Level1_l#
    Level1_l=Activation('relu')(Level1_l)
    Level1_l=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_l)#(Level1_l)# ##  kernel_initializer='glorot_uniform' is the default
    Level1_l=BatchNormalization(axis=-1)(Level1_l)
    #Level1_l=InstanceNormalization(axis=-1)(Level1_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_l=Activation('relu')(Level1_l)
    #Level1_l=Dropout(0.5)(Level1_l)   
    Level1_l=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_l)
    Level1_l=BatchNormalization(axis=-1)(Level1_l)
    #Level1_l=InstanceNormalization(axis=-1)(Level1_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_l=Add()([Level1_l,Level1_l_shortcut])
    Level1_l=Activation('relu')(Level1_l)


    Level2_l=Conv2D(filters=64,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level1_l)
    Level2_l=BatchNormalization(axis=-1)(Level2_l)
    Level2_l_shortcut=Level2_l
    Level2_l=Activation('relu')(Level2_l)
    #Level2_l=BatchNormalization(axis=-1)(Level2_l)
    #Level2_l=ZeroPadding2D(padding=(1,1))(Level2_l)
    Level2_l=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_l)
    Level2_l=BatchNormalization(axis=-1)(Level2_l)
    #Level2_l=InstanceNormalization(axis=-1)(Level2_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_l=Activation('relu')(Level2_l)
    #Level2_l=Dropout(0.5)(Level2_l)
    Level2_l=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_l)
    Level2_l=BatchNormalization(axis=-1)(Level2_l)
    #Level2_l=InstanceNormalization(axis=-1)(Level2_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_l=Add()([Level2_l,Level2_l_shortcut])
    Level2_l=Activation('relu')(Level2_l)
    
    
    Level3_l=Conv2D(filters=128,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level2_l)
    Level3_l=BatchNormalization(axis=-1)(Level3_l)
    Level3_l_shortcut=Level3_l
    Level3_l=Activation('relu')(Level3_l)
    #Level3_l=ZeroPadding2D(padding=(1,1))(Level3_l)
    Level3_l=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_l)
    Level3_l=BatchNormalization(axis=-1)(Level3_l)
    #Level3_l=InstanceNormalization(axis=-1)(Level3_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_l=Activation('relu')(Level3_l)
    #Level3_l=Dropout(0.5)(Level3_l)
    Level3_l=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_l)
    Level3_l=BatchNormalization(axis=-1)(Level3_l)
    #Level3_l=InstanceNormalization(axis=-1)(Level3_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_l=Add()([Level3_l,Level3_l_shortcut])
    Level3_l=Activation('relu')(Level3_l)
    
    
    Level4_l=Conv2D(filters=256,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level3_l)
    Level4_l=BatchNormalization(axis=-1)(Level4_l)
    Level4_l_shortcut=Level4_l
    Level4_l=Activation('relu')(Level4_l)
    #Level4_l=ZeroPadding2D(padding=(1,1))(Level4_l)
    Level4_l=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_l)
    Level4_l=BatchNormalization(axis=-1)(Level4_l)
    #Level4_l=InstanceNormalization(axis=-1)(Level4_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_l=Activation('relu')(Level4_l)
    #Level4_l=Dropout(0.5)(Level4_l)
    Level4_l=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_l)
    Level4_l=BatchNormalization(axis=-1)(Level4_l)
    #Level4_l=InstanceNormalization(axis=-1)(Level4_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_l=Add()([Level4_l,Level4_l_shortcut])
    Level4_l=Activation('relu')(Level4_l)


    Level5_l=Conv2D(filters=512,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level4_l)
    Level5_l=BatchNormalization(axis=-1)(Level5_l)
    Level5_l_shortcut=Level5_l
    Level5_l=Activation('relu')(Level5_l)
    #Level5_l=BatchNormalization(axis=-1)(Level5_l) 
    #Level5_l=ZeroPadding2D(padding=(1,1))(Level5_l)
    Level5_l=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level5_l)
    Level5_l=BatchNormalization(axis=-1)(Level5_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_l=Activation('relu')(Level5_l)
    #Level5_l=Dropout(0.5)(Level5_l)
    Level5_l=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level5_l)
    Level5_l=BatchNormalization(axis=-1)(Level5_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_l=Add()([Level5_l,Level5_l_shortcut])
    Level5_l=Activation('relu')(Level5_l)


    Level6_l=Conv2D(filters=1024,kernel_size=(3,3),strides=3,kernel_regularizer=regularizers.l2(reg))(Level5_l)
    Level6_l=BatchNormalization(axis=-1)(Level6_l)
    Level6_l_shortcut=Level6_l
    Level6_l=Activation('relu')(Level6_l)
    #Level5_l=BatchNormalization(axis=-1)(Level5_l) 
    #Level5_l=ZeroPadding2D(padding=(1,1))(Level5_l)
    Level6_l=Conv2D(filters=1024,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level6_l)
    Level6_l=BatchNormalization(axis=-1)(Level6_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level6_l=Activation('relu')(Level6_l)
    #Level5_l=Dropout(0.5)(Level5_l)
    Level6_l=Conv2D(filters=1024,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level6_l)
    Level6_l=BatchNormalization(axis=-1)(Level6_l)
    #Level5_l=InstanceNormalization(axis=-1)(Level5_l)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level6_l=Add()([Level6_l,Level6_l_shortcut])
    Level6_l=Activation('relu')(Level6_l)
    
    Level5_r=Conv2DTranspose(filters=512,kernel_size=(3,3),strides=3,kernel_regularizer=regularizers.l2(reg))(Level6_l)
    #Level4_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level5_l)
    #Level4_r=Conv2D(filters=256,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_r)
    Level5_r=BatchNormalization(axis=-1)(Level5_r)
    Level5_r_shortcut=Level5_r
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_r=Activation('relu')(Level5_r)
    merge5=Concatenate(axis=-1)([Level5_l,Level5_r])
    Level5_r=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge5)
    Level5_r=BatchNormalization(axis=-1)(Level5_r)
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_r=Activation('relu')(Level5_r)
    #Level4_r=Dropout(0.5)(Level4_r)
    Level5_r=Conv2D(filters=512,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level5_r)
    Level5_r=BatchNormalization(axis=-1)(Level5_r)
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level5_r=Add()([Level5_r,Level5_r_shortcut])
    Level5_r=Activation('relu')(Level5_r)

    
    Level4_r=Conv2DTranspose(filters=256,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level5_r)
    #Level4_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level5_l)
    #Level4_r=Conv2D(filters=256,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_r)
    Level4_r=BatchNormalization(axis=-1)(Level4_r)
    Level4_r_shortcut=Level4_r
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_r=Activation('relu')(Level4_r)
    merge4=Concatenate(axis=-1)([Level4_l,Level4_r])
    Level4_r=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge4)
    Level4_r=BatchNormalization(axis=-1)(Level4_r)
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_r=Activation('relu')(Level4_r)
    #Level4_r=Dropout(0.5)(Level4_r)
    Level4_r=Conv2D(filters=256,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level4_r)
    Level4_r=BatchNormalization(axis=-1)(Level4_r)
    #Level4_r=InstanceNormalization(axis=-1)(Level4_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level4_r=Add()([Level4_r,Level4_r_shortcut])
    Level4_r=Activation('relu')(Level4_r)
    
    
    Level3_r=Conv2DTranspose(filters=128,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level4_r)
    #Level3_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level4_r)
    #Level3_r=Conv2D(filters=128,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_r)
    Level3_r=BatchNormalization(axis=-1)(Level3_r)
    Level3_r_shortcut=Level3_r
    #Level3_r=InstanceNormalization(axis=-1)(Level3_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_r=Activation('relu')(Level3_r)
    merge3=Concatenate(axis=-1)([Level3_l,Level3_r])
    Level3_r=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge3)
    Level3_r=BatchNormalization(axis=-1)(Level3_r)
    #Level3_r=InstanceNormalization(axis=-1)(Level3_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_r=Activation('relu')(Level3_r)
    #Level3_r=Dropout(0.5)(Level3_r)
    Level3_r=Conv2D(filters=128,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level3_r)
    Level3_r=BatchNormalization(axis=-1)(Level3_r)
    #Level3_r=InstanceNormalization(axis=-1)(Level3_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level3_r=Add()([Level3_r,Level3_r_shortcut])
    Level3_r=Activation('relu')(Level3_r)
    
    
    Level2_r=Conv2DTranspose(filters=64,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level3_r)
    #Level2_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level3_r)
    #Level2_r=Conv2D(filters=64,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_r)
    Level2_r=BatchNormalization(axis=-1)(Level2_r)
    Level2_r_shortcut=Level2_r
    #Level2_r=InstanceNormalization(axis=-1)(Level2_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_r=Activation('relu')(Level2_r)
    merge2=Concatenate(axis=-1)([Level2_l,Level2_r])
    Level2_r=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge2)
    Level2_r=BatchNormalization(axis=-1)(Level2_r)
    #Level2_r=InstanceNormalization(axis=-1)(Level2_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_r=Activation('relu')(Level2_r)
    #Level2_r=Dropout(0.5)(Level2_r)
    Level2_r=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level2_r)
    Level2_r=BatchNormalization(axis=-1)(Level2_r)
    #Level2_r=InstanceNormalization(axis=-1)(Level2_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level2_r=Add()([Level2_r,Level2_r_shortcut])
    Level2_r=Activation('relu')(Level2_r)
    
    
    Level1_r=Conv2DTranspose(filters=32,kernel_size=(2,2),strides=2,kernel_regularizer=regularizers.l2(reg))(Level2_r)
    #Level1_r=UpSampling2D(size=(2, 2),interpolation='nearest')(Level2_r)
    #Level1_r=Conv2D(filters=32,kernel_size=(2,2),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_r)
    Level1_r=BatchNormalization(axis=-1)(Level1_r)
    Level1_r_shortcut=Level1_r
    #Level1_r=InstanceNormalization(axis=-1)(Level1_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_r=Activation('relu')(Level1_r)
    merge1=Concatenate(axis=-1)([Level1_l,Level1_r])
    Level1_r=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(merge1)
    Level1_r=BatchNormalization(axis=-1)(Level1_r)
    #Level1_r=InstanceNormalization(axis=-1)(Level1_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_r=Activation('relu')(Level1_r)
    #Level1_r=Dropout(0.5)(Level1_r)
    Level1_r=Conv2D(filters=32,kernel_size=(3,3),strides=1,padding='same',kernel_regularizer=regularizers.l2(reg))(Level1_r)
    Level1_r=BatchNormalization()(Level1_r)
    #Level1_r=InstanceNormalization(axis=-1)(Level1_r)  ## Instance Normalization. Use InstanceNormalization() for Layer Normalization.
    Level1_r=Add()([Level1_r,Level1_r_shortcut])
    Level1_r=Activation('relu')(Level1_r)
    output=Conv2D(filters=13,kernel_size=(1,1),strides=1,kernel_regularizer=regularizers.l2(reg))(Level1_r)
    #output=BatchNormalization(axis=-1)(output)
    output=Lambda(lambda x : softmax(x,axis=-1))(output)
    output=Concatenate(axis=-1)([output,weight_matrix])
    model=Model(inputs=inputs,outputs=output)
    return model



def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))  ## l1 regularization. Implement per-layer as kernel_regularizer=l1_reg




class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

def weighted_loss(y_true,y_pred):
    weight_matrix=K.flatten(y_pred[:,:,:,-1])
    #weight_matrix=K.ones(K.shape(K.flatten(y_pred[:,:,:,-1])))
    y_pre=y_pred[:,:,:,:-1]
    E=-1/(1)*K.dot(K.transpose(K.expand_dims(weight_matrix,axis=-1)),K.expand_dims(K.log(K.flatten(tf.math.reduce_sum(tf.multiply(y_true,y_pre),-1))),axis=-1))
    return E[:,0]



def generate_data(batch_size,path):
    l=list(range(1,4501))
    while True:
        s=random.choice(l)
        l.remove(s)
        arr=np.load(os.path.join(path,'train_'+str(s)+'.npy'))
        X=np.expand_dims(np.stack([arr[:,:,0],arr[:,:,-1]],axis=-1),0)
        y=np.expand_dims(arr[:,:,1:-1],0)
        for k in range(2,batch_size+1):
            r=random.choice(l)
            l.remove(r)
            arr=np.load(os.path.join(path,'train_'+str(r)+'.npy'))
            X_k=np.expand_dims(np.stack([arr[:,:,0],arr[:,:,-1]],axis=-1),0)
            y_k=np.expand_dims(arr[:,:,1:-1],0)
            X=np.concatenate([X,X_k],axis=0)
            y=np.concatenate([y,y_k],axis=0)
        if l==[]:
            l=list(range(1,4501))
        yield X,y

def generate_data_val(batch_size,path):
    l=list(range(1,501))
    while True:
        s=random.choice(l)
        l.remove(s)
        arr=np.load(os.path.join(path,'train_'+str(s)+'.npy'))
        X=np.expand_dims(np.stack([arr[:,:,0],arr[:,:,-1]],axis=-1),0)
        y=np.expand_dims(arr[:,:,1:-1],0)
        for k in range(2,batch_size+1):
            r=random.choice(l)
            l.remove(r)
            arr=np.load(os.path.join(path,'train_'+str(r)+'.npy'))
            X_k=np.expand_dims(np.stack([arr[:,:,0],arr[:,:,-1]],axis=-1),0)
            y_k=np.expand_dims(arr[:,:,1:-1],0)
            X=np.concatenate([X,X_k],axis=0)
            y=np.concatenate([y,y_k],axis=0)
        if l==[]:
            l=list(range(1,501))
        yield X,y

class DataGenerator(Sequence):
    def __init__(self, path, list_X=list(range(1,4501)), batch_size=20, dim=(432,432), shuffle=True):
        'Initialization'
        self.dim=dim
        self.batch_size = batch_size
        self.list_X = list_X
        self.path = path
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_X_temp = [self.list_X[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_X_temp, self.path)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_X_temp, path):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 2))
        y = np.empty((self.batch_size, *self.dim, 13))

        # Generate data
        for i, j in enumerate(list_X_temp):
            # Store sample
            arr=np.load(os.path.join(path,'train_'+str(j)+'.npy'))
            
            X[i,] = np.stack([arr[:,:,0],arr[:,:,-1]],axis=-1)

            # Store class
            y[i,] = arr[:,:,1:-1]

        return X, y

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

'''
def evaluation_metric(y_true, y_pred):
    acc=0
    y_pred=y_pred[:,:,:,:-1]
    for j in range(1,13):
        elements_per_class=tf.math.reduce_sum(y_true[:,:,:,j])
        predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,j])
        if elements_per_class!=0 or predicted_per_class!=0:
            intersection=tf.math.reduce_sum(np.multiply(y_pred[:,:,:,j],y_true[:,:,:,j]))
            union=elements_per_class+predicted_per_class-intersection
            acc+=intersection/union
    return acc/12
'''

def evaluation_metric_J(y_true, y_pred): ## Probabilistic Jaccard
    acc=0
    y_pred=y_pred[:,:,:,:-1]
    for j in range(13):
        elements_per_class=tf.math.reduce_sum(y_true[:,:,:,j])
        predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,j])
        intersection=tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,j],y_true[:,:,:,j]))
        union=elements_per_class+predicted_per_class-intersection
        acc+=intersection/(union+0.000001)
    return acc/13

def evaluation_metric(y_true, y_pred): ## Probabilistic Dice
    acc=0
    y_pred=y_pred[:,:,:,:-1]
    for j in range(13):
        elements_per_class=tf.math.reduce_sum(y_true[:,:,:,j])
        predicted_per_class=tf.math.reduce_sum(y_pred[:,:,:,j])
        intersection=tf.math.scalar_mul(2.0,tf.math.reduce_sum(tf.math.multiply(y_pred[:,:,:,j],y_true[:,:,:,j])))
        union=elements_per_class+predicted_per_class
        acc+=intersection/(union+0.000001)
    return acc/13





