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
    inputs=Input(shape=(128,128))
    reshape=Reshape((128,128,1))(inputs)

    reg=0.01
    
    #reshape=Dropout(0.0)(reshape)   ## Hyperparameter optimization only on visible layer
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

    Level_f=Flatten()(Level5_l)
    output=Dense(2,activation='softmax',kernel_regularizer=regularizers.l2(reg))(Level_f)
    #output=Dense(2,kernel_regularizer=regularizers.l2(reg))(Level_f)
    #output=Lambda(lambda x : softmax(x,axis=-1))(output)
    ##output=Dense(1,activation=contracted_sigmoid,kernel_regularizer=regularizers.l2(reg))(Level_f) #loss='binary_crossentropy'
    #output=BatchNormalization(axis=-1)(output)
    model=Model(inputs=inputs,outputs=output)
    return model


    

def contracted_sigmoid(x):
    return K.sigmoid(tf.math.scalar_mul(1,x))

def generate_data(batch_size,path):
    l=list(range(1,4501))
    while True:
        s=random.choice(l)
        l.remove(s)
        arr=np.load(os.path.join(path,'train_'+str(s)+'.npy'))
        categoric=np.load(os.path.join(path,'categoric_'+str(s)+'.npy'))
        X=np.expand_dims(arr,0)
        y=np.expand_dims(categoric,0)
        for k in range(2,batch_size+1):
            r=random.choice(l)
            l.remove(r)
            arr=np.load(os.path.join(path,'train_'+str(r)+'.npy'))
            categoric=np.load(os.path.join(path,'categoric_'+str(r)+'.npy'))
            X_k=np.expand_dims(arr,0)
            y_k=np.expand_dims(categoric,0)
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
        categoric=np.load(os.path.join(path,'categoric_'+str(s)+'.npy'))
        X=np.expand_dims(arr,0)
        y=np.expand_dims(categoric,0)
        for k in range(2,batch_size+1):
            r=random.choice(l)
            l.remove(r)
            arr=np.load(os.path.join(path,'train_'+str(r)+'.npy'))
            categoric=np.load(os.path.join(path,'categoric_'+str(r)+'.npy'))
            X_k=np.expand_dims(arr,0)
            y_k=np.expand_dims(categoric,0)
            X=np.concatenate([X,X_k],axis=0)
            y=np.concatenate([y,y_k],axis=0)
        if l==[]:
            l=list(range(1,501))
        yield X,y

class DataGenerator(Sequence):
    def __init__(self, path, list_X=list(range(1,4501)), batch_size=20, dim=(128,128), shuffle=True):
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
        X = np.empty((self.batch_size, *self.dim))
        #y = np.empty((self.batch_size, 1)) #Dense(1,sigmoid)
        y = np.empty((self.batch_size, 2))

        # Generate data
        for i, j in enumerate(list_X_temp):
            # Store sample
            arr=np.load(os.path.join(path,'train_'+str(j)+'.npy'))
            categoric=np.load(os.path.join(path,'categoric_'+str(j)+'.npy'))
            
            X[i,] = arr

            # Store class
            y[i,] = categoric

        return X, y






