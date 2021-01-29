#!/usr/bin/env python
# coding: utf-8

"""
Created on Wed Jun 10 15:22:51 2020

@author: abramo
"""


import numpy as np
import nibabel as nib
import os
import skimage
from skimage.morphology import square
from scipy import ndimage
from scipy.ndimage import zoom
import math


def padorcut(arrayin, newSize, axis = None):
    nDims = arrayin.ndim
    
    # extend dimensions
    while nDims < len(newSize):
        arrayin = np.expand_dims(arrayin, nDims)
        nDims = arrayin.ndim
        
    if type(axis) is int:
        # check if newsz is iterable, otherwise assume it's a number
        try:
            newSz = newSize[axis]
        except:
            newSz = int(newSize)
        oldSz = arrayin.shape[axis]
        if oldSz < newSz:
            padBefore = int(math.floor(float(newSz - oldSz)/2))
            padAfter = int(math.ceil(float(newSz - oldSz)/2))
            padding = []
            for i in range(nDims):
                if i == axis:
                    padding.append( (padBefore, padAfter) )
                else:
                    padding.append( (0,0) )
            return np.pad(arrayin, padding, 'constant')
        elif oldSz > newSz:
            cutBefore = int(math.floor(float(oldSz - newSz)/2))
            cutAfter = int(math.ceil(float(oldSz - newSz)/2))
            slc = [slice(None)]*nDims
            slc[axis] = slice(cutBefore, -cutAfter)
            return arrayin[tuple(slc)]
        else:
            return arrayin
    else:
        for ax in range(nDims):
            arrayin = padorcut(arrayin, newSize, ax)
        return arrayin

def slice_along_zaxis(img,resolution):
    '''
    Function that divides the 3D MR image along the z-axis. 
    
    Input
    -----
    img : array-like of shape (432,432,img.shape[-1])
          The input 3D image stored as numpy array
    
    Output
    ------
    slices : dictionary
             Dictionary with keys the numbered names of the slices and with values the correspoding slice
    '''
    slices={}
    for z in range(img.shape[-1]):
        MODEL_RESOLUTION = np.array([1.037037, 1.037037])
        MODEL_SIZE = (432, 432)
        zoomFactor = resolution[0:2]/MODEL_RESOLUTION
        img1 = zoom(img[:,:,z], zoomFactor) # resample the image to the model resolution
        img2 = padorcut(img1, MODEL_SIZE)
        #values are the antitransposed of the original slice
        slices['slice_'+str(z)]=img2[::-1,::-1].T
    return slices

def mask(categorical_mask):
    segmentation_mask=np.zeros((432,432))
    for i in range(categorical_mask.shape[1]):
        for j in range(categorical_mask.shape[1]):
            segmentation_mask[i,j]=categorical_mask[0,i,j,:13].argmax()  
    return segmentation_mask

def mask_gamba(categorical_mask):
    segmentation_mask=np.zeros((432,432))
    for i in range(categorical_mask.shape[1]):
        for j in range(categorical_mask.shape[1]):
            segmentation_mask[i,j]=categorical_mask[0,i,j,:7].argmax()  
    return segmentation_mask

def segment_and_save(path,img,resolution,img_shape,netc,netg,netcg):
    '''
    Function that takes the numpy array obtained from the nifti image we want to
    segment, and segments slice by slice along the z-axis. It also returns a Nifti
    image of the same shape of img containing the final segmentation slice-by-slice.
    
    Input
    -----
    img : array-like of shape (432,432,number of slices)
    
    Output
    ------
    final_segmentation: array-like of the same shape of img
                        Nifti containing the final segmentation
    '''
    
    final_segmentation=np.zeros(img_shape)
    slices=slice_along_zaxis(img,resolution)
    medial=int(math.floor(img.shape[-1]/2))
    img128=skimage.transform.resize(slices['slice_'+str(medial)], #'slice_18'
                               (128,128),
                               mode='constant',
                               cval=0,
                               anti_aliasing=True,
                               preserve_range=True,
                               order=3)
    img128=img128/4096
    categoric=netcg.predict(np.expand_dims(img128,axis=0))
    if categoric[0].argmax()==0:
        for j in range(4,img.shape[-1]-5):
            img432=slices['slice_'+str(j)]
            segmentation=netc.predict(np.expand_dims(np.stack([img432,np.zeros((432,432))],axis=-1),axis=0))
            segmentationnum=mask(segmentation)
            MODEL_RESOLUTION = np.array([1.037037, 1.037037])
            zoomFactor = resolution[0:2]/MODEL_RESOLUTION
            segmentationnum = zoom(segmentationnum, 1/zoomFactor, order=0)
            segmentationnum2 = padorcut(segmentationnum, (img_shape[1],img_shape[0])).astype(np.int8)
            final_segmentation[:,:,j]=segmentationnum2[::-1,::-1].T
        aff_trans=np.eye(4)
        aff_trans[0,0]=resolution[0]
        aff_trans[1,1]=resolution[1]
        aff_trans[2,2]=resolution[2]
        final_segmentation=nib.Nifti1Image(final_segmentation,affine=aff_trans)
        save_path=os.path.join(path,'thigh_roi.nii.gz')
        nib.save(final_segmentation,save_path)
    if categoric[0].argmax()==1:
        for j in range(4,img.shape[-1]-5): 
            img432=slices['slice_'+str(j)]
            segmentation=netg.predict(np.expand_dims(np.stack([img432,np.zeros((432,432))],axis=-1),axis=0))
            segmentationnum=mask_gamba(segmentation)
            MODEL_RESOLUTION = np.array([1.037037, 1.037037])
            zoomFactor = resolution[0:2]/MODEL_RESOLUTION
            segmentationnum = zoom(segmentationnum, 1/zoomFactor, order=0)
            segmentationnum2 = padorcut(segmentationnum, (img_shape[1],img_shape[0])).astype(np.int8)
            final_segmentation[:,:,j]=segmentationnum2[::-1,::-1].T
        aff_trans=np.eye(4)
        aff_trans[0,0]=resolution[0]
        aff_trans[1,1]=resolution[1]
        aff_trans[2,2]=resolution[2]
        final_segmentation=nib.Nifti1Image(final_segmentation,affine=aff_trans)
        save_path=os.path.join(path,'leg_roi.nii.gz')
        nib.save(final_segmentation,save_path)
            
