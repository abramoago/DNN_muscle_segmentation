#!/usr/bin/env python
# coding: utf-8

"""
Created on Wed Jun 16 15:27:31 2020

@author: abramo
"""


import nibabel as nib
import numpy as np
import os
import preprocessing_segmentation
import thigh
import leg
import thigh_leg



path='./Patients'

netc=thigh.unet()
netc.load_weights('./thigh/weights_thigh.hdf5')
netg=leg.unet()
netg.load_weights('./leg/weights_leg.hdf5')
netcg=thigh_leg.unet()
netcg.load_weights('./thigh_leg/weights_thighleg.hdf5')


for patient in os.listdir(path):
    path_save=os.path.join(path,patient)
    nii=nib.load(os.path.join(path_save,'TE1_mag.nii.gz'))
    resolution=nii.header['pixdim'][1:4]
    img_data = nii.get_data()
    img = np.asarray(img_data)
    preprocessing_segmentation.segment_and_save(path_save,img,resolution,nii.shape,netc,netg,netcg)
