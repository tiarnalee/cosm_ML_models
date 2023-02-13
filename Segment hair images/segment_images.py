#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:17:49 2022

@author: tle19
"""

import os
os.chdir('/home/tle19/Documents/new_deeplab')
import torch
import cv2
import yaml
from predictors.predictor import Predictor
import numpy as np
import torchvision.transforms as T
import glob

#x Predict on test set
# im_paths=[os.path.join(root, name) for root, dirs, files in os.walk('/home/tle19/Desktop/ResNet_pretrained/Images') for name in files]

# Load config file
with open ('/home/tle19/Documents/new_deeplab/configs/config_inference.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Load predictor
predictor = Predictor(config, checkpoint_path='/home/tle19/Documents/new_deeplab/experiments/last.pth.tar')
base_size=config['image']['base_size']    

hair_types=['3A', '3B', '3C', '4A', '4B', '4C']

for hair_type in hair_types:
    # Path with images
    # This is set to load all images in folders 3A-4C
    im_paths=glob.glob('/home/tle19/Desktop/ResNet_pretrained/Images/'+hair_type+'/*')
    
    for index in range(len(im_paths)):
        # Produce segmentation mask for hair and background
        _, prediction = predictor.segment_image(im_paths[index])
    
        # Load original image
        image=cv2.imread(im_paths[index])
        
        # Crop original image
        image=T.functional.center_crop(torch.from_numpy(image.transpose(2, 0, 1)), (base_size, base_size)).numpy().transpose(1,2,0)
        
        # Multiply image by segmentation mask to produce segmented image
        segment_im=np.multiply(image, np.dstack((prediction, prediction, prediction)))
    
        cv2.imwrite(os.path.join('/home/tle19/Desktop/ResNet_pretrained/segmented_images/', hair_type, im_paths[index].split('/')[-1]), segment_im)
