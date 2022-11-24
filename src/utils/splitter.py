import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os 
import open3d as o3d
import cv2
import math
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import random

def cropper(img):
    # Crop data using pillow. data expected to be pillow object

    width, height = img.size

    if width <= height:
        img = img.rotate(90, Image.NEAREST, expand = 1)
  

    # Setting the points for cropped image
    # _ = img.crop((left, top, right, bottom))

    left = img.crop((0, 0, height, height))

    center = img.crop((width/2-height/2, 0, width/2+height/2, height))

    right = img.crop((width-height, 0, width, height))

    return left,center,right

if __name__ == "__main__":
    # Get Subfolders 
    SOURCE_DIR = os.getcwd()+'/assets/depthmap_data/depthmap_data/'
    folders = os.listdir(SOURCE_DIR)

    for f in folders:
        filelist = glob.glob(SOURCE_DIR + f +'/*.png', recursive=True)
        fname = [(os.path.split(i)[1]).split('.')[0] for i in filelist]

        
        indices = np.arange(0,len(fname),1)
        random.shuffle(indices)

        

        train_i = indices[:int(len(indices)*0.7)]  # TRAIN indices
        test_i = indices[int(len(indices)*0.7):]   # TEST  indices

        TRAIN_DIR = os.getcwd()+'/assets/depthmap_data/DepthMap/train/'+f+'/'
        if os.path.exists(TRAIN_DIR) == False:
            print('Creating Folder:   ',TRAIN_DIR)
            os.makedirs(TRAIN_DIR)

        for index in train_i:
            print(fname[index])
            raw_img = Image.open(filelist[index])

            left,center,right = cropper(img = raw_img)
            
            plt.imsave(TRAIN_DIR+fname[index]+'_L.png', np.asarray(left), dpi=1, cmap='gray')
            plt.imsave(TRAIN_DIR+fname[index]+'_C.png', np.asarray(center), dpi=1, cmap='gray')
            plt.imsave(TRAIN_DIR+fname[index]+'_R.png', np.asarray(right), dpi=1, cmap='gray')
        
        TEST_DIR = os.getcwd()+'/assets/depthmap_data/DepthMap/test/'+f+'/'
        if os.path.exists(TEST_DIR) == False:
            print('Creating Folder:   ',TEST_DIR)
            os.makedirs(TEST_DIR)

        for index in test_i:
            print(fname[index])
            raw_img = Image.open(filelist[index])

            left,center,right = cropper(img = raw_img)
            
            plt.imsave(TEST_DIR+fname[index]+'_L.png', np.asarray(left), dpi=1, cmap='gray')
            plt.imsave(TEST_DIR+fname[index]+'_C.png', np.asarray(center), dpi=1, cmap='gray')
            plt.imsave(TEST_DIR+fname[index]+'_R.png', np.asarray(right), dpi=1, cmap='gray')


        

                
                
