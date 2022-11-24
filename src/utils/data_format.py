import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os 
import open3d as o3d
import cv2
import math
import glob
from PIL import Image, ImageEnhance, ImageChops
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


def enhance_img(img):
    enhancer = ImageEnhance.Contrast(img)

    enhanced = enhancer.enhance(2)

    return enhanced


def darken_bg(img): 
    
    _, bw_img = cv2.threshold(np.asarray(enhance_img(img)), 200, 255, cv2.THRESH_BINARY_INV)

    bw_img = Image.fromarray(bw_img)

    img_rm_bg = ImageChops.darker(img, enhance_img(bw_img))

    return img_rm_bg
    

def crop_by_index(indices,target_dir,filelist,is_save = True):
    fname = [(os.path.split(i)[1]).split('.')[0] for i in filelist]

    if os.path.exists(target_dir) == False:
        print('Creating Folder:   ',target_dir)
        os.makedirs(target_dir)

    for index in indices:
        print(fname[index])
        raw_img = Image.open(filelist[index])

        left,center,right = cropper(img = raw_img)

        left = darken_bg(left)
        center = darken_bg(center)
        right = darken_bg(right)
        
        plt.imsave(target_dir+fname[index]+'_L.png', np.asarray(left), dpi=1, cmap='gray_r')
        plt.imsave(target_dir+fname[index]+'_C.png', np.asarray(center), dpi=1, cmap='gray_r')
        plt.imsave(target_dir+fname[index]+'_R.png', np.asarray(right), dpi=1, cmap='gray_r')


if __name__ == "__main__":
    # Get Subfolders 
    SOURCE_DIR = os.getcwd()+'/assets/depthmap_data/depthmap_data/'
    folders = os.listdir(SOURCE_DIR)
    
    subtypes = ['/train/' , '/test/']

    for f in folders:
        filelist = glob.glob(SOURCE_DIR + f +'/*.png', recursive=True)
        fname = [(os.path.split(i)[1]).split('.')[0] for i in filelist]

        indices = np.arange(0,len(fname),1)
        random.shuffle(indices)

        train_i = indices[:int(len(indices)*0.7)]  # TRAIN indices
        test_i = indices[int(len(indices)*0.7):]   # TEST  indices
        
        for s in subtypes:
            DIR = os.getcwd()+'/assets/depthmap_data/DepthMap/' + s + f + '/'
        
            crop_by_index(indices=train_i, target_dir=DIR, filelist=filelist, is_save = True)

        

                
                
