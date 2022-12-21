import numpy as np
import matplotlib.pyplot as plt
from torch import rand
import trimesh
import os 
import open3d.visualization as o3d
import cv2
import math
import glob
from PIL import Image, ImageFilter
import random


def blur(img):
    blurred = img.filter(ImageFilter.GaussianBlur)
    
    blurred = blurred.filter(ImageFilter.BoxBlur(radius=2))
    return blurred

def h_flip(img):
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped

def rotate_90(img):
    rotated = img.transpose(Image.ROTATE_90)
    return rotated

def rand_augment(img):
    augmented = Image.fromarray(np.asarray(img))
    augmented = augmented.convert("L")

    # choose 2 from rotate,flip, and blur
    # 0: blur   1: flip horizontal     2: rotate 90
    choices = np.arange(0,3,1)
    np.random.shuffle(choices)

    picks = choices[0:2]
    
    print('Augment Picks:  ', picks)

    for pick in picks:
        if pick == 0:
            augmented = blur(augmented)
        elif pick == 1:
            augmented = h_flip(augmented)
        elif pick == 2:
            augmented = rotate_90(augmented)
    
    return augmented



if __name__ == "__main__":
    
    # Get Subfolders 
    soruce_path = os.getcwd()+'/assets/ModelNet10/ModelNet10/'
    folders = os.listdir(soruce_path)
    subtype = ['/test/','/train/']

    
    vis = o3d.Visualizer()
    vis.("PCD", 1200, 1200)

    for f in folders:
        for s in subtype:
            target_path = os.getcwd()+'/assets/ModelNet10/DepthMap/'+s+f+'/'

            if os.path.exists(target_path) == False:
                print('Creating Folder:   ',target_path)
                os.makedirs(target_path)
           
            filelist = glob.glob(soruce_path + f + s +'*.off', recursive=True)
            fname = [(os.path.split(i)[1]).split('.')[0] for i in filelist]

            for i,name in enumerate(fname):
                mesh = trimesh.load(filelist[i])
                points = mesh.sample(128*2048)

                pointcloud = np.asarray(points)
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointcloud)

                # cam = o3d.visualization.rendering.Camera()

                vis.add_geometry(pcd)
                # vis.update_geometry(pcd)

                x = 0
                y = -60

                x_rot = (x*math.pi/180)/0.003
                y_rot = (y*math.pi/180)/0.003

                ctr = vis.get_view_control()
                # ctr.rotate(math.asin(math.tan(30*math.pi/180))*180/math.pi, 45)
                
                # Idempotent
                # ctr.rotate(2094, 2094)    
                
                ctr.rotate(2094/6, -2094/8)    

                # vis.run()
                depth = vis.capture_depth_float_buffer(True)
                image = vis.capture_screen_float_buffer(True)
                
               
                augmented_depth = rand_augment(depth)
                print(name)
                
                plt.imsave(target_path+name+'_augment.png', np.asarray(augmented_depth), dpi=1, cmap='gray')
                plt.close()

                plt.imsave(target_path+name+'.png', np.asarray(depth), dpi=1, cmap='gray')
                plt.close()
                vis.remove_geometry(pcd)

                

    vis.destroy_window()

                
