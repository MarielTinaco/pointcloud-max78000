import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os 
import open3d as o3d
import cv2
import math
import glob

if __name__ == "__main__":
    
    # Get Subfolders 
    soruce_path = os.getcwd()+'/assets/ModelNet10/ModelNet10/'
    folders = os.listdir(soruce_path)
    subtype = ['/test/','/train/']

    vis = o3d.visualization.Visualizer()
    vis.create_window("PCD", 1200, 1200)

    for f in folders:
        for s in subtype:
            target_path = os.getcwd()+'/assets/ModelNet10/DepthMap/'+f+s

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
                
                # plt.imsave(target_path+name+'.png', np.asarray(depth), dpi=1, cmap='gray')
                # plt.close()
                vis.remove_geometry(pcd)

                

    vis.destroy_window()

                
