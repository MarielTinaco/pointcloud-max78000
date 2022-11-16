import numpy as np

def pointcloud_to_depth_map(pointcloud: np.ndarray, theta_res=150, phi_res=32, max_depth=50, phi_min_degrees=60,
                            phi_max_degrees=100) -> np.ndarray:
    """
        All params are set so they match default carla lidar settings
    """
    assert pointcloud.shape[1] == 3, 'Must have (N, 3) shape'
    assert len(pointcloud.shape) == 2, 'Must have (N, 3) shape'

    xs = pointcloud[:, 0]
    ys = pointcloud[:, 1]
    zs = pointcloud[:, 2]

    rs = np.sqrt(np.square(xs) + np.square(ys) + np.square(zs))

    phi_min = np.deg2rad(phi_min_degrees)
    phi_max = np.deg2rad(phi_max_degrees)
    phi_range = phi_max - phi_min
    phis = np.arccos(zs / rs)

    THETA_MIN = -np.pi
    THETA_MAX = np.pi
    THETA_RANGE = THETA_MAX - THETA_MIN
    thetas = np.arctan2(xs, ys)

    phi_indices = ((phis - phi_min) / phi_range) * (phi_res - 1)
    phi_indices = np.rint(phi_indices).astype(np.int16)

    theta_indices = ((thetas - THETA_MIN) / THETA_RANGE) * theta_res
    theta_indices = np.rint(theta_indices).astype(np.int16)
    theta_indices[theta_indices == theta_res] = 0

    normalized_r = rs / max_depth

    canvas = np.ones(shape=(theta_res, phi_res), dtype=np.float32)
    # We might need to filter out out-of-bound phi values, if min-max degrees doesnt match lidar settings
    canvas[theta_indices, phi_indices] = normalized_r

    return canvas

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import trimesh
    import os 
    import open3d as o3d
    import cv2 as cv
    import math

    mesh = trimesh.load("assets/table_0001.off")
    points = mesh.sample(128*2048)

    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # ax.set_axis_off()

    pointcloud = np.asarray(points)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    vis = o3d.visualization.Visualizer()

    vis.create_window("PCD", 1200, 1200)

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
    
    ctr.rotate(2094/8, -2094/8)    

    # ctr.rotate(2094/10, y_rot)
    # ctr.rotate(0,-2094/8)
    # custom_draw_geometry_with_camera_trajectory.index = -1
    # custom_draw_geometry_with_camera_trajectory.trajectory =\
    #     o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    # custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    # )

    # glb = custom_draw_geometry_with_camera_trajectory()
    # glb.index >= 0:


    # vis.run()
    depth = vis.capture_depth_float_buffer(True)
    image = vis.capture_screen_float_buffer(True)
    # vis.destroy_window()

    
    plt.imsave(os.path.join('', '{:05d}.png'.format(1)),
                    np.asarray(depth),
                    dpi=1, cmap='gray')
    plt.imsave(os.path.join('', '{:05d}.png'.format(2)),
                np.asarray(image),
                dpi=1)

    # vis.update_renderer()
    
    # ch = cv.waitKey(1)
    # if ch & 0xFF == ord('q'):
    #     vis.destroy_window()


    # x_min = np.min(pointcloud[:,0])
    # x_max = np.max(pointcloud[:,0])

    # y_min = np.min(pointcloud[:,1])
    # y_max = np.max(pointcloud[:,1])

    # z_min = np.min(pointcloud[:,2])
    # z_max = np.max(pointcloud[:,2])

    # dx = x_max - x_min
    # dy = y_max - y_min
    # dz = z_max - z_min

    # resize_width = 8
    # resize_height = 6

    # # fig.set_figheight(resize_height)
    # # fig.set_figwidth(resize_width)


    # width = 128
    # height = 128

    # w = width - 1
    # h = height -1

    # figaro = np.zeros(shape=(width, height))

    # data = list(range(32*2048))

    # index = []
    # values = []




    # for p in pointcloud:
    #     col = round(((p[0] - x_min)/dx)*w)
    #     row = round(((y_max - p[1])/dy)*h)
    #     val = ((p[2] - z_min)/dz)*255

    #     i = 4*(width*row + col)
    #     index.append(i)
    #     values.append(val)

    #     if data[i] < val:
    #         data[i] =  val
    #         data[i+1]=  val
    #         data[i+2]=  val

    # figaro.data = np.asarray(data)

    # print(max(values))
    # print(min(values))

    # plt.imshow(figaro, cmap='gray_r')
    # plt.show()


    # pointcloud = np.load("lidar.npy")

    # depth_map = pointcloud_to_depth_map(pointcloud)

    # depth_map = depth_map * 256
    # depth_map = np.flip(depth_map, axis=1) # so floor is down
    # depth_map = np.swapaxes(depth_map, 0, 1)

    # plt.imshow(depth_map, cmap='gray_r')
    # plt.show()