'''Utility functions related to projection fo 3D points to 2D image plane.'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import kitti
import point_cloud


if __name__ == "__main__":

    matplotlib.use('Agg')

    output_dir = "projection-output"
    dataset_dir = "/mnt/d/KITTI Dataset"
    date = "2011_09_26"
    drive = 5
    cam = 2
    frame = 50

    velo_to_cam, r_rect, p_rect = kitti.calib_data(date, cam)
    print(f'Velo to cam:\n{velo_to_cam}')
    print(f'R rect:\n{r_rect}')
    print(f'P rect:\n{p_rect}')

    points_lidar = kitti.velodyne_data(date, drive, frame)
    print(f'Points lidar shape: {points_lidar.shape}')
    points_lidar_diameter = point_cloud.approximate_diameter(points_lidar)
    print(f'Diameter: {points_lidar_diameter}')
    point_cloud.plot_number_of_points_in_radius(
        points_lidar, np.linspace(0, points_lidar_diameter / 2, 100),
        sample_size=200,
    )

    img = kitti.image(date, drive, cam, frame)
    img_height, img_width = img.shape[0:2]
    print(f'Image shape: {img.shape}')
    print(f'Height: {img_height}')
    print(f'Width: {img_width}')

    points_img, idx = point_cloud.draw_velodyne_on_image(
        img, points_lidar, velo_to_cam, r_rect, p_rect, alpha=0.1,
        title='Velodyne Points',
        output_name='velodyne_points_on_image.png'
    )
    print(f'Points image shape: {points_img.shape}')
    print(f'Max height: {points_img[:, 0].max()}')
    print(f'Max width: {points_img[:, 1].max()}')
    print(f'Good points image shape: {points_img[idx, :].shape}')
    print(f'Max height: {points_img[idx, :][:, 0].max()}')
    print(f'Max width: {points_img[idx, :][:, 1].max()}')

    normal_vectors = point_cloud.normals(points_lidar)
    displaced_points_lidar = points_lidar + normal_vectors * 1
    displaced_points_img, idx = point_cloud.draw_velodyne_on_image(
        img, displaced_points_lidar, velo_to_cam, r_rect, p_rect, alpha=0.1,
        title='Displaced Velodyne Points',
        output_name='displaced_velodyne_points_on_image.png'
    )

    scale_field_on_lidar = point_cloud.scale_field_on_lidar(
        points_lidar, img_height, img_width,
        velo_to_cam=velo_to_cam,
        r_rect=r_rect,
        p_rect=p_rect,
        scaling=-0.3,
        neighbors=50,
        radius=10
    )
    print(f'Scale field on points shape: {scale_field_on_lidar.shape}')

    scale_field_on_img = point_cloud.to_field_on_img(
        scale_field_on_lidar, points_lidar,
        img_height, img_width,
        velo_to_cam=velo_to_cam,
        r_rect=r_rect,
        p_rect=p_rect
    )
    print(f'Scale field on image shape: {scale_field_on_img.shape}')

    point_cloud.draw_field_on_image(
        scale_field_on_img*100, img, title='Scale Field', output_name='scale_field_on_image.png')
    print(scale_field_on_img)
