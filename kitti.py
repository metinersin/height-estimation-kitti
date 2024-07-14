'''
This module provides utilities for working with the KITTI dataset, \
    specifically for accessing and visualizing data. The KITTI dataset is a \
    comprehensive set of data for mobile robotics and autonomous driving \
    research, including images, LiDAR scans, and GPS/IMU data collected from \
    various sensors mounted on a moving vehicle.

Constants:
    DATASET_PATH (str): The root path to the KITTI dataset on the local file system.
'''

import os
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "/mnt/d/KITTI Dataset"


def data_path(date: str, drive: int) -> str:
    """
    The path to the KITTI dataset for a specific date and drive.

    Parameters:
        date (str): The date of the dataset in the format 'YYYY_MM_DD'.
        drive (int): The drive number.

    Returns:
        str: The path to the KITTI dataset for the specified date and drive.
    """

    return os.path.join(DATASET_PATH, date, f'{date}_drive_{drive:04}_sync')


def velodyne_data_path(date: str, drive: int) -> str:
    """
    Returns the path to the velodyne data for a specific date and drive.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.

    Returns:
    - str: The path to the Velodyne data.
    """
    return os.path.join(data_path(date, drive), 'velodyne_points', 'data')


def velodyne_path(date: str, drive: int, frame: int) -> str:
    """
    Returns the path to the Velodyne data file for a specific date, drive, and frame.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.
    - frame (int): The frame number.

    Returns:
    - str: The path to the Velodyne data file.
    """
    return os.path.join(velodyne_data_path(date, drive), f'{frame:010}.bin')


def velodyne_data(date: str, drive: int, frame: int) -> np.ndarray:
    """
    Read the Velodyne data from the specified date, drive, and frame.

    Parameters:
        date (str): The date of the data in the format 'YYYY_MM_DD'.
        drive (int): The drive number.
        frame (int): The frame number.

    Returns:
        np.ndarray: The velodyne data as a numpy array with shape (N, 3), \
            where N is the number of points.
    """

    with open(velodyne_path(date, drive, frame), 'rb') as f:
        velo_data = np.fromfile(f, dtype=np.float32).reshape((-1, 4))
    velo_data = velo_data[:, :3]
    return velo_data


def image_data_path(date: str, drive: int, cam: int) -> str:
    """
    Returns the path to the image data for a specific date, drive, and camera.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.
    - cam (int): The camera number.

    Returns:
    - str: The path to the image data.
    """
    return os.path.join(data_path(date, drive), f'image_{cam:02}', 'data')


def image_path(date: str, drive: int, cam: int, frame: int) -> str:
    """
    Returns the path to the image file for a specific date, drive, camera, and frame.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.
    - cam (int): The camera number.
    - frame (int): The frame number.

    Returns:
    - str: The path to the image file.
    """
    return os.path.join(image_data_path(date, drive, cam), f'{frame:010}.png')


def image_shape(date: str, drive: int, cam: int) -> tuple[int, int] | tuple[int, int, int]:
    """
    Returns the shape of the images for a specific date, drive, and camera.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.
    - cam (int): The camera number.

    Returns:
    - tuple[int, int] | tuple[int, int, itn]: The shape of the images as a tuple (H, W) or (H, W, 3).
    """
    return image(date, drive, cam, 0).shape


def image(date: str, drive: int, cam: int, frame: int) -> np.ndarray:
    """
    Read the KITTI image data for a specific date, drive, camera, and frame.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number of the KITTI dataset.
    - cam (int): The camera number of the KITTI dataset.
    - frame (int): The frame number of the KITTI dataset.

    Returns:
    - np.ndarray: The image as a (H, W) or (H, W, 3) NumPy array.
    """
    return plt.imread(image_path(date, drive, cam, frame))


def find_line_starting_with(path: str, s: str) -> str | None:
    """
    Find the first line in a file that starts with a given string.

    Parameters:
        path (str): The path to the file.
        s (str): The string to search for at the beginning of a line.

    Returns:
        str | None: The first line that starts with the given string, or None \
            if no such line is found.
    """
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if line.startswith(s):
                return line.strip()
    return None


def velo_to_cam_calib_path(date: str) -> str:
    """
    Returns the path to the calibration file for the transformation from the \
    Velodyne to the camera.

    Parameters:
    - date (str): The date for which the calibration data is required. Its \
            format should be 'YYYY_MM_DD'.

    Returns:
    - str: The path to the calibration file.
    """
    return os.path.join(DATASET_PATH, date, "calib_velo_to_cam.txt")


def velo_to_cam_calib_matrix(date: str) -> np.ndarray:
    """
    Returns the Velodyne Lidar to camera calibration data for a specific date.

    Parameters:
        date (str): The date for which the calibration data is required. Its \
            format should be 'YYYY_MM_DD'.

    Returns:
        np.ndarray: 4x4 Velodyne Lidar to camera calibration matrix.

    Raises:
        ValueError: If the rotation or translation data is not found in the calibration file.
    """
    path = velo_to_cam_calib_path(date)

    rotation = find_line_starting_with(path, 'R')
    if rotation is None:
        raise ValueError(f'R not found in {path}')
    rotation = rotation.split()[1:]  # type: ignore
    rotation = np.asarray(list(map(float, rotation))
                          ).reshape(3, 3)  # type: ignore

    translation = find_line_starting_with(path, 'T')
    if translation is None:
        raise ValueError(f'T not found in {path}')
    translation = translation.split()[1:]  # type: ignore
    translation = np.asarray(list(map(float, translation))
                             ).reshape(3, 1)  # type: ignore

    velo_to_cam = np.column_stack((rotation, translation))
    velo_to_cam = np.row_stack((velo_to_cam, np.array([0, 0, 0, 1])))
    return velo_to_cam


def cam_to_cam_calib_path(date: str) -> str:
    """
    Returns the path to the calibration file for the transformation from the \
    camera to the camera.

    Parameters:
    - date (str): The date for which the calibration data is required. Its \
            format should be 'YYYY_MM_DD'.

    Returns:
    - str: The path to the camera to camera calibration file.
    """
    return os.path.join(DATASET_PATH, date, "calib_cam_to_cam.txt")


def r_rect(date: str) -> np.ndarray:
    """
    Return the R_rect matrix for a given date.

    Parameters:
        date (str): The date for which the calibration data is required. Its \
            format should be 'YYYY_MM_DD'.

    Returns:
        np.ndarray: 4x4 R_rect matrix as a numpy array.

    Raises:
        ValueError: If R_rect_00 is not found in the calibration file.
    """

    path = cam_to_cam_calib_path(date)

    line = find_line_starting_with(path, 'R_rect_00')
    if line is None:
        raise ValueError(
            f'R_rect_00 not found in {path}')

    r_rect = line.split()[1:]
    r_rect = np.asarray(list(map(float, r_rect))).reshape(3, 3)  # type: ignore
    tmp = np.eye(4)
    tmp[:3, :3] = r_rect
    r_rect = tmp  # type: ignore
    return r_rect  # type: ignore


def p_rect(date: str, cam: int) -> np.ndarray:
    """
    Return the P_rect matrix for a given date and camera.

    Parameters:
        date (str): The date for which the calibration data is required. Its \
            format should be 'YYYY_MM_DD'.
        cam (int): The camera number.

    Returns:
        np.ndarray: 3x4 P_rect matrix as a numpy array.

    Raises:
        ValueError: If P_rect_{cam:02} is not found in the calibration file.
    """

    path = cam_to_cam_calib_path(date)

    line = find_line_starting_with(path, f'P_rect_{cam:02}')
    if line is None:
        raise ValueError(f'P_rect_{cam:02} not found in {path}')

    p_rect = line.split()[1:]
    p_rect = np.asarray(list(map(float, p_rect))).reshape(3, 4)  # type: ignore
    return p_rect  # type: ignore


def calib_data(date: str, cam: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the calibration data for a specific date and camera.

    Parameters:
        date (str): The date for which the calibration data is required. Its \
            format should be 'YYYY_MM_DD'.
        cam (int): The camera index.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the \
            calibration matrices:
            - velo_to_cam: The transformation matrix from Velodyne coordinates \
                to camera coordinates.
            - r_rect_0: The rectifying rotation matrix for the left camera.
            - p_rect_cam: The projection matrix for the specified camera.
    """
    velo_to_cam = velo_to_cam_calib_matrix(date)
    r_rect_0 = r_rect(date)
    p_rect_cam = p_rect(date, cam)

    return velo_to_cam, r_rect_0, p_rect_cam  # type: ignore
