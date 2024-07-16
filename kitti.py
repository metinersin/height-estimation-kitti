"""
This module provides utilities for working with the KITTI dataset, \
    specifically for accessing and visualizing data. The KITTI dataset is a \
    comprehensive set of data for mobile robotics and autonomous driving \
    research, including images, LiDAR scans, and GPS/IMU data collected from \
    various sensors mounted on a moving vehicle.

Constants:
    DATASET_PATH (str): The root path to the KITTI dataset on the local file system.
"""

import os
from typing import Literal
import re

import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "/mnt/d/KITTI Dataset"


def all_date_drive() -> list[dict[str, str | int]]:  # type: ignore
    """
    Returns a list of dictionaries containing valid dates and drives from the KITTI dataset.

    Returns:
        A list of dictionaries, where each dictionary contains the 'date' and 'drive' information.
        The 'date' is a string representing the valid date, and 'drive' is an integer representing the drive number.
    """
    lst = []
    for date in os.listdir(DATASET_PATH):
        if not is_valid_date(date):
            continue

        for drive in os.listdir(os.path.join(DATASET_PATH, date)):
            if re.match(date + r"_drive_\d{4}_sync", drive) is None:
                continue

            drive = int(drive.strip().split("_")[4])

            lst.append({"date": date, "drive": drive})

    return lst


def is_valid_date(date: str) -> bool:
    """
    Check if the given date string is in the format 'YYYY_MM_DD'.

    Args:
        date (str): The date string to be validated.

    Returns:
        bool: True if the date string is in the correct format, False otherwise.
    """
    return re.match(r"^\d{4}_\d{2}_\d{2}$", date) is not None


def valid_date(date_str: str) -> str:
    """Validate that the provided string matches the YYYY_MM_DD format."""
    if not is_valid_date(date_str):
        raise ValueError(f"Not a valid date: '{date_str}'. Expected format: YYYY_MM_DD")
    return date_str


def data_path(date: str, drive: int) -> str:
    """
    The path to the KITTI dataset for a specific date and drive.

    Parameters:
        date (str): The date of the dataset in the format 'YYYY_MM_DD'.
        drive (int): The drive number.

    Returns:
        str: The path to the KITTI dataset for the specified date and drive.
    """

    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    return os.path.join(DATASET_PATH, date, f"{date}_drive_{drive:04}_sync")


def velodyne_data_path(date: str, drive: int) -> str:
    """
    Returns the path to the velodyne data for a specific date and drive.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.

    Returns:
    - str: The path to the Velodyne data.
    """
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    return os.path.join(data_path(date, drive), "velodyne_points", "data")


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
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    if frame < 0:
        raise ValueError(f"Frame number must be a non-negative integer: {frame}")
    return os.path.join(velodyne_data_path(date, drive), f"{frame:010}.bin")


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
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    if frame < 0:
        raise ValueError(f"Frame number must be a non-negative integer: {frame}")
    with open(velodyne_path(date, drive, frame), "rb") as f:
        velo_data = np.fromfile(f, dtype=np.float32).reshape((-1, 4))
    velo_data = velo_data[:, :3]
    return velo_data


def image_data_path(date: str, drive: int, cam: Literal[0, 1, 2, 3]) -> str:
    """
    Returns the path to the image data for a specific date, drive, and camera.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.
    - cam (Literal[0, 1, 2, 3]): The camera number.

    Returns:
    - str: The path to the image data.
    """
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    return os.path.join(data_path(date, drive), f"image_{cam:02}", "data")


def image_path(date: str, drive: int, cam: Literal[0, 1, 2, 3], frame: int) -> str:
    """
    Returns the path to the image file for a specific date, drive, camera, and frame.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.
    - cam (Literal[0, 1, 2, 3]): The camera number.
    - frame (int): The frame number.

    Returns:
    - str: The path to the image file.
    """
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    if frame < 0:
        raise ValueError(f"Frame number must be a non-negative integer: {frame}")
    return os.path.join(image_data_path(date, drive, cam), f"{frame:010}.png")


def mask_data_path(date: str, drive: int, cam: Literal[0, 1, 2, 3], prompt: str) -> str:
    """
    Constructs the path for the mask data based on the given parameters.

    Args:
        date (str): The date of the data.
        drive (int): The drive number.
        cam (Literal[0, 1, 2, 3]): The camera number.
        prompt (str): The prompt for the data.

    Returns:
        str: The path for the masked data.
    """
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    return os.path.join(data_path(date, drive), f"segmentation_{cam:02}", f"{prompt}")


def mask_path(
    date: str, drive: int, cam: Literal[0, 1, 2, 3], prompt: str, frame: int
) -> str:
    """
    Constructs the path for the mask file corresponding to the given parameters.

    Args:
        date (str): The date of the data.
        drive (int): The drive number.
        cam (Literal[0, 1, 2, 3]): The camera number.
        frame (int): The frame number.
        prompt (str): The prompt for the mask.

    Returns:
        str: The path to the mask file.
    """
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    if frame < 0:
        raise ValueError(f"Frame number must be a non-negative integer: {frame}")
    return os.path.join(mask_data_path(date, drive, cam, prompt), f"{frame:010}.npy")


def image_shape(
    date: str, drive: int, cam: Literal[0, 1, 2, 3]
) -> tuple[int, int] | tuple[int, int, int]:
    """
    Returns the shape of the images for a specific date, drive, and camera.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.
    - cam (Literal[0, 1, 2, 3]): The camera number.

    Returns:
    - tuple[int, int] | tuple[int, int, itn]: The shape of the images as a tuple (H, W) or (H, W, 3).
    """
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    return image(date, drive, cam, 0).shape  # type:ignore


def image(date: str, drive: int, cam: Literal[0, 1, 2, 3], frame: int) -> np.ndarray:
    """
    Read the KITTI image data for a specific date, drive, camera, and frame.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number of the KITTI dataset.
    - cam (Literal[0, 1, 2, 3]): The camera number of the KITTI dataset.
    - frame (int): The frame number of the KITTI dataset.

    Returns:
    - np.ndarray: The image as a (H, W) or (H, W, 3) NumPy array.
    """
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    if frame < 0:
        raise ValueError(f"Frame number must be a non-negative integer: {frame}")
    return plt.imread(image_path(date, drive, cam, frame))


def mask(
    date: str, drive: int, cam: Literal[0, 1, 2, 3], prompt: str, frame: int
) -> np.ndarray:
    """
    Read the mask for a specific date, drive, camera, prompt, and frame.

    Parameters:
    - date (str): The date of the data in the format 'YYYY_MM_DD'.
    - drive (int): The drive number.
    - cam (Literal[0, 1, 2, 3]): The camera number.
    - prompt (str): The prompt value.
    - frame (int): The frame number.

    Returns:
    - np.ndarray: The mask data as a NumPy array.
    """
    valid_date(date)
    if drive <= 0:
        raise ValueError(f"Drive number must be a positive integer: {drive}")
    if frame < 0:
        raise ValueError(f"Frame number must be a non-negative integer: {frame}")
    return np.load(mask_path(date, drive, cam, prompt, frame))


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
    with open(path, "r", encoding="utf8") as f:
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
    valid_date(date)
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
    valid_date(date)
    path = velo_to_cam_calib_path(date)

    rotation = find_line_starting_with(path, "R")
    if rotation is None:
        raise ValueError(f"R not found in {path}")
    rotation = rotation.split()[1:]  # type: ignore
    rotation = np.asarray(list(map(float, rotation))).reshape(3, 3)  # type: ignore

    translation = find_line_starting_with(path, "T")
    if translation is None:
        raise ValueError(f"T not found in {path}")
    translation = translation.split()[1:]  # type: ignore
    translation = np.asarray(list(map(float, translation))).reshape(
        3, 1
    )  # type: ignore

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
    valid_date(date)
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
    valid_date(date)

    path = cam_to_cam_calib_path(date)

    line = find_line_starting_with(path, "R_rect_00")
    if line is None:
        raise ValueError(f"R_rect_00 not found in {path}")

    r_rect = line.split()[1:]
    r_rect = np.asarray(list(map(float, r_rect))).reshape(3, 3)  # type: ignore
    tmp = np.eye(4)
    tmp[:3, :3] = r_rect
    r_rect = tmp  # type: ignore
    return r_rect  # type: ignore


def p_rect(date: str, cam: Literal[0, 1, 2, 3]) -> np.ndarray:
    """
    Return the P_rect matrix for a given date and camera.

    Parameters:
        date (str): The date for which the calibration data is required. Its \
            format should be 'YYYY_MM_DD'.
        cam (Literal[0, 1, 2, 3]): The camera number.

    Returns:
        np.ndarray: 3x4 P_rect matrix as a numpy array.

    Raises:
        ValueError: If P_rect_{cam:02} is not found in the calibration file.
    """
    valid_date(date)

    path = cam_to_cam_calib_path(date)

    line = find_line_starting_with(path, f"P_rect_{cam:02}")
    if line is None:
        raise ValueError(f"P_rect_{cam:02} not found in {path}")

    p_rect = line.split()[1:]
    p_rect = np.asarray(list(map(float, p_rect))).reshape(3, 4)  # type: ignore
    return p_rect  # type: ignore


def calib_data(
    date: str, cam: Literal[0, 1, 2, 3]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the calibration data for a specific date and camera.

    Parameters:
        date (str): The date for which the calibration data is required. Its \
            format should be 'YYYY_MM_DD'.
        cam (Literal[0, 1, 2, 3]): The camera index.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the \
            calibration matrices:
            - velo_to_cam: The transformation matrix from Velodyne coordinates \
                to camera coordinates.
            - r_rect_0: The rectifying rotation matrix for the left camera.
            - p_rect_cam: The projection matrix for the specified camera.
    """
    valid_date(date)

    velo_to_cam = velo_to_cam_calib_matrix(date)
    r_rect_0 = r_rect(date)
    p_rect_cam = p_rect(date, cam)

    return velo_to_cam, r_rect_0, p_rect_cam  # type: ignore
