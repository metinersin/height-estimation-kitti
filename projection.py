import os
import argparse
import pickle as pkl
from typing import Literal

from tqdm import tqdm

import kitti
import plotting
import point_cloud as pc
import arg_utils as au


def parse_args() -> tuple[str, int, Literal[0, 1, 2, 3], bool]:
    """
    Parse the command line arguments and return them as a tuple. Example usage: \
        python projection.py 2011_09_26 1 2 --debug

    Returns:
        date (str): Date in YYYY_MM_DD format.
        drive (int): Drive number. It is a positive integer.
        cam (0 | 1 | 2 | 3): Cam no.
        debug (bool): Whether to save points on images.
    """
    parser = argparse.ArgumentParser(
        description="Project the velodyne points at the specified date, drive on to the image plane. \
            Example usage: python projection.py 2011_09_26 1 2 --debug"
    )

    # Add arguments
    parser.add_argument("date", type=au.valid_date, help="Date in YYYY_MM_DD format")
    parser.add_argument(
        "drive", type=au.positive_int, help="Drive number (non-negative integer)"
    )
    parser.add_argument(
        "cam", type=int, choices=[0, 1, 2, 3], help="Cam number (must be 0, 1, 2, or 3)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save debugging images",
    )

    # Parse the arguments
    args = parser.parse_args()
    return args.date, args.drive, args.cam, args.debug


def main() -> None:

    # read the dataset parameters and the segmentation prompt
    date, drive, cam, debug = parse_args()

    print(f"Main folder: {kitti.DATASET_PATH}")

    data_path = kitti.data_path(date, drive)
    print(f"Current folder: {data_path}")

    image_data_path = kitti.image_data_path(date, drive, cam)
    print(f"Images folder: {image_data_path}")

    num_img = len(os.listdir(image_data_path))
    print(f"Number of images: {num_img}")

    img_shape = kitti.image_shape(date, drive, cam)
    print(f"Shape of the images: {img_shape}")

    velodyne_data_path = kitti.velodyne_data_path(date, drive)
    print(f"Velodyne folder: {velodyne_data_path}")

    num_velo = len(os.listdir(velodyne_data_path))
    print(f"Number of velodyne files: {num_velo}")

    assert num_img == num_velo, "Number of images and velodyne files do not match."
    num_frames = num_img

    # read the transformation matrices
    velo_to_cam, r_rect, p_rect = kitti.calib_data(date, cam)
    print(f"Velodyne to camera transformation matrix:")
    print(f"{velo_to_cam}")
    print(f"Rectifying rotation matrix:")
    print(f"{r_rect}")
    print(f"Projection matrix:")
    print(f"{p_rect}")

    # output folder
    output_folder = os.path.join(data_path, f"velodyne_{cam:02}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Output folder: {output_folder}")

    # debug output folder
    if debug:
        debug_folder = os.path.join(data_path, f"velodyne_{cam:02}_debug")
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        print(f"Debug mode is active.")
        print(f"Debug folder: {debug_folder}")

    for frame in tqdm(range(num_frames), desc="Projecting points"):

        # read the points
        points_lidar = kitti.velodyne_data(date, drive, frame)

        # convert from lidar coordinates to camera coordinates
        points_cam = pc.lidar_to_cam(points_lidar, velo_to_cam=velo_to_cam)

        # project the points onto the image plane
        points_img, idx = pc.cam_to_img(
            points_cam,
            r_rect=r_rect,
            p_rect=p_rect,
            img_height=img_shape[0],
            img_width=img_shape[1],
        )

        # save the projected points
        fname = os.path.join(output_folder, f"{frame:010}.pkl")
        with open(fname, "wb") as f:
            pkl.dump((points_img, idx), f)

        if debug:
            # read the image
            img = kitti.image(date, drive, cam, frame)

            # save the scatter plot
            debug_fname = os.path.join(debug_folder, f"{frame:010}.png")
            plotting.scatter_on_image(
                img,
                points_img[idx],
                c=points_cam[idx, 2],
                output_name=debug_fname,
            )


if __name__ == "__main__":

    import warnings

    warnings.filterwarnings("ignore")

    main()
