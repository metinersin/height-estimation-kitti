import os
import argparse
import pickle as pkl
from typing import Literal

from tqdm import tqdm
import numpy as np

import kitti
import plotting
import point_cloud as pc
import arg_utils as au


def parse_args() -> tuple[
    str,
    int,
    Literal[0, 1, 2, 3],
    str,
    Literal["linear", "nearest", "cubic"] | None,
    bool,
]:
    """
    Parse the command line arguments and return them as a tuple.

    Returns:
        date (str): Date in YYYY_MM_DD format.
        drive (int): Drive number. It is a positive integer.
        cam (0 | 1 | 2 | 3): Cam no.
        debug (bool): Whether to save points on images.
    """
    parser = argparse.ArgumentParser(
        description="Calculate the scale field produced by the velodyne points at the \
            specified date, drive and cam onto the image plane. Scale field is calculated inside a \
            segmentation mask, so you need to run the segmentation.py first. Example usage: python \
            scale_field.py 2011_09_26 2 2 --prompt ground --debug"
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
        "--prompt", type=str, default="ground", help="Prompt value (default: 'ground')"
    )
    parser.add_argument(
        "--interpolation",
        default="linear",
        choices=["linear", "nearest", "cubic", None],
        help="Interpolation method",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save points on images",
    )

    # Parse the arguments
    args = parser.parse_args()
    return args.date, args.drive, args.cam, args.prompt, args.interpolation, args.debug


def main() -> None:

    # read the dataset parameters, interpolation mode, and the segmentation prompt
    date, drive, cam, prompt, interpolation, debug = parse_args()

    print(f"Main folder: {kitti.DATASET_PATH}")

    data_path = kitti.data_path(date, drive)
    print(f"Current folder: {data_path}")

    image_data_path = kitti.image_data_path(date, drive, cam)
    print(f"Images folder: {image_data_path}")

    num_img = len(os.listdir(image_data_path))
    print(f"Number of images: {num_img}")

    mask_data_path = kitti.mask_data_path(date, drive, cam, prompt)
    print(f"Segmentation mask folder: {mask_data_path}")

    img_shape = kitti.image_shape(date, drive, cam)
    img_size = img_shape[:2]
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
    output_folder = os.path.join(data_path, f"scale_field_{cam:02}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Output folder: {output_folder}")

    # debug output folder
    if debug:
        debug_folder = os.path.join(data_path, f"scale_field_{cam:02}_debug")
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        print(f"Debug mode is active.")
        print(f"Debug folder: {debug_folder}")

    for frame in tqdm(range(num_frames), desc="Calculating the scale field"):

        # read the points
        points_lidar = kitti.velodyne_data(date, drive, frame)

        # calculate the scale field
        scale_field_on_lidar = pc.scale_field_on_lidar(
            points_lidar,
            img_size=img_size,
            velo_to_cam=velo_to_cam,
            r_rect=r_rect,
            p_rect=p_rect,
        )

        # read the mask
        mask = kitti.mask(date, drive, cam, prompt, frame)

        # transform the scale field to the image plane
        scale_field_on_img = pc.to_field_on_img(
            scale_field_on_lidar,
            points_lidar,
            mask=mask,
            interpolation=interpolation,
            img_size=img_size,
            velo_to_cam=velo_to_cam,
            r_rect=r_rect,
            p_rect=p_rect,
        )

        # save the scale field
        fname = os.path.join(output_folder, f"{frame:010}.npy")
        np.save(fname, scale_field_on_img)

        # create and save the debug image if debug mode is enabled
        if debug:
            # read the image
            img = kitti.image(date, drive, cam, frame)

            # save the projected image
            debug_fname = os.path.join(debug_folder, f"{frame:010}.png")
            plotting.draw_field_on_image(
                scale_field_on_img,
                img,
                alpha=0.5,
                output_name=debug_fname,
                vmin=0,
                vmax=250,
            )


if __name__ == "__main__":

    import warnings

    warnings.filterwarnings("ignore")

    main()
