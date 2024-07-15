import os
import argparse
import pickle as pkl

from tqdm import tqdm

import kitti
import point_cloud as pc
import arg_utils as au


def parse_args() -> tuple[str, int, int, bool]:
    """
    Parse the command line arguments and return them as a tuple. Example usage: \
        python projection.py 2011_09_26 1 0 --debug

    Returns:
        date (str): Date in YYYY_MM_DD format.
        drive (int): Drive number. It is a positive integer.
        cam (0 | 1 | 2 | 3): Cam no.
        debug (bool): Whether to save points on images.
    """
    parser = argparse.ArgumentParser(
        description="Project the velodyne points at the specified date, drive and cam onto the image plane. \
            Example usage: python projection.py 2011_09_26 1 0 --debug"
    )

    # Add arguments
    parser.add_argument("date", type=au.valid_date, help="Date in YYYY_MM_DD format")
    parser.add_argument(
        "drive", type=au.positive_int, help="Drive number (non-negative integer)"
    )
    parser.add_argument(
        "cam", type=au.valid_cam, help="Cam number (must be 0, 1, 2, or 3)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save points on images",
    )

    # Parse the arguments
    args = parser.parse_args()
    return args.date, args.drive, args.cam, args.debug


def main() -> None:

    # read the dataset parameters and the segmentation prompt
    date, drive, cam, debug = parse_args()

    # path to the images and velodyne points
    data_path = kitti.data_path(date, drive)
    image_data_path = kitti.image_data_path(date, drive, cam)
    velodyne_data_path = kitti.velodyne_data_path(date, drive)
    num_img = len(os.listdir(image_data_path))
    num_frames = len(os.listdir(velodyne_data_path))
    img_shape = kitti.image_shape(date, drive, cam)

    assert num_img == num_frames, "Number of images and velodyne data do not match."

    # read the transformation matrices
    velo_to_cam, r_rect, p_rect = kitti.calib_data(date, cam)

    # output folder
    output_folder = os.path.join(data_path, f"velodyne_{cam:02}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # debug output folder
    if debug:
        debug_folder = os.path.join(data_path, f"velodyne_{cam:02}_debug")
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)

    print()
    print(f"Projecting velodyne points at {velodyne_data_path}")
    print(f"Number of point clouds: {num_frames}")
    print(f"Image folder: {image_data_path}")
    print(f"Shape of the images: {img_shape}")
    print(f"Number of images: {num_img}")
    print(f"Saving the masks to {output_folder}")

    if debug:
        print(f"Debug mode is enabled.")
        print(f"Saving debug images to {debug_folder}")
    print()

    print(f"Velodyne to camera transformation matrix:")
    print(f"{velo_to_cam}")
    print(f"Rectifying rotation matrix:")
    print(f"{r_rect}")
    print(f"Projection matrix:")
    print(f"{p_rect}")
    print()

    for frame in tqdm(range(num_frames), desc="Projecting points"):

        # read the points
        points_lidar = kitti.velodyne_data(date, drive, frame)

        if not debug:
            # project the points onto the image plane
            points_img, idx = pc.lidar_to_img(
                points_lidar,
                velo_to_cam=velo_to_cam,
                r_rect=r_rect,
                p_rect=p_rect,
                img_height=img_shape[0],
                img_width=img_shape[1],
            )

            # save the projected points
            fname = os.path.join(output_folder, f"{frame:010}.pkl")
            with open(fname, "wb") as f:
                pkl.dump((points_img, idx), f)

        # create and save the projected image if debug mode is enabled
        if debug:
            # read the image
            img = kitti.image(date, drive, cam, frame)

            # save the projected image
            debug_fname = os.path.join(debug_folder, f"{frame:010}.png")
            points_img, idx = pc.draw_velodyne_on_image(
                img,
                points_lidar,
                velo_to_cam=velo_to_cam,
                r_rect=r_rect,
                p_rect=p_rect,
                output_name=debug_fname,
            )

            # save the projected points
            fname = os.path.join(output_folder, f"{frame:010}.pkl")
            with open(fname, "wb") as f:
                pkl.dump((points_img, idx), f)


if __name__ == "__main__":

    import warnings

    warnings.filterwarnings("ignore")

    main()
