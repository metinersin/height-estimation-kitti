import os
import argparse

import numpy as np
from tqdm import tqdm

import kitti
import plotting
import img_utils as iu
import arg_utils as au


def parse_args() -> tuple[str, int, int, str, bool]:
    """
    Parse the command line arguments and return them as a tuple. Example usage: \
        python segmentation.py 2011_09_26 1 0 --prompt ground --debug

    Returns:
        date (str): Date in YYYY_MM_DD format.
        drive (int): Drive number. It is a positive integer.
        cam (0 | 1 | 2 | 3): Cam no.
        prompt (str): Prompt for segmentation. Defaults to 'ground'.
        debug (bool): Whether to save masked images.
    """
    parser = argparse.ArgumentParser(
        description="Segment the images at the specified date, drive and cam using the prompt. \
            Example usage: python segmentation.py 2011_09_26 1 0 --prompt ground --debug"
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
        "--prompt", type=str, default="ground", help="Prompt value (default: 'ground')"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to save masked images"
    )

    # Parse the arguments
    args = parser.parse_args()
    return args.date, args.drive, args.cam, args.prompt, args.debug


def main() -> None:

    # read the dataset parameters and the segmentation prompt
    date, drive, cam, prompt, debug = parse_args()

    # path to the images
    data_path = kitti.data_path(date, drive)
    image_data_path = kitti.image_data_path(date, drive, cam)
    num_img = len(os.listdir(image_data_path))
    img_shape = kitti.image_shape(date, drive, cam)

    # output folder
    output_folder = os.path.join(data_path, f"segmentation_{cam:02}", f"{prompt}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # debug output folder
    if debug:
        debug_folder = os.path.join(
            data_path, f"segmentation_{cam:02}", f"{prompt}_debug"
        )
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)

    # load the segmentation model
    # from mock_model import predict
    from lang_sam import LangSAM

    model = LangSAM()

    print()
    print(f"Segmenting images at {image_data_path}")
    print(f"Shape of the images: {img_shape}")
    print(f"Number of images: {num_img}")
    print(f"Segmentation prompt: {prompt}")
    print(f"Saving the masks to {output_folder}")

    if debug:
        print(f"Debug mode is enabled.")
        print(f"Saving masked images to {debug_folder}")
    print()

    for frame in tqdm(range(num_img), desc="Segmenting images"):
        # read the image
        img = kitti.image(date, drive, cam, frame)

        # segment the image and get the mask
        mask = iu.segment(model, img, prompt, min_size_ratio=0.002)

        # save the mask
        fname = os.path.join(output_folder, f"{frame:010}.npy")
        np.save(fname, mask)

        # create and save the masked image if debug mode is enabled
        if debug:
            masked_img = iu.apply_mask(img, mask, alpha=0.2, color=(255, 0, 0))
            debug_fname = os.path.join(debug_folder, f"{frame:010}.png")
            plotting.draw(masked_img, output_name=debug_fname)


if __name__ == "__main__":

    import warnings

    warnings.filterwarnings("ignore")

    main()
