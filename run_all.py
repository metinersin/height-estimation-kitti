import os
import sys
import subprocess
import argparse

import kitti


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run projection.py, segmentation.py, scale_field.py, and normal_vectors.py \
            scripts on the entire dataset."
    )
    parser.add_argument("--prompt", help="Segmentation prompt")
    parser.add_argument(
        "--interpolation", help="Interpolation method, linear, nearest, cubic, None"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def run_script(script_name, args):

    command = [sys.executable, script_name] + args

    print(f"Running: {script_name} with {args}")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in process.stdout:  # type: ignore
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            print(f"Error: {script_name} exited with status {process.returncode}")

    except Exception as e:
        print(f"Error running {script_name}: {e}")


def main():
    args = parse_arguments()

    all_date_drive = kitti.all_date_drive()
    cam = 2
    prompt = args.prompt
    interpolation = args.interpolation
    debug = args.debug

    num = len(all_date_drive)

    print(f"Main folder: {kitti.DATASET_PATH}")
    print(f"Number of date-drives: {num}")

    for d in all_date_drive:
        date = d['date']
        drive = d['drive']
        print(f"Current date-drive: {date} - {drive}: ")

        # projection
        args = [date, str(drive), str(cam)]
        if debug:
            args.append('--debug')
        run_script("projection.py", args)

        # segmentation
        args = [date, str(drive), str(cam), '--prompt', prompt]
        if debug:
            args.append('--debug')
        run_script("segmentation.py", args)

        # scale_field
        args = [date, str(drive), str(cam), '--prompt', prompt, '--interpolation', interpolation]
        if debug:
            args.append('--debug')
        run_script("scale_field.py", args)

        # normal_vectors
        args = [date, str(drive), str(cam), '--prompt', prompt, '--interpolation', interpolation]
        if debug:
            args.append('--debug')
        run_script("normal_vectors.py", args)


if __name__ == "__main__":
    main()
