import subprocess
import sys
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run projection.py, segmentation.py, scale_field.py, and normal_vectors.py \
            scripts with the specified arguments."
    )
    parser.add_argument("date", help="Date in the format YYYY_MM_DD")
    parser.add_argument("drive", type=int, help="Drive, positive integer")
    parser.add_argument("cam", type=int, help="Camera number, 0, 1, 2, or 3")
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

    date = args.date
    drive = args.drive
    cam = args.cam
    prompt = args.prompt
    interpolation = args.interpolation
    debug = args.debug

    # projection
    args = [date, str(drive), str(cam)]
    if debug:
        args.append("--debug")
    run_script("projection.py", args)

    # segmentation
    args = [date, str(drive), str(cam), "--prompt", prompt]
    if debug:
        args.append("--debug")
    run_script("segmentation.py", args)

    # scale_field
    args = [
        date,
        str(drive),
        str(cam),
        "--prompt",
        prompt,
        "--interpolation",
        interpolation,
    ]
    if debug:
        args.append("--debug")
    run_script("scale_field.py", args)

    # normal_vectors
    args = [
        date,
        str(drive),
        str(cam),
        "--prompt",
        prompt,
        "--interpolation",
        interpolation,
    ]
    if debug:
        args.append("--debug")
    run_script("normal_vectors.py", args)


if __name__ == "__main__":
    main()
