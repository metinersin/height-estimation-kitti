import argparse
import re

def valid_date(date_str: str) -> str:
    """Validate that the provided string matches the YYYY_MM_DD format."""
    pattern = r'^\d{4}_\d{2}_\d{2}$'
    if not re.match(pattern, date_str):
        raise argparse.ArgumentTypeError(
            f"Not a valid date: '{date_str}'. Expected format: YYYY_MM_DD")
    return date_str


def non_negative_int(value: str) -> int:
    """Validate that the provided string is a non-negative integer."""
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError(
                f"Not a non-negative integer: '{value}'")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid integer: '{value}'")


def positive_int(value: str) -> int:
    """Validate that the provided string is a positive integer."""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f"Not a positive integer: '{value}'")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid integer: '{value}'")


def valid_cam(value: str) -> int:
    """Validate that the provided string is one of {0, 1, 2, 3}."""
    try:
        ivalue = int(value)
        if ivalue not in {0, 1, 2, 3}:
            raise argparse.ArgumentTypeError(
                f"Not a valid cam value: '{value}'. Must be 0, 1, 2, or 3")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid integer: '{value}'")
