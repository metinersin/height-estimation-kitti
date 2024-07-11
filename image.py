from typing import Any, Literal, Callable
import functools

import numpy as np
from PIL import Image as Pimage

HW = 'HW'
HW3 = 'HW3'
HW4 = 'HW4'
L = 'L'
RGB = 'RGB'
RGBA = 'RGBA'

NP_FORMATS = ['HW', 'HW3', 'HW4']
PIL_FORMATS = ['L', 'RGB', 'RGBA']

NP_TO_PIL = {HW: L, HW3: RGB, HW4: RGBA}
PIL_TO_NP = {L: HW, RGB: HW3, RGBA: HW4}

FORMATS = NP_FORMATS + PIL_FORMATS

Format = Literal['HW', 'HW3', 'HW4', 'L', 'RGB', 'RGBA']


def _type(img: Any) -> Format:
    """
    Get the format of the image.

    Parameters:
        img (Any): The image whose format is to be determined.

    Returns:
        Format: The format of the image.
    """

    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return 'HW'
        if img.ndim == 3:
            if img.shape[2] == 3:
                return 'HW3'
            if img.shape[2] == 4:
                return 'HW4'
        raise ValueError(f'Invalid image shape: {img.shape}')

    if isinstance(img, Pimage.Image):
        return img.mode

    raise ValueError(f'Invalid image type: {type(img)}')


def _pil_to_pil(img: Pimage.Image, mode: Literal['L', 'RGB', 'RGBA']) -> Pimage.Image:
    return img.convert(mode)


def _pil_to_np(img: Pimage.Image) -> np.ndarray:
    return np.array(img)


def _np_to_pil(img: np.ndarray) -> Pimage.Image:
    mode = NP_TO_PIL[_type(img)]
    return Pimage.fromarray(img, mode=mode).convert(mode)


def _converter(fmt1: Format, fmt2: Format) -> Callable:
    if fmt1 in PIL_FORMATS and fmt2 in PIL_FORMATS:
        return lambda img: _pil_to_pil(img, fmt2)

    if fmt1 in PIL_FORMATS and fmt2 in NP_FORMATS:
        return lambda img: _pil_to_np(_pil_to_pil(img, NP_TO_PIL[fmt2]))

    if fmt1 in NP_FORMATS and fmt2 in PIL_FORMATS:
        return lambda img: _pil_to_pil(_np_to_pil(img), fmt2)

    if fmt1 in NP_FORMATS and fmt2 in NP_FORMATS:
        return lambda img: _pil_to_np(_pil_to_pil(_np_to_pil(img), NP_TO_PIL[fmt2]))


class Image:

    def __init__(self, img: Any) -> None:
        """
        Initialize an image object with the provided image.

        Parameters:
            img (Any): The image to be stored.

        Returns:
            None
        """

        self._orig_fmt = _type(img)
        self._orig_img = img

        self.data = {fmt: None for fmt in FORMATS}
        self.data[self._orig_fmt] = self._orig_img

    def as_(self, fmt: Format) -> Any:
        """
        Get the image data of the specified type.

        Parameters:
            fmt (str): It must be one of the following: 'HW', 'HW3', 'HW4', 'L', 'RGB', 'RGBA'.

        Returns:
            Any: The image data of the specified type.
        """

        if self.data[fmt] is not None:
            return self.data[fmt]

        self.data[fmt] = _converter(self._orig_fmt, fmt)(self._orig_img)
        return self.data[fmt]

    def __getitem__(self, fmt: Format) -> Any:
        return self.as_(fmt)

    @functools.cached_property
    def height(self) -> int:
        return self.asformat(HW).shape[0]

    @functools.cached_property
    def width(self) -> int:
        return self.asformat(HW).shape[1]
