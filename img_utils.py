from PIL import Image
import numpy as np


def np_to_pil(img: np.ndarray) -> Image.Image:
    """
    Convert a PIL image to a Numpy array. 

    Parameters:
        img (np.ndarray): A Numpy array of dtype float and shape (H, W), (H, W, 3) or (H, W, 4).

    Returns:
        img_pil (Image.Image): PIL image. It is grayscale if img is of shape (H, W), RGB if \
            img is of shape (H, W, 3) and RGBA if img is of shape (H, W, 4).
    """

    # [0, 1] float to [0, 255] uint8
    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    else:
        img = img.astype(np.uint8)

    # determine whether the image is grayscale, RGB or RGBA
    img = img.squeeze()

    if img.ndim == 2:
        mode = 'L'
    elif img.ndim == 3:
        if img.shape[2] == 3:
            mode = 'RGB'
        elif img.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise ValueError(
                f'Invalid image shape. Image has shape {img.shape} but it must have shape (H, W, 3) or (H, W, 4).')
    else:
        raise ValueError(
            f'Invalid image shape. Image has shape {img.shape} but it must have shape (H, W) or (H, W, 3) or (H, W, 4).')

    img_pil = Image.fromarray(img, mode=mode).convert(mode=mode)
    return img_pil


def pil_to_np(img_pil: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a Numpy array. 

    Parameters:
        img_pil (Image.Image): Input image. 

    Returns:
        img_np (np.ndarray): A Numpy array of dtype float and shape (H, W) if img_pil is \
            grayscale, (H, W, 3) if img_pil is RGB and (H, W, 4) if img_pil is RGBA.
    """

    img = np.array(img_pil).squeeze()

    if img_pil.mode == 'L':
        if img.ndim != 2:
            raise ValueError(
                f'Invalid image shape. Image has shape {img.shape} but it must have shape (H, W).')

    if img_pil.mode == 'RGB':
        if not (img.ndim == 3 and img.shape[2] == 3):
            raise ValueError(
                f'Invalid image shape. Image has shape {img.shape} but it must have shape (H, W, 3).')

    if img_pil.mode == 'RGBA':
        if not (img.ndim == 3 and img.shape[2] == 4):
            raise ValueError(
                f'Invalid image shape. Image has shape {img.shape} but it mush have shape (H, W, 4).')

    img = img.astype(np.float32) / 255
    return img


def apply_mask(
    img: np.ndarray, mask: np.ndarray,
    *, alpha: float = 0.2, color: tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Apply a mask to an image and return the masked image.

    Parameters:
        img (np.ndarray): Input image as a Numpy array of shape (H, W), (H, W, 3), or (H, W, 4) \
            and dtype float.
        mask (np.ndarray): The mask to be applied to the image as a Numpy array of shape (H, W) \
            and dtype bool.
        alpha (float, optional): Float in the range [0, 1] representing the the transparency \
            level of the mask. Defaults to 0.2.
        color (tuple[int, int, int], optional): A 3-tuple of integers in [0, 255] representing \
            the color of the mask. Defaults to (255, 0, 0).

    Returns:
        masked image (np.ndarray): The resulting image as a Numpy array of shape (H, W, 3) \
            after superimposing mask on to img.
    """

    # convert the img to RGBA if it is not already
    if img.ndim == 2:
        img = img[:, :, np.newaxis].repeat(4, axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = np.concatenate([img, np.ones_like(img[:, :, :1])], axis=2)
    else:
        raise ValueError(
            f'Invalid image shape. Image has shape {img.shape} but it must have shape (H, W, 2) or (H, W, 3).')

    assert img.ndim == 3
    assert img.shape[2] == 4

    # Convert the mask to an alpha channelled image
    if not (mask.shape == img.shape[:2]):
        raise ValueError(
            f'Invalid mask shape. Mask has shape {mask.shape} but it must has the shape {img.shape[:2]}.')

    if mask.dtype != bool:
        raise ValueError('Mask should be a boolean array.')

    mask_alpha = np.zeros_like(img, dtype=np.uint8)
    mask_alpha[:, :, :3] = color
    mask_alpha[mask,  3] = np.uint8(alpha * 255)

    # superimpose the mask on the image
    img_pil = np_to_pil(img)
    mask_pil = np_to_pil(mask_alpha)
    img_masked_pil = Image.alpha_composite(img_pil, mask_pil)
    img_masked_pil = img_masked_pil.convert('RGB')
    img_masked = pil_to_np(img_masked_pil)

    return img_masked


def segment(model, img: np.ndarray, prompt: str) -> np.ndarray:
    """
    Segment an image using a model and a prompt.

    Parameters:
        model: The segmentation model.
        img (np.ndarray): The image to be segmented as a Numpy array of shape (H, W) or (H, W, 3).
        prompt (str): The prompt to be used for segmentation.

    Returns:
        mask (np.ndarray): The mask as a Numpy array of shape (H, W) and of dtype bool.
    """

    # convert the image to PIL
    img_pil = np_to_pil(img).convert('RGB')

    # segment the image
    masks, _, _, _ = model.predict(img_pil, prompt)

    # convert masks to a Numpy array
    masks = masks.numpy()
    assert masks.dtype == bool, masks.dtype

    # combine all the masks
    mask = masks.sum(axis=0).astype(bool)

    return mask
