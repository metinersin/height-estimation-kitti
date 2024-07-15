"""
This module provides visualization utilities for point cloud and image data \
    using matplotlib.

The primary function in this module, `imshow`, wraps around matplotlib's `imshow` \
    function, offering a simplified interface for displaying numpy arrays as \
    images.
"""

import numpy as np
import matplotlib.pyplot as plt


def imshow(
    data: np.ndarray,
    *,
    cmap: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    output_name: str | None = None,
    **kwargs
) -> None:
    """
    Visualize data using matplotlib's `imshow` function with optional title, x-label, \
        y-label, and save the image if output_name is provided.

    Parameters:
        data (np.ndarray): 2D or 3D Numpy array. The image to be displayed. 
        cmap (str | None, optional): The colormap to use for the image. If None, \
            the default colormap is used. Defaults to None.
        title (str, optional): The title of the image. Defaults to an empty string.
        xlabel (str, optional): The label for the x-axis. Defaults to an empty string.
        ylabel (str, optional): The label for the y-axis. Defaults to an empty string.
        output_name (str | None, optional): The name of the output file to \
            save the image. Defaults to None.
        **kwargs: Additional keyword arguments for the matplotlib's `imshow` function.

    Returns:
        None
    """

    assert data.ndim in (2, 3)

    plt.imshow(data, cmap=cmap, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    height = data.shape[0]
    width = data.shape[1]
    plt.xlim(0, width)
    plt.ylim(height, 0)

    if output_name is not None:
        plt.show()
        plt.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close()


def draw(
    img: np.ndarray,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    output_name: str | None = None
) -> None:
    """
    Display an image with optional title, x-label, y-label, and save the image \
        if output_name is provided.

    Parameters:
        img (np.ndarray): The image to be displayed as a Numpy array of shape (H, W) \
            or (H, W, 3) where H is the height and W is the width of the image.
        title (str, optional): The title of the plot. Defaults to an empty string.
        xlabel (str, optional): The label for the x-axis. Defaults to an empty string.
        ylabel (str, optional): The label for the y-axis. Defaults to an empty string.
        output_name (str | None, optional): The name of the output file to save the \
            plot. Defaults to None.

    Returns:
        None
    """

    assert img.ndim in (2, 3)
    assert img.shape[0] > 10
    assert img.shape[1] > 10

    # If the image is a 3D array, it is assumed to be in RGB format.
    cmap = "gray" if img.ndim == 2 else None
    imshow(img, title=title, xlabel=xlabel, ylabel=ylabel, cmap=cmap)

    if output_name is not None:
        plt.show()
        plt.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close()


def scatter_on_image(
    img: np.ndarray,
    points_img: np.ndarray,
    *,
    c: np.ndarray | None = None,
    alpha: float = 0.2,
    size: int = 2,
    cmap: str = "rainbow_r",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    output_name: str | None = None
) -> None:
    """
    Scatter data points on an image and save the plot as an image file.

    Parameters:
        img (np.ndarray): Image as a Numpy array of shape (H, W) \
            or (H, W, 3) where H is the height and W is the width of the image.
        points_img (np.ndarray): The coordinates of the data points with respect \
            to the image plane as a Numpy array of shape (N, 2) where N is the number of points.
        c (np.ndarray | None, optional): Numpy array of shape (N,) denoting the colors of the \
            data points. If None, default colors will be used. Defaults to None.
        alpha (float, optional): The transparency of the data points. Defaults to 0.5.
        size (int, optional): The size of the data points. Defaults to 2.
        cmap (str, optional): The colormap to use for coloring the data points. \
            Defaults to 'rainbow_r'.
        title (str, optional): The title of the plot. Defaults to an empty string.
        xlabel (str, optional): The label for the x-axis. Defaults to an empty string.
        ylabel (str, optional): The label for the y-axis. Defaults to an empty string.
        output_name (str | None, optional): The name of the output image file. Defaults \
            to None.

    Returns:
        None: This function does not return anything. The scatter plot is saved \
            as an image file.
    """

    assert img.ndim in (2, 3)
    assert img.shape[-1] == 3 or img.shape[-1] >= 10
    assert points_img.ndim == 2
    assert points_img.shape[1] == 2
    if c is not None:
        assert c.ndim == 1
        assert c.shape[0] == points_img.shape[0]

    # show the image
    draw(img, title=title, xlabel=xlabel, ylabel=ylabel)

    # scatter the points
    u, v = points_img.T
    plt.scatter(u, v, c=c, cmap=cmap, alpha=alpha, s=size)

    if output_name is not None:
        plt.show()
        plt.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close()


def draw_field_on_image(
    field_on_img: np.ndarray,
    img: np.ndarray,
    *,
    cmap: str = "hot",
    alpha: float = 0.2,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    output_name: str | None = None,
    **kwargs
) -> None:
    """
    Draw the heatmap plot of a field on an image.

    Parameters:
        field_on_img (np.ndarray): The heatmap to overlay on the image as a Numpy \
            array of shape (H, W) where H is the height and W is the \
            width of the image.
        img (np.ndarray): The input image as a Numpy array of shape (H, W) or \
            (H, W, 3) where H is the height and W is the width of the image.
        cmap (str, optional): The colormap to use for the heatmap. Defaults to 'hot'.
        alpha (float, optional): The transparency of the heatmap. Defaults to 0.5.
        title (str, optional): The title of the plot. Defaults to an empty string.
        xlabel (str, optional): The label for the x-axis. Defaults to an empty string.
        ylabel (str, optional): The label for the y-axis. Defaults to an empty string.
        output_name (str | None, optional): The name of the output file to save the \
            plot. Defaults to None.

    Returns:
        None
    """

    assert field_on_img.ndim == 2
    assert field_on_img.shape == img.shape[0:2]
    assert img.ndim in (2, 3)
    assert img.shape[0] > 10
    assert img.shape[1] > 10

    draw(img, title=title, xlabel=xlabel, ylabel=ylabel)

    imshow(field_on_img, cmap=cmap, alpha=alpha, **kwargs)
    plt.colorbar(label="Intensity")

    if output_name is not None:
        plt.show()
        plt.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close()
