"""
This module contains functions for processing and transforming point cloud data, \
    particularly focusing on the conversion between lidar and camera coordinate \
    systems. It is designed to work with point cloud data typically used in \
    autonomous vehicle and robotics applications.
"""

from typing import Literal
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import plotting


def _check_img_size(img_size: tuple[int, int]) -> None:
    """
    Check the shape of the input img_size.

    Parameters:
        img_size (tuple[int, int]): Size of the image as a tuple of two integers.

    Returns:
        None
    """

    if img_size[0] <= 0 or img_size[1] <= 0:
        raise ValueError(
            f"Invalid value for `img_size`. Both values must be greater than 0 but it is {img_size}."
        )


def _check_img(img: np.ndarray) -> None:
    """
    Check the shape of the input img.

    Parameters:
        img (np.ndarray): Image as a Numpy array of shape (H, W) or (H, W, 3) where H is the height and W is the width \
            of the image.

    Returns:
        None
    """

    if img.ndim not in (2, 3):
        raise ValueError(
            f"Invalid shape for `img`. It must be of shape (H, W) or (H, W, 3) but it is of shape {img.shape}."
        )
    if img.shape[0] <= 10 or img.shape[1] <= 10:
        raise ValueError(
            f"Invalid shape for `img`. Both dimensions must be greater than 10 but it is {img.shape}."
        )


def _check_points_lidar(points_lidar: np.ndarray) -> None:
    """
    Check the shape of the input points_lidar.

    Parameters:
        points_lidar (np.ndarray): Coordinates of the velodyne points in lidar coordinates as a Numpy array of shape \
            (N, 3).

    Returns:
        None
    """

    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError(
            f"Invalid shape for `points_lidar`. It must be of shape (N, 3) but it is of shape {points_lidar.shape}."
        )


def _check_points_cam(points_cam: np.ndarray) -> None:
    """
    Check the shape of the input points_cam.

    Parameters:
        points_cam (np.ndarray): Coordinates of the velodyne points in camera coordinates as a Numpy array of shape \
            (N, 3).

    Returns:
        None
    """

    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(
            f"Invalid shape for `points_cam`. It must be of shape (N, 3) but it is of shape {points_cam.shape}."
        )


def _check_points_img(points_img: np.ndarray) -> None:
    """
    Check the shape of the input points_img.

    Parameters:
        points_img (np.ndarray): Coordinates of the velodyne points in image coordinates as a Numpy array of shape \
            (N, 2).

    Returns:
        None
    """

    if points_img.ndim != 2 or points_img.shape[1] != 2:
        raise ValueError(
            f"Invalid shape for `points_img`. It must be of shape (N, 2) but it is of shape {points_img.shape}."
        )


def _check_velo_to_cam(velo_to_cam: np.ndarray) -> None:
    """
    Check the shape of the input velo_to_cam.

    Parameters:
        velo_to_cam (np.ndarray): Transformation matrix from lidar to camera coordinates as a Numpy array of shape \
            (4, 4).

    Returns:
        None
    """

    if velo_to_cam.shape != (4, 4):
        raise ValueError(
            f"Invalid shape for `velo_to_cam`. It must be of shape (4, 4) but it is of shape {velo_to_cam.shape}."
        )
    if not np.allclose(velo_to_cam[3, :], np.array([0, 0, 0, 1])):
        raise ValueError(
            f"Invalid value for `velo_to_cam`. The last row must be [0, 0, 0, 1] but it is {velo_to_cam[3, :]}."
        )


def _check_r_rect(r_rect: np.ndarray) -> None:
    """
    Check the shape of the input r_rect.

    Parameters:
        r_rect (np.ndarray): Rectification matrix as a Numpy array of shape (4, 4).

    Returns:
        None
    """

    if r_rect.shape != (4, 4):
        raise ValueError(
            f"Invalid shape for `r_rect`. It must be of shape (4, 4) but it is of shape {r_rect.shape}."
        )


def _check_p_rect(p_rect: np.ndarray) -> None:
    """
    Check the shape of the input p_rect.

    Parameters:
        p_rect (np.ndarray): Projection matrix as a Numpy array of shape (3, 4).

    Returns:
        None
    """

    if p_rect.shape != (3, 4):
        raise ValueError(
            f"Invalid shape for `p_rect`. It must be of shape (3, 4) but it is of shape {p_rect.shape}."
        )


def lidar_to_cam(points_lidar: np.ndarray, *, velo_to_cam: np.ndarray) -> np.ndarray:
    """
    Convert lidar coordinates to camera coordinates.

    Parameters:
        points_lidar (np.ndarray): Array of lidar coordinates with shape (N, 3).
        velo_to_cam (np.ndarray): Transformation matrix from lidar to camera \
            coordinates with shape (4, 4).

    Returns:
        np.ndarray: Array of camera coordinates with shape (N, 3).
    """

    # check the input
    _check_points_lidar(points_lidar)
    _check_velo_to_cam(velo_to_cam)

    points_lidar = np.column_stack((points_lidar, np.ones(points_lidar.shape[0])))
    points_cam_coord = points_lidar @ velo_to_cam.T
    points_cam_coord = points_cam_coord[:, :3]
    return points_cam_coord


def cam_to_img(
    points_cam: np.ndarray,
    *,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    img_height: int | None = None,
    img_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts camera coordinates to image coordinates.

    Parameters:
        points_cam (np.ndarray): Array of camera coordinates with shape (N, 3).
        r_rect (np.ndarray): Rectification matrix with shape (4, 4).
        p_rect (np.ndarray): Projection matrix with shape (3, 4).
        img_height (int, optional): Height of the image. Defaults to None.
        img_width (int, optional): Width of the image. Defaults to None.

    Returns:
        points_img (np.ndarray): Array of image coordinates with shape (N, 2).
        idx (np.ndarray): Boolean mask indicating which points are within the image boundaries.
    """
    # check the input
    _check_points_cam(points_cam)
    _check_r_rect(r_rect)
    _check_p_rect(p_rect)

    if img_height is not None:
        if img_height <= 0:
            raise ValueError(
                f"Invalid value for `img_height`. It must be greater than 0 but it is {img_height}."
            )
    if img_width is not None:
        if img_width <= 0:
            raise ValueError(
                f"Invalid value for `img_width`. It must be greater than 0 but it is {img_width}."
            )

    points_cam = np.column_stack((points_cam, np.ones(points_cam.shape[0])))

    points_img = (points_cam @ r_rect.T) @ p_rect.T
    points_img = points_img / points_img[:, 2, np.newaxis]
    points_img = points_img[:, :2]
    points_img = points_img.round()
    points_img = points_img.astype(int)

    idx = points_cam[:, 2] >= 0
    idx = np.logical_and(idx, points_img[:, 0] >= 0)
    idx = np.logical_and(idx, points_img[:, 1] >= 0)
    if img_width is not None:
        idx = np.logical_and(idx, points_img[:, 0] < img_width)
    if img_height is not None:
        idx = np.logical_and(idx, points_img[:, 1] < img_height)

    return points_img, idx


def lidar_to_img(
    points_lidar: np.ndarray,
    *,
    velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    img_height: int | None = None,
    img_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts lidar coordinates to image coordinates.

    Parameters:
        points_lidar (np.ndarray): Velodyne points in lidar coordinates as a Numpy array of shape (N, 3).
        velo_to_cam (np.ndarray): Transformation matrix from lidar to camera \
            coordinates with shape (4, 4).
        r_rect (np.ndarray): Rectification matrix with shape (4, 4).
        p_rect (np.ndarray): Projection matrix with shape (3, 4).
        img_height (int, optional): Height of the image. Defaults to None.
        img_width (int, optional): Width of the image. Defaults to None.

    Returns:
        points_img (np.ndarray): Array of image coordinates with shape (N, 2).
        idx (np.ndarray): Boolean mask indicating which points are within the image boundaries.
    """
    # check the input
    _check_points_lidar(points_lidar)
    _check_velo_to_cam(velo_to_cam)
    _check_r_rect(r_rect)
    _check_p_rect(p_rect)

    if img_height is not None:
        if img_height <= 0:
            raise ValueError(
                f"Invalid value for `img_height`. It must be greater than 0 but it is {img_height}."
            )
    if img_width is not None:
        if img_width <= 0:
            raise ValueError(
                f"Invalid value for `img_width`. It must be greater than 0 but it is {img_width}."
            )

    points_cam = lidar_to_cam(points_lidar, velo_to_cam=velo_to_cam)
    points_img, idx = cam_to_img(
        points_cam,
        r_rect=r_rect,
        p_rect=p_rect,
        img_height=img_height,
        img_width=img_width,
    )

    return points_img, idx


def normals(
    points_lidar: np.ndarray, *, neighbors: int = 30, radius: float | None = 10
) -> np.ndarray:
    """
    Compute the normal vectors of a point cloud.

    Parameters:
        points_lidar (np.ndarray): The input point cloud as a numpy array of shape \
            (N, 3), where N is the number of points and each point is \
            represented by its (x, y, z) coordinates.
        neighbors (int, optional): The number of nearest neighbors to consider \
            when estimating normals. Defaults to 30.
        radius (float | None, optional): The radius within which to search for \
            neighbors when estimating normals. If set to None, the search is \
            performed based on the number of neighbors. Defaults to 0.1.

    Returns:
        np.ndarray: The computed normal vectors as a numpy array of shape \
            (N, 3), where N is the number of points and each normal vector is \
            represented by its (nx, ny, nz) components.
    """

    # check the input
    _check_points_lidar(points_lidar)

    if neighbors <= 0:
        raise ValueError(
            f"Invalid value for `neighbors`. It must be greater than 0 but it is {neighbors}."
        )

    if radius is not None:
        if radius <= 0:
            raise ValueError(
                f"Invalid value for `radius`. It must be positive but it is {radius}."
            )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_lidar)

    if radius is None:
        search_param = o3d.geometry.KDTreeSearchParamKNN(knn=neighbors)
    else:
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=neighbors
        )

    pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_consistent_tangent_plane(100)

    return np.asarray(pcd.normals)


def to_field_on_img(
    field_on_lidar: np.ndarray,
    points_lidar: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    interpolation: Literal["linear", "nearest", "cubic"] | None = None,
    img_size: tuple[int, int],
    velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
) -> np.ndarray:
    """
    Convert a field defined on lidar coordinates to a field defined on image coordinates.

    Args:
        field_on_lidar (np.ndarray): Field defined on lidar coordinates as a Numpy array \
            of shape (N,)
        points_lidar (np.ndarray): lidar points as a Numpy array of shape (N, 3)
        mask (np.ndarray, optional): Mask to ignore points outside the mask as a Numpy array \
            of shape `img_size`. Defaults to None.
        interpolation (None | linear | nearest | cubic): Interpolation method. If None, no \
            interpolation occurs. Defaults to None.
        img_size (tuple[int, int]): Size of the image.
        velo_to_cam (np.ndarray): Transformation matrix from lidar to camera coordinates.
        r_rect (np.ndarray): Rectification matrix.
        p_rect (np.ndarray): Projection matrix.

    Returns:
        np.ndarray: Field defined on image coordinates as a Numpy array of shape `img_size'
    """
    # check the input
    _check_img_size(img_size)
    _check_points_lidar(points_lidar)
    _check_velo_to_cam(velo_to_cam)
    _check_r_rect(r_rect)
    _check_p_rect(p_rect)

    if mask is not None:
        if mask.shape != img_size:
            raise ValueError(
                f"Invalid shape for `mask`. It must be of shape {img_size} but it is of shape {mask.shape}."
            )

    # convert from lidar coordinates to camera coordinates
    points_img, idx = lidar_to_img(
        points_lidar,
        velo_to_cam=velo_to_cam,
        r_rect=r_rect,
        p_rect=p_rect,
        img_height=img_size[0],
        img_width=img_size[1],
    )

    # interpolate
    if interpolation is None:
        field_on_img = np.full((img_size[0], img_size[1]), np.nan)
        field_on_img[points_img[idx][:, 1], points_img[idx][:, 0]] = field_on_lidar[idx]
    else:
        grid_x, grid_y = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))

        field_on_img = scipy.interpolate.griddata(
            points_img[idx], field_on_lidar[idx], (grid_x, grid_y), method=interpolation
        )

        interp = scipy.interpolate.NearestNDInterpolator(
            points_img[idx], field_on_lidar[idx]
        )
        field_on_img = interp(grid_x, grid_y)

    # ignore the outside of the mask
    if mask is not None:
        field_on_img = np.where(mask, field_on_img, np.nan)

    return field_on_img


def scale_field_on_lidar(
    points_lidar: np.ndarray,
    *,
    img_size: tuple[int, int],
    velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    scaling: float = 0.3,
    neighbors: int = 50,
    radius: float | None = 10,
) -> np.ndarray:
    """
    Calculates the scale field for height estimation.

    Parameters:
        points_lidar (np.ndarray): Velodyne points in lidar coordinates as a Numpy array of shape (N, 3).
        img_size (tuple[int, int]): Size of the image.
        velo_to_cam (np.ndarray): Transformation matrix from lidar to camera \
            coordinates with shape (4, 4).
        r_rect (np.ndarray): Rectification matrix with shape (4, 4).
        p_rect (np.ndarray): Projection matrix with shape (3, 4).
        scaling (float, optional): Scaling factor for the normal vectors. Defaults to 0.3.
        neighbors (int, optional): Number of neighbors to consider for normal \
            vector calculation. Defaults to 50.
        radius (float | None, optional): Radius for neighborhood search. Defaults to 10.

    Returns:
        np.ndarray: Scale field for height estimation.
    """

    # check the input
    _check_img_size(img_size)
    _check_points_lidar(points_lidar)
    _check_velo_to_cam(velo_to_cam)
    _check_r_rect(r_rect)
    _check_p_rect(p_rect)

    if scaling == 0:
        raise ValueError(
            f"Invalid value for `scaling`. It must be non-zero but it is {scaling}."
        )

    if neighbors <= 0:
        raise ValueError(
            f"Invalid value for `neighbors`. It must be greater than 0 but it is {neighbors}."
        )

    if radius is not None:
        if radius <= 0:
            raise ValueError(
                f"Invalid value for `radius`. It must be positive but it is {radius}."
            )

    # change from lidar coordinates to camera coordinates and project points on to image plane
    points_img, _ = lidar_to_img(
        points_lidar,
        velo_to_cam=velo_to_cam,
        r_rect=r_rect,
        p_rect=p_rect,
        img_height=img_size[0],
        img_width=img_size[1],
    )

    # calculate the normal vectors
    normal_vectors = normals(points_lidar, neighbors=neighbors, radius=radius)
    normal_vectors *= scaling

    # change from lidar coordinates to camera coordinates and project the displaced points on to image plane
    displaced_points_lidar = points_lidar + normal_vectors
    displaced_points_img, _ = lidar_to_img(
        displaced_points_lidar,
        velo_to_cam=velo_to_cam,
        r_rect=r_rect,
        p_rect=p_rect,
        img_height=img_size[0],
        img_width=img_size[1],
    )

    # calculate the scale field on lidar
    field_on_lidar = np.linalg.norm(displaced_points_img - points_img, axis=1) / scaling

    return field_on_lidar


def scale_field_on_img(
    points_lidar: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    interpolation: Literal["linear", "nearest", "cubic"] | None = None,
    img_size: tuple[int, int],
    velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    scaling: float = 0.3,
    neighbors: int = 50,
    radius: float | None = 10,
) -> np.ndarray:
    """
    Calculates the scale field for height estimation on the image.

    Parameters:
        points_lidar (np.ndarray): Velodyne points in lidar coordinates as a Numpy array of shape \
            (N, 3).
        img_height (int): Height of the image.
        img_width (int): Width of the image.
        velo_to_cam (np.ndarray): Transformation matrix from lidar to camera coordinates with shape \
            (4, 4).
        r_rect (np.ndarray): Rectification matrix with shape (4, 4).
        p_rect (np.ndarray): Projection matrix with shape (3, 4).
        scaling (float, optional): Scaling factor for the normal vectors. Defaults to 0.3.
        neighbors (int, optional): Number of neighbors to consider for normal \
            vector calculation. Defaults to 50.
        radius (float | None, optional): Radius for neighborhood search. Defaults to 10.

    Returns:
        np.ndarray: Scale field for height estimation on the image.
    """
    # check the input
    _check_img_size(img_size)
    _check_points_lidar(points_lidar)
    _check_velo_to_cam(velo_to_cam)
    _check_r_rect(r_rect)
    _check_p_rect(p_rect)

    if scaling == 0:
        raise ValueError(
            f"Invalid value for `scaling`. It must be non-zero but it is {scaling}."
        )

    if neighbors <= 0:
        raise ValueError(
            f"Invalid value for `neighbors`. It must be greater than 0 but it is {neighbors}."
        )

    if radius is not None:
        if radius <= 0:
            raise ValueError(
                f"Invalid value for `radius`. It must be positive but it is {radius}."
            )

    field_on_lidar = scale_field_on_lidar(
        points_lidar,
        img_size=img_size,
        velo_to_cam=velo_to_cam,
        r_rect=r_rect,
        p_rect=p_rect,
        scaling=scaling,
        neighbors=neighbors,
        radius=radius,
    )

    field_on_img = to_field_on_img(
        field_on_lidar,
        points_lidar,
        mask=mask,
        interpolation=interpolation,
        img_size=img_size,
        velo_to_cam=velo_to_cam,
        r_rect=r_rect,
        p_rect=p_rect,
    )

    return field_on_img


# plotting functions


def draw_velodyne_on_image(
    img: np.ndarray,
    points_lidar: np.ndarray,
    *,
    velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project velodyne points on to image plane.

    Parameters:
        img (np.ndarray): Image as a Numpy array of shape (H, W) \
            or (H, W, 3) where H is the height and W is the width of the image.
        points_lidar (np.ndarray): The coordinates of the velodyne points with respect \
            to lidar as a Numpy array of shape (N, 3) where N is the number of points. 
        velo_to_cam (np.ndarray): The transformation matrix from lidar to camera coordinates \
            as a Numpy array of shape (4, 4).
        r_rect (np.ndarray): The rotation matrix for rectifying the camera coordinates \
            as a Numpy array of shape (4, 4).
        p_rect (np.ndarray): The projection matrix for rectifying the camera coordinates \
            as a Numpy array of shape (3, 4).
        **kwargs: Additional keyword arguments for scatter_data_on_image function.

    Returns:
        np.ndarray: Coordinates of the points with respect to image plane coordinates.
        np.ndarray: The indices of the points that were successfully projected on \
            to the image plane.
    """
    # check the input
    _check_img(img)
    _check_points_lidar(points_lidar)
    _check_velo_to_cam(velo_to_cam)
    _check_r_rect(r_rect)
    _check_p_rect(p_rect)

    img_height, img_width = img.shape[:2]

    # convert from lidar coordinates to camera coordinates
    points_cam = lidar_to_cam(points_lidar, velo_to_cam=velo_to_cam)

    # project the points on to the image plane
    points_img, idx = cam_to_img(
        points_cam,
        r_rect=r_rect,
        p_rect=p_rect,
        img_height=img_height,
        img_width=img_width,
    )

    plotting.scatter_on_image(img, points_img[idx], c=points_cam[idx, 2], **kwargs)

    return points_img, idx


# data exploration functions


def in_radius(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Returns a boolean array indicating whether each point in the given array
    is inside the specified radius from the center point.

    Parameters:
        points (np.ndarray): Array of points with shape (N, D), where N is the
            number of points and D is the number of dimensions.
        center (np.ndarray): Center point with shape (D,) specifying the
            coordinates of the center.
        radius (float): Radius value specifying the maximum distance allowed
            from the center point.

    Returns:
        np.ndarray: Boolean array with shape (N,) indicating whether each point
        is inside the specified radius.
    """

    displacements = points - center
    distances = np.linalg.norm(displacements, axis=1)
    points_inside = distances <= radius
    return points_inside


def approximate_diameter(points: np.ndarray, sample_size: int | None = 100) -> float:
    """
    Calculate the diameter of a point cloud by random sampling.

    Parameters:
        points (numpy.ndarray): The array of points in the point cloud.
        sample_size (int | None): Number of points to sample for approximation. \
            If set to None, all of the points are considered.

    Returns:
        float: The approximate maximum distance between any two points in the \
            point cloud.
    """

    assert points.ndim == 2
    assert points.shape[1] == 3
    if sample_size is not None:
        assert sample_size > 0

    if sample_size is None:
        sampled_points = points
    else:
        sample_size = min(sample_size, points.shape[0])
        idx = np.random.choice(points.shape[0], size=sample_size, replace=False)
        sampled_points = points[idx, :]

    distances = np.linalg.norm(sampled_points[:, np.newaxis] - sampled_points, axis=2)

    return distances.max()


def plot_number_of_points_in_radius(
    points: np.ndarray,
    radius: np.ndarray,
    *,
    sample_size: int | None = 100,
    title: str = "Average Number of Points in Radius vs Radius",
    xlabel: str = "Radius",
    ylabel: str = "Average Number of Points in Radius",
    output_name: str = "average_number_of_points_in_radius.png",
) -> None:
    """
    Plots the average number of points in a given radius.

    Parameters:
        points (np.ndarray): The array of points with shape (N, 3).
        radius (np.ndarray): The array of radii.
        sample_size (int | None, optional): The number of points to sample. \
            Defaults to 100. If set to None, all points are considered.
        title (str, optional): The title of the plot. Defaults to 'Average \
            Number of Points in Radius vs Radius'.
        xlabel (str, optional): The label for the x-axis. Defaults to 'Radius'.
        ylabel (str, optional): The label for the y-axis. Defaults to 'Average \
            Number of Points in Radius'.
        output_name (str, optional): The name of the output file. Defaults to \
            'average_number_of_points_in_radius.png'.

    Returns:
        None
    """
    assert points.ndim == 2
    assert points.shape[1] == 3
    if sample_size is not None:
        assert sample_size > 0

    if sample_size is None:
        sampled_points = points
    else:
        sample_size = min(sample_size, points.shape[0])
        idx = np.random.choice(points.shape[0], size=sample_size, replace=False)
        sampled_points = points[idx, :]

    distances = np.linalg.norm(
        sampled_points[:, np.newaxis, :] - sampled_points, axis=2
    )

    idx = distances[:, :, np.newaxis] <= radius

    num_points_in_radius = idx.sum(axis=1).mean(axis=0)

    plt.plot(radius, num_points_in_radius)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close()

    return num_points_in_radius


def test_scatter_data_on_image(
    img: np.ndarray,
    data: np.ndarray | str = "random",
    colors: np.ndarray | str | None = "random",
) -> None:
    """
    Scatter data points on an image.

    Parameters:
        img (np.ndarray): The input image.
        data (np.ndarray | str): The data points to scatter. If 'random', \
            random data points will be generated.
        colors (np.ndarray | str | None): The colors of the data points. If \
            'random', random colors will be generated. If None, default colors \
            will be used.

    Returns:
        None
    """

    img_width = img.shape[1]
    img_height = img.shape[0]

    if isinstance(data, str):
        num_points = 50
        data = np.random.rand(num_points, 2) * np.array([img_width, img_height])

    if isinstance(colors, str):
        colors = np.random.rand(data.shape[0])  # type: ignore

    plotting.scatter_on_image(img, data, c=colors)
