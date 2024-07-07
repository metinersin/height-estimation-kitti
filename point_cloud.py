'''
This module contains functions for processing and transforming point cloud data, \
    particularly focusing on the conversion between LiDAR and camera coordinate \
    systems. It is designed to work with point cloud data typically used in \
    autonomous vehicle and robotics applications.
'''

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def lidar_to_cam(points_lidar: np.ndarray, velo_to_cam: np.ndarray) -> np.ndarray:
    """
    Convert Lidar coordinates to camera coordinates.

    Parameters:
        points_lidar (np.ndarray): Array of Lidar coordinates with shape (N, 3).
        velo_to_cam (np.ndarray): Transformation matrix from Lidar to camera \
            coordinates with shape (4, 4).

    Returns:
        np.ndarray: Array of camera coordinates with shape (N, 3).
    """

    assert points_lidar.ndim == 2
    assert points_lidar.shape[1] == 3
    assert velo_to_cam.shape == (4, 4)
    assert np.allclose(velo_to_cam[3, :], np.array([0, 0, 0, 1]))

    points_lidar = np.column_stack(
        (points_lidar, np.ones(points_lidar.shape[0])))
    points_cam_coord = points_lidar @ velo_to_cam.T
    points_cam_coord = points_cam_coord[:, :3]
    return points_cam_coord


def cam_to_img(
    points_cam: np.ndarray, r_rect: np.ndarray, p_rect: np.ndarray,
    img_height: int | None = None, img_width: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts camera coordinates to image coordinates.

    Parameters:
        points_cam (np.ndarray): Array of camera coordinates with shape (N, 3).
        r_rect (np.ndarray): Rectification matrix with shape (4, 4).
        p_rect (np.ndarray): Projection matrix with shape (3, 4).
        img_height (int | None, optional): Height of the image. Defaults to None.
        img_width (int | None, optional): Width of the image. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the converted image \
            coordinates (points_img) and a boolean mask (idx) indicating which \
            points are within the image boundaries.
    """
    assert points_cam.ndim == 2
    assert points_cam.shape[1] == 3
    assert r_rect.shape == (4, 4)
    assert p_rect.shape == (3, 4)
    if img_height is not None:
        assert img_height > 0
    if img_width is not None:
        assert img_width > 0

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
    points_lidar: np.ndarray, velo_to_cam: np.ndarray,
    r_rect: np.ndarray, p_rect: np.ndarray,
    img_height: int | None = None, img_width: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts LiDAR coordinates to image coordinates.

    Parameters:
        points_lidar (np.ndarray): Numpy array of LiDAR points with shape (N, 3).
        velo_to_cam (np.ndarray): Transformation matrix from LiDAR to camera \
            coordinates with shape (4, 4).
        r_rect (np.ndarray): Rectification matrix with shape (4, 4).
        p_rect (np.ndarray): Projection matrix with shape (3, 4).
        img_height (int, optional): Height of the image. Defaults to None.
        img_width (int, optional): Width of the image. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the array of image \
            coordinates points_img with shape (N, 2) and the corresponding \
                indices idx with shape (N,).
    """
    assert points_lidar.ndim == 2
    assert points_lidar.shape[1] == 3
    assert velo_to_cam.shape == (4, 4)
    assert np.allclose(velo_to_cam[3, :], np.array([0, 0, 0, 1]))
    assert r_rect.shape == (4, 4)
    assert p_rect.shape == (3, 4)
    if img_height is not None:
        assert img_height > 0
    if img_width is not None:
        assert img_width > 0

    points_cam = lidar_to_cam(points_lidar, velo_to_cam)
    points_img, idx = cam_to_img(
        points_cam, r_rect, p_rect,
        img_height=img_height, img_width=img_width
    )

    return points_img, idx


def normals(
    points: np.ndarray, *, neighbors: int = 30, radius: float | None = 10
) -> np.ndarray:
    """
    Compute the normal vectors of a point cloud.

    Parameters:
        points (np.ndarray): The input point cloud as a numpy array of shape \
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if radius is None:
        search_param = o3d.geometry.KDTreeSearchParamKNN(knn=neighbors)
    else:
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=neighbors)

    pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_consistent_tangent_plane(100)

    return np.asarray(pcd.normals)


def to_field_on_img(
    field_on_lidar: np.ndarray,
    points_lidar: np.ndarray,
    img_height: int,
    img_width: int,
    velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray
) -> np.ndarray:
    """
    Converts a field on points to fields on image based on camera parameters.

    Args:
        field (np.ndarray): The field on points.
        points_lidar (np.ndarray): Lidar points.
        img_height (int): Height of the image.
        img_width (int): Width of the image.
        velo_to_cam (np.ndarray): Transformation matrix from lidar to camera coordinates.
        r_rect (np.ndarray): Rectification matrix.
        p_rect (np.ndarray): Projection matrix.

    Returns:
        np.ndarray: The field on image.

    """

    points_cam = lidar_to_cam(points_lidar, velo_to_cam)
    points_img, idx = cam_to_img(
        points_cam, r_rect, p_rect, img_height=img_height, img_width=img_width
    )
    field_on_img = np.zeros((img_height, img_width)) - 1
    field_on_img[points_img[idx][:, 1],
                 points_img[idx][:, 0]] = field_on_lidar[idx]

    return field_on_img


def scale_field_on_lidar(
    points_lidar: np.ndarray,
    img_height: int,
    img_width: int,
    velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    *,
    scaling: float = 0.3, neighbors: int = 50, radius: float | None = 10
) -> np.ndarray:
    """
    Calculates the scale field for height estimation.

    Parameters:
        points_lidar (np.ndarray): Lidar points in the LiDAR coordinate system.
        img_height (int): Height of the image.
        img_width (int): Width of the image.
        velo_to_cam (np.ndarray): Transformation matrix from LiDAR to camera coordinates.
        r_rect (np.ndarray): Rectification matrix.
        p_rect (np.ndarray): Projection matrix.
        scaling (float, optional): Scaling factor for the normal vectors. Defaults to 0.3.
        neighbors (int, optional): Number of neighbors to consider for normal \
            vector calculation. Defaults to 50.
        radius (float | None, optional): Radius for neighborhood search. Defaults to 10.

    Returns:
        np.ndarray: Scale field for height estimation.
    """

    # project originial points onto img
    points_cam = lidar_to_cam(points_lidar, velo_to_cam)
    points_img, _ = cam_to_img(
        points_cam, r_rect, p_rect, img_height=img_height, img_width=img_width
    )

    # calculate the normal vectors
    normal_vectors = normals(
        points_lidar, neighbors=neighbors, radius=radius
    )
    normal_vectors *= scaling

    # project displaced points onto img
    displaced_points_lidar = points_lidar + normal_vectors
    displaced_points_cam = lidar_to_cam(displaced_points_lidar, velo_to_cam)
    displaced_points_img, _ = cam_to_img(
        displaced_points_cam, r_rect, p_rect, img_height=img_height, img_width=img_width
    )

    # calculate the scale field
    field_on_lidar = np.linalg.norm(
        displaced_points_img - points_img, axis=1) / scaling

    return field_on_lidar


def scale_field_on_img(
    points_lidar: np.ndarray,
    img_height: int,
    img_width: int,
    velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    *,
    scaling: float = 0.3, neighbors: int = 50, radius: float | None = 10
) -> np.ndarray:
    """
    Calculates the scale field for height estimation on the image.

    Parameters:
        points_lidar (np.ndarray): Lidar points in the LiDAR coordinate system.
        img_height (int): Height of the image.
        img_width (int): Width of the image.
        velo_to_cam (np.ndarray): Transformation matrix from LiDAR to camera coordinates.
        r_rect (np.ndarray): Rectification matrix.
        p_rect (np.ndarray): Projection matrix.
        scaling (float, optional): Scaling factor for the normal vectors. Defaults to 0.3.
        neighbors (int, optional): Number of neighbors to consider for normal \
            vector calculation. Defaults to 50.
        radius (float | None, optional): Radius for neighborhood search. Defaults to 10.

    Returns:
        np.ndarray: Scale field for height estimation on the image.
    """

    field_on_lidar = scale_field_on_lidar(
        points_lidar, img_height, img_width,
        velo_to_cam=velo_to_cam, r_rect=r_rect, p_rect=p_rect,
        scaling=scaling, neighbors=neighbors, radius=radius
    )

    field_on_img = to_field_on_img(
        field_on_lidar, points_lidar, img_height, img_width,
        velo_to_cam=velo_to_cam, r_rect=r_rect, p_rect=p_rect
    )

    return field_on_img

# plotting functions


def draw_img(
    img: np.ndarray,
    *,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    output_name: str | None = 'scatter_plot_on_image.png'
) -> None:
    """
    Display an image with optional title, x-label, y-label, and save the image if output_name is provided.

    Parameters:
        img (np.ndarray): The image to be displayed.
        title (str, optional): The title of the image. Defaults to an empty string.
        xlabel (str, optional): The label for the x-axis. Defaults to an empty string.
        ylabel (str, optional): The label for the y-axis. Defaults to an empty string.
        output_name (str | None, optional): The name of the output file to save the image. Defaults to 'scatter_plot_on_image.png'.

    Returns:
        None
    """

    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

    if output_name is not None:
        plt.savefig(output_name, dpi=300, bbox_inches='tight')


def scatter_on_image(
        img: np.ndarray,
        points_img: np.ndarray, *,
        c: np.ndarray | None = None,
        alpha: float = 0.5,
        size: int = 2,
        cmap: str = 'rainbow_r',
        title: str = 'Scatter Plot on Image',
        xlabel: str = '',
        ylabel: str = '',
        output_name: str = 'scatter_plot_on_image.png'
) -> None:
    """
    Scatter data points on an image and save the plot as an image file.

    Parameters:
        img (np.ndarray): The input image as a NumPy array.
        data (np.ndarray): The data points to scatter on the image.
        c (np.ndarray | None, optional): The colors of the data points. If \
            None, default colors will be used. Defaults to None.
        alpha (float, optional): The transparency of the data points. Defaults \
            to 0.5.
        size (int, optional): The size of the data points. Defaults to 2.
        cmap (str, optional): The colormap to use for coloring the data points. \
            Defaults to 'rainbow_r'.
        title (str, optional): The title of the plot. Defaults to 'Scatter Plot \
            on Image'.
        xlabel (str, optional): The label for the x-axis. Defaults to ''.
        ylabel (str, optional): The label for the y-axis. Defaults to ''.
        output_name (str, optional): The name of the output image file. Defaults \
            to 'scatter_plot_on_image.png'.

    Returns:
        None: This function does not return anything. The scatter plot is saved \
            as an image file.

    Raises:
        AssertionError: If the input image or data has invalid dimensions or shapes.
    """
    assert img.ndim in (2, 3)
    assert img.shape[-1] == 3 or img.shape[-1] >= 10

    assert points_img.ndim == 2
    assert points_img.shape[1] == 2

    if c is not None:
        assert c.ndim == 1
        assert c.shape[0] == points_img.shape[0]

    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    u, v = points_img.T
    plt.scatter(u, v, c=c, cmap=cmap, alpha=alpha, s=size)

    img_height = img.shape[0]
    img_width = img.shape[1]
    plt.xlim(0, img_width)
    plt.ylim(img_height, 0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()


def draw_velodyne_on_image(
    img: np.ndarray,
    points_lidar: np.ndarray,
    velo_to_cam: np.ndarray,
    r_rect: np.ndarray,
    p_rect: np.ndarray,
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draws velodyne points on the given image.

    Parameters:
        img (np.ndarray): The input image.
        points_lidar (np.ndarray): The Lidar points in Lidar coordinate system.
        velo_to_cam (np.ndarray): The transformation matrix from Lidar to \
            camera coordinate system.
        r_rect (np.ndarray): The rotation matrix for rectifying the camera \
            coordinate system.
        p_rect (np.ndarray): The projection matrix for rectifying the camera \
            coordinate system.
        **kwargs: Additional keyword arguments for scatter_data_on_image.

    Returns:
        np.ndarray: Points in image coordinate system.
        np.ndarray: The indices of the points that were successfully drawn on \
            the image.
    """

    img_height, img_width = img.shape[:2]

    points_cam = lidar_to_cam(points_lidar, velo_to_cam)
    points_img, idx = cam_to_img(
        points_cam, r_rect, p_rect, img_height=img_height, img_width=img_width
    )
    scatter_on_image(img, points_img[idx], c=points_cam[idx, 2], **kwargs)

    return points_img, idx


def draw_field_on_image(
    field_on_img: np.ndarray,
    img: np.ndarray,
    title: str = 'Scatter Plot on Image',
    xlabel: str = '',
    ylabel: str = '',
    output_name: str = 'scatter_plot_on_image.png'
) -> None:

    # Plot the image
    plt.imshow(img)

    # Overlay the heatmap
    plt.imshow(field_on_img, cmap='Reds', interpolation='nearest', alpha=0.5)

    # Customize as needed (e.g., add labels, colorbar, etc.)
    plt.title(title)
    plt.colorbar(label="Intensity")

    plt.show()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')


# data exploration functions

def in_radius(
    points: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
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


def approximate_diameter(
    points: np.ndarray, sample_size: int | None = 100
) -> float:
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
        idx = np.random.choice(
            points.shape[0], size=sample_size, replace=False)
        sampled_points = points[idx, :]

    distances = np.linalg.norm(
        sampled_points[:, np.newaxis] - sampled_points, axis=2
    )

    return distances.max()


def plot_number_of_points_in_radius(
    points: np.ndarray, radius: np.ndarray, *,
    sample_size: int | None = 100,
    title: str = 'Average Number of Points in Radius vs Radius',
    xlabel: str = 'Radius',
    ylabel: str = 'Average Number of Points in Radius',
    output_name: str = 'average_number_of_points_in_radius.png'
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
        idx = np.random.choice(
            points.shape[0], size=sample_size, replace=False)
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
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.close()

    return num_points_in_radius


def test_scatter_data_on_image(
    img: np.ndarray,
    data: np.ndarray | str = 'random',
    colors: np.ndarray | str | None = 'random'
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
        data = np.random.rand(num_points, 2) * \
            np.array([img_width, img_height])

    if isinstance(colors, str):
        colors = np.random.rand(data.shape[0])  # type: ignore

    scatter_on_image(img, data, c=colors)
