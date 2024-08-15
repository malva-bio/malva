"""
This code is adapted from spacemake.
"""

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, vstack, coo_matrix


def nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True):
    """
    Modify the endpoints of a range as needed to avoid singularities.

    This code was adapted from matplotlib.transforms
    "Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved"

    Args
        vmin, vmax: float, the initial endpoints.
        expander: float, default: 0.001, fractional amount by which *vmin* and
                  *vmax* are expanded if the original interval is too small,
                  based on *tiny*.
        tiny : float, default: 1e-15, hreshold for the ratio of the interval to
            the maximum absolute value of its endpoints.
            If the interval is smaller than this, it will be expanded.
            This value should be around 1e-15 or larger;
            otherwise the interval will be approaching the double precision
            resolution limit.
        increasing: bool, default: True, if True, swap *vmin*, *vmax* if *vmin* > *vmax*.
    Returns:
        vmin, vmax: float, endpoints, expanded and/or swapped if necessary.
                    If either input is inf or NaN, or if both inputs are 0 or very
                    close to zero, it returns -*expander*, *expander*.
    """
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)):
        return -expander, expander

    swapped = False
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        swapped = True

    vmin, vmax = map(float, [vmin, vmax])

    maxabsvalue = max(abs(vmin), abs(vmax))
    if maxabsvalue < (1e6 / tiny) * np.finfo(float).tiny:
        vmin = -expander
        vmax = expander

    elif vmax - vmin <= maxabsvalue * tiny:
        if vmax == 0 and vmin == 0:
            vmin = -expander
            vmax = expander
        else:
            vmin -= expander * abs(vmin)
            vmax += expander * abs(vmax)

    if swapped and not increasing:
        vmin, vmax = vmax, vmin
    return vmin, vmax


def create_mesh(width, height, diameter, distance, push_x=0, push_y=0):
    """
    Create a mesh grid of points within a specified area, based on the given parameters.

    Args:
        width: float, the width of the area in which to create the mesh.
        height: float, the height of the area in which to create the mesh.
        diameter: float, the diameter of each mesh point (used to define spacing).
        distance: float, the distance between the centers of the mesh points.
        push_x: float, default: 0, horizontal offset for the starting point of the mesh.
        push_y: float, default: 0, vertical offset for the starting point of the mesh.

    Returns:
        xy: numpy.ndarray, matrix of shape (n_points, 2) with x, y coordinates of the mesh points.
    """
    distance_y = np.sqrt(3) * distance

    x_coord = np.arange(push_x, width + diameter, distance)
    y_coord = np.arange(push_y, height + diameter, distance_y)

    X, Y = np.meshgrid(x_coord, y_coord)
    xy = np.vstack((X.flatten(), Y.flatten())).T

    return xy


def binning_hexagon(x, y, gridsize, extent=None, last_row=False):
    """Bins x,y points into a mesh of gridsize (across x axis, or xy axes).
    Points are assigned to the closest hexagon, without explicitly calculating
    pairwise distances.

    Does not require to create a mesh beforehand, this function handles the
    mesh and nearest neighbor.

    This code was adapted from matplotlib
    "Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved"

    Args
        x, y: numpy.ndarray, x, y spatial coordinates of points
        gridsize: float or tuple, the amount of hexagons in x direction (float);
                  y direction is automatically computed; or specified if gridsize is
                  a tuple.
        extent: tuple, of (x_min, x_max, y_min, y_max)
        last_row: bool, whether a last row is created in mesh or not
    Returns:
        coordinates: numpy.ndarray, a (x_mesh x y_mesh) x 2 matrix, with the coordinates
                     (centres) of each hexagon in the binned mesh
        accumulated: list, contains the indices from the x, y arrays that were binned
                     to each hexagon in the mesh.
    """
    if np.iterable(gridsize):
        nx, ny = gridsize
    else:
        nx = gridsize
        ny = int(nx / np.sqrt(3))
    # Count the number of data in each hexagon
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Will be log()'d if necessary, and then rescaled.
    tx = x
    ty = y

    if extent is not None:
        xmin, xmax, ymin, ymax = extent
    else:
        xmin, xmax = (tx.min(), tx.max()) if len(x) else (0, 1)
        ymin, ymax = (ty.min(), ty.max()) if len(y) else (0, 1)

        # to avoid issues with singular data, expand the min/max pairs
        xmin, xmax = nonsingular(xmin, xmax, expander=0.1)
        ymin, ymax = nonsingular(ymin, ymax, expander=0.1)

    nx1 = nx + 1
    if last_row:
        ny1 = ny + 1
    else:
        ny1 = ny
    nx2 = nx
    ny2 = ny
    n = nx1 * ny1 + nx2 * ny2

    # In the x-direction, the hexagons exactly cover the region from
    # xmin to xmax. Need some padding to avoid roundoff errors.
    padding = 1.0e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny

    # Positions in hexagon index coordinates.
    ix = (tx - xmin) / sx
    iy = (ty - ymin) / sy
    ix1 = np.round(ix).astype(int)
    iy1 = np.round(iy).astype(int)
    ix2 = np.floor(ix).astype(int)
    iy2 = np.floor(iy).astype(int)

    # flat indices, plus one so that out-of-range points go to position 0.
    i1 = np.where((0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1), ix1 * ny1 + iy1 + 1, 0)
    i2 = np.where((0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2), ix2 * ny2 + iy2 + 1, 0)

    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    bdist = d1 < d2

    Cs_at_i1 = [[] for _ in range(1 + nx1 * ny1)]
    Cs_at_i2 = [[] for _ in range(1 + nx2 * ny2)]
    for i in range(len(x)):
        if bdist[i]:
            Cs_at_i1[i1[i]].append(i)
        else:
            Cs_at_i2[i2[i]].append(i)

    accumulated = [acc for Cs_at_i in [Cs_at_i1, Cs_at_i2] for acc in Cs_at_i[1:]]

    coordinates = np.zeros((n, 2), float)
    coordinates[: nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
    coordinates[: nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
    coordinates[nx1 * ny1 :, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
    coordinates[nx1 * ny1 :, 1] = np.tile(np.arange(ny2), nx2) + 0.5
    coordinates[:, 0] *= sx
    coordinates[:, 1] *= sy
    coordinates[:, 0] += xmin
    coordinates[:, 1] += ymin

    return coordinates, accumulated


def aggregate_adata_by_indices(adata, idx_to_aggregate, idx_aggregated, coordinates_aggregated=None):
    """
    Aggregate the data (along the `obs` dimensions) of an AnnData object by specified indices. The size
    of the resulting `obs` has to be less or equal than the size of the input.

    Args:
        adata: AnnData object, to be aggregated.
        idx_to_aggregate: array-like, the indices of observations to aggregate.
        idx_aggregated: array-like, the indices indicating how to aggregate the data.
        coordinates_aggregated: Union[numpy.ndarray, None], (optional) coordinates of the aggregated spots.

    Returns:
        aggregated_adata: AnnData object, the aggregated data, with spatial coordinates updated.
                          The variable names are preserved from the input `adata`, and an additional
                          observation field "n_joined" is included, representing the number of spots
                          that were aggregated into each bin.
    """
    adata = adata[idx_to_aggregate]
    
    n_bins = idx_aggregated.max() + 1
    row_indices = idx_aggregated
    col_indices = np.arange(adata.X.shape[0])
    data = np.ones_like(col_indices)
    
    binning_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(n_bins, adata.X.shape[0])).tocsr()
    aggregated_data = binning_matrix.dot(adata.X)
    
    has_spots = np.unique(idx_aggregated)
    filtered_data = aggregated_data[has_spots]
    
    aggregated_adata = anndata.AnnData(X=filtered_data, obsm={'spatial': coordinates_aggregated})
    aggregated_adata.var_names = adata.var_names
    aggregated_adata.obs["n_joined"] = binning_matrix[has_spots].sum(axis=1)

    return aggregated_adata



def create_meshed_adata(
    adata,
    px_by_um,
    spot_diameter_um=55,
    spot_distance_um=100,
    bead_diameter_um=10,
    mesh_type="circle",
    start_at_minimum=False,
    optimized_binning=True,
):
    """
    Create a meshed AnnData object by binning the spatial data into a grid.

    Args:
        adata: AnnData object, containing spatial data to be meshed.
        px_by_um: float, the pixel-to-micron ratio.
        spot_diameter_um: float, default: 55, the diameter of the spots in microns.
        spot_distance_um: float, default: 100, the distance between the centers of the spots in microns.
        bead_diameter_um: float, default: 10, the diameter of the beads in microns.
        mesh_type: str, default: "circle", the type of mesh to create ("circle" or "hexagon").
        start_at_minimum: bool, default: False, if True, the mesh grid starts at the minimum coordinate.
        optimized_binning: bool, default: True, if True, use optimized binning method for creating the mesh.

    Returns:
        meshed_adata: AnnData object, the meshed data with aggregated observations according to the specified mesh.
                      The spatial coordinates are updated to the mesh grid, and the original data is aggregated.
    """
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances

    if not mesh_type in ["circle", "hexagon"]:
        raise ValueError(f"unrecognised mesh type {mesh_type}")

    if mesh_type == "hexagon":
        spot_diameter_um = np.sqrt(3) * spot_diameter_um
        spot_distance_um = spot_diameter_um

    coords = adata.obsm["spatial"]

    if start_at_minimum:
        top_left_corner = np.min(coords, axis=0)
    else:
        top_left_corner = [0, 0]

    bottom_right_corner = np.max(coords, axis=0)
    width_px = abs(top_left_corner[0] - bottom_right_corner[0])
    height_px = abs(top_left_corner[1] - bottom_right_corner[1])

    # capture area width
    spot_radius_um = spot_diameter_um / 2

    um_by_px = 1.0 / px_by_um
    width_um = um_by_px * width_px
    spot_diameter_px = spot_diameter_um / um_by_px
    spot_radius_px = spot_radius_um / um_by_px
    spot_distance_px = spot_distance_um / um_by_px

    height_um = height_px * um_by_px

    # create meshgrid with one radius push
    xy = create_mesh(
        width_um,
        height_um,
        spot_diameter_um,
        spot_distance_um,
        push_x=spot_radius_um,
        push_y=spot_radius_um,
    )
    # create pushed meshgrid, pushed by
    xy_pushed = create_mesh(
        width_um,
        height_um,
        spot_diameter_um,
        spot_distance_um,
        push_x=spot_radius_um + spot_distance_um / 2,
        push_y=spot_radius_um + np.sqrt(3) * spot_distance_um / 2,
    )

    mesh = np.vstack((xy, xy_pushed))
    mesh_px = mesh / um_by_px
    # add the top_left_corner
    mesh_px = mesh_px + top_left_corner

    # example: the diameter of one visium spot is 55um. if we take any bead, which center
    # falls into that, assuming that a bead is 10um, we would in fact take all beads
    # within 65um diameter. for this reason, the max distance should be 45um/2 between
    # visium spot center and other beads
    max_distance_px = (spot_diameter_um - bead_diameter_um) / um_by_px / 2

    def _create_optimized_hex_mesh_properties(mesh_px):
        _y_values = np.unique(mesh_px[:, 1])
        _x_values = np.unique(mesh_px[:, 0])
        grid_x = len(mesh_px[mesh_px[:, 1] == _y_values[1]])
        grid_y = len(mesh_px[mesh_px[:, 0] == _x_values[1]])

        # Calculating the extent
        if len(mesh_px[mesh_px[:, 0] == _x_values[0]]) == grid_y:
            _offset = np.diff(_y_values[0:2])
            _last_row = False
        else:
            _offset = 0
            _last_row = True

        _extent = (
            mesh_px[:, 0].min(),
            mesh_px[:, 0].max(),
            mesh_px[:, 1].min(),
            mesh_px[:, 1].max() + _offset,
        )

        return grid_x, grid_y, _extent, _last_row

    if mesh_type == "circle":
        if optimized_binning:
            grid_x, grid_y, _extent, _last_row = _create_optimized_hex_mesh_properties(mesh_px)
            mesh_px, accum = binning_hexagon(
                coords[:, 0],
                coords[:, 1],
                gridsize=(grid_x, grid_y),
                extent=_extent,
                last_row=_last_row,
            )

            new_ilocs = [[i] * len(accum[i]) for i in range(len(accum))]
            new_ilocs = np.array([n for new_iloc in new_ilocs for n in new_iloc])
            original_ilocs = np.concatenate(accum).astype(int)

            distance_filter = (
                np.linalg.norm(
                    np.array(coords[original_ilocs]) - np.array(mesh_px[new_ilocs]),
                    axis=1,
                )
                < max_distance_px
            )

            new_ilocs = new_ilocs[distance_filter]
            original_ilocs = original_ilocs[distance_filter]
        else:
            distance_M = euclidean_distances(mesh_px, coords)
            # circle mesh type: we create spots in a hexagonal mesh
            # new_ilocs contains the indices of the columns of the sparse matrix to be created
            # original_ilocs contains the column location of the original adata.X (csr_matrix)
            new_ilocs, original_ilocs = np.nonzero(distance_M < max_distance_px)
    elif mesh_type == "hexagon":
        if optimized_binning:
            grid_x, grid_y, _extent, _last_row = _create_optimized_hex_mesh_properties(mesh_px)
            mesh_px, accum = binning_hexagon(
                coords[:, 0],
                coords[:, 1],
                gridsize=(grid_x, grid_y),
                extent=_extent,
                last_row=_last_row,
            )

            new_ilocs = np.zeros(len(coords), dtype=int)
            for i in range(len(accum)):
                new_ilocs[accum[i]] = i
        else:
            # we simply create a hex mesh, without holes. For each spot, we find the
            # hexagon it belongs to.
            # Not recommended this non-optimized approach; high memory
            # usage if spot distance is low -> O(n^2) complexity!
            # Kept for reproducibility with legacy runs of spacemake
            distance_M = euclidean_distances(mesh_px, coords)
            new_ilocs = np.argmin(distance_M, axis=0)

        original_ilocs = np.arange(new_ilocs.shape[0])
        # we need to sort the new ilocs so that they are in increasing order
        sorted_ix = np.argsort(new_ilocs)
        new_ilocs = new_ilocs[sorted_ix]
        original_ilocs = original_ilocs[sorted_ix]

    joined_coordinates = mesh_px[np.unique(new_ilocs)]

    meshed_adata = aggregate_adata_by_indices(
        adata,
        idx_to_aggregate=original_ilocs,
        idx_aggregated=new_ilocs,
        coordinates_aggregated=joined_coordinates,
    )

    return meshed_adata
