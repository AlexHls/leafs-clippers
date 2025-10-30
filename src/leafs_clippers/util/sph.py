import numpy as np
from scipy.spatial import cKDTree
from numba import njit

_H_DIST_MIN_ = 1e-7
_W_MIN_ = 1e-16

########################
### Helper Functions ###
########################


@njit
def _w_m4_scalar(q, h, dim):
    """
    W4 kernel implementation following Monaghan 1992.
    """
    if q < 0.0:
        return 0.0

    if dim == 1:
        sigma = 2.0 / 3.0
    elif dim == 2:
        sigma = 10.0 / (7.0 * np.pi)
    else:
        sigma = 1.0 / np.pi

    if q < 1.0:
        wval = 1.0 - 1.5 * q * q + 0.75 * q * q * q
    elif q < 2.0:
        t = 2.0 - q
        wval = 0.25 * t * t * t
    else:
        wval = 0.0

    return sigma * wval / (h**dim)


@njit
def _deposit_tracer_with_species(
    mass_field, species_field, mesh_idx, r_arr, tracer_mass, tracer_x_row, h, dim, eps
):
    n = mesh_idx.shape[0]
    if n == 0:
        return

    wsum = 0.0
    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        q = r_arr[i] / h
        wi = _w_m4_scalar(q, h, dim)
        w[i] = wi
        wsum += wi

    if wsum <= eps:
        return

    inv_sum = 1.0 / wsum
    s = tracer_x_row.shape[0]
    for i in range(n):
        idx = mesh_idx[i]
        factor = tracer_mass * w[i] * inv_sum
        mass_field[idx] += factor
        for j in range(s):
            species_field[idx, j] += factor * tracer_x_row[j]


def _compute_adaptive_h(tracer_pos, n_ngb=32, factor=0.5):
    tree = cKDTree(tracer_pos)
    k = min(len(tracer_pos), n_ngb + 1)
    dists, _ = tree.query(tracer_pos, k=k)

    if dists.ndim == 1:
        rn = dists
    else:
        rn = dists[:, -1]
    if np.any(rn <= 0.0):
        positive = rn[rn > 0.0]
        if positive.size > 0:
            rn[rn <= 0.0] = np.mean(positive)
        else:
            rn[rn <= 0.0] = _H_DIST_MIN_

    return factor * rn


#########################
### Mapping functions ###
#########################


def deposit_to_mesh_adaptive_numba(
    tracer_pos, tracer_mass, mesh_centers, tracer_x, n_ngb=32, eps=_W_MIN_
):
    """
    Deposit tracer particles onto a mesh using adaptive smoothing lengths.

    Parameters
    ----------
    tracer_pos : ndarray
        Array of shape (N, D) containing the positions of the tracer particles.
    tracer_mass : float
        Mass of each tracer particle.
    mesh_centers : ndarray
        Array of shape (M, D) containing the centers of the mesh cells.
    tracer_x : ndarray
        Array of shape (N, S) containing species information for each tracer particle.
    n_ngb : int
        Number of neighbors to consider for adaptive smoothing length.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    mass_field : ndarray
        Array of shape (M,) containing the deposited mass in each mesh cell.
    species_field : ndarray
        Array of shape (M, S) containing the deposited species in each mesh cell.
    h_i : ndarray
        Array of shape (N,) containing the adaptive smoothing lengths for each tracer particle.

    """

    n_tracers = tracer_pos.shape[0]
    n_cells = mesh_centers.shape[0]
    dim = tracer_pos.shape[1]

    h_i = _compute_adaptive_h(tracer_pos, n_ngb=n_ngb)
    mesh_tree = cKDTree(mesh_centers)

    s = tracer_x.shape[1]
    mass_field = np.zeros(n_cells, dtype=np.float64)
    species_field = np.zeros((n_cells, s), dtype=np.float64)

    for i in range(n_tracers):
        hi = float(h_i[i])
        idx_list = mesh_tree.query_ball_point(tracer_pos[i], hi * 2.0, p=2.0)
        if len(idx_list) == 0:
            continue
        idx_arr = np.array(idx_list, dtype=np.int64)
        r_arr = np.linalg.norm(mesh_centers[idx_arr] - tracer_pos[i], axis=1).astype(
            np.float64
        )
        _deposit_tracer_with_species(
            mass_field,
            species_field,
            idx_arr,
            r_arr,
            tracer_mass[i].astype(np.float64),
            tracer_x[i].astype(np.float64),
            hi,
            dim,
            eps,
        )

        return mass_field, species_field, h_i


def conservative_remap_to_mesh_with_centers(
    mesh_centers, mass_field, species_field, mesh_rho, cell_volume, eps=_W_MIN_
):
    """
    Perform conservative remapping of species fields to ensure mass conservation.

    Parameters
    ----------
    mesh_centers : ndarray
        Array of shape (M, D) containing the centers of the mesh cells.
    mass_field : ndarray
        Array of shape (M,) containing the deposited mass in each mesh cell.
    species_field : ndarray
        Array of shape (M, S) containing the deposited species in each mesh cell.
    mesh_rho : ndarray
        Array of shape (M,) containing the density of each mesh cell.
    cell_volume : ndarray
        Array of shape (M,) containing the volume of each mesh cell.
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    mass_final : ndarray
        Array of shape (M,) containing the final mass in each mesh cell after remapping.
    species_final : ndarray
        Array of shape (M, S) containing the final species in each mesh cell after remapping.
    x_final : ndarray
        Array of shape (M, S) containing the final species fractions in each mesh cell after remapping.

    """

    n_cells = mass_field.shape[0]
    mesh_mass = mesh_rho * cell_volume
    mass_final = np.zeros_like(mass_field)
    species_final = np.zeros_like(species_field)

    mask = mass_field > eps
    scale_factors = np.zeros(n_cells, dtype=np.float64)
    scale_factors[mask] = mesh_mass[mask] / mass_field[mask]
    mass_final[mask] = mesh_mass[mask]
    species_final[mask, :] = species_field[mask, :] * scale_factors[mask][:, None]

    if not np.all(mask):
        nonempty_idx = np.where(mask)[0]
        empty_idx = np.where(~mask)[0]
        if nonempty_idx.size > 0:
            tree = cKDTree(mesh_centers[nonempty_idx])
            nearest = tree.query(mesh_centers[empty_idx], k=1)[1]
            x_nonempty = (
                species_final[nonempty_idx, :] / mass_final[nonempty_idx][:, None]
            )
            x_copy = x_nonempty[nearest, :]
            species_final[empty_idx, :] = x_copy * mesh_mass[empty_idx][:, None]
            mass_final[empty_idx] = mesh_mass[empty_idx]

    x_final = species_final / mass_final[:, None]

    return mass_final, species_final, x_final
