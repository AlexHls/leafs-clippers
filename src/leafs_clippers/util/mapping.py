import numpy as np
from numba import njit, prange


@njit
def _compute_overlap_matrix(src_edges, dst_edges):
    """
    Compute geometric overlap lengths between two rectilinear grids.
    src_edges, dst_edges : 1D arrays of cell edges.
    Returns L[dst, src] = physical overlap length between dst and src cells.
    """
    n_src = len(src_edges) - 1
    n_dst = len(dst_edges) - 1
    L = np.zeros((n_dst, n_src), dtype=float)
    for i_dst in range(n_dst):
        d0, d1 = dst_edges[i_dst], dst_edges[i_dst + 1]
        for i_src in range(n_src):
            s0, s1 = src_edges[i_src], src_edges[i_src + 1]
            overlap = max(0.0, min(d1, s1) - max(d0, s0))
            if overlap > 0:
                L[i_dst, i_src] = overlap
    return L


@njit
def _remap_2d(Lx, Ly, src_data):
    return Lx @ src_data @ Ly.T


@njit(parallel=True)
def _remap_3d(Lx, Ly, Lz, src_data):
    """
    Equivalent to: np.einsum("Ii,ijk,Jj,Kk->IJK", Lx, src_data, Ly, Lz),
    but implemented with Numba parallelization for performance.
    """
    I_size = Lx.shape[0]
    J_size = Ly.shape[0]
    K_size = Lz.shape[0]

    i_size = Lx.shape[1]
    j_size = Ly.shape[1]
    k_size = Lz.shape[1]

    result = np.zeros((I_size, J_size, K_size), dtype=src_data.dtype)

    for I in prange(I_size):
        for J in prange(J_size):
            for K in prange(K_size):
                sum_val = 0.0

                for i in range(i_size):
                    Lx_Ii = Lx[I, i]

                    for j in range(j_size):
                        Ly_Jj = Ly[J, j]

                        product_i_j = Lx_Ii * Ly_Jj

                        for k in range(k_size):
                            term = product_i_j * src_data[i, j, k] * Lz[K, k]
                            sum_val += term

                result[I, J, K] = sum_val

    return result


class ConservativeRemap:
    def __init__(self, src_edges, dst_shape):
        """
        Conservative downsampling from a fine rectilinear grid (2D or 3D)
        to a coarser rectilinear grid, conserving quantities such as
        total mass or energy.

        Parameters
        ----------
        src_edges : tuple of np.ndarray
            Tuple of 1D arrays defining the edges of the source grid in each dimension.
            It is assumed that len(src_edges) = dim
        dst_shape : tuple of int
            Shape of the destination grid.
        """

        self.dim = len(src_edges)
        if self.dim not in (2, 3):
            raise ValueError("Only 2D and 3D grids are supported.")
        if len(dst_shape) != self.dim:
            raise ValueError(
                "dst_shape must have the same number of dimensions as src_edges."
            )

        self.src_edges = src_edges
        self.dst_shape = dst_shape
        self.dst_edges, self.dst_volumes = self._get_dst_edges()

        self.L_mats = [
            _compute_overlap_matrix(src_edges[i], self.dst_edges[i])
            for i in range(self.dim)
        ]

    def _get_dst_edges(self):
        # Compute edges
        dst_edges = np.empty(self.dim, dtype=object)
        for i in range(self.dim):
            x_min, x_max = self.src_edges[i][0], self.src_edges[i][-1]
            dst_edges[i] = np.linspace(x_min, x_max, self.dst_shape[i] + 1)

        # Compute volumes
        dst_deltas = [np.diff(edges) for edges in dst_edges]
        if self.dim == 2:
            dst_volumes = np.outer(dst_deltas[0], dst_deltas[1])
        else:
            dst_volumes = (
                dst_deltas[0][:, None, None]
                * dst_deltas[1][None, :, None]
                * dst_deltas[2][None, None, :]
            )
        return dst_edges, dst_volumes

    def _remap_2d(self, src_data):
        Lx, Ly = self.L_mats
        return _remap_2d(Lx, Ly, src_data)

    def _remap_3d(self, src_data):
        Lx, Ly, Lz = self.L_mats
        return _remap_3d(Lx, Ly, Lz, src_data)

    def remap(self, src_data):
        if src_data.ndim != self.dim:
            raise ValueError(
                "src_data must have the same number of dimensions as src_edges."
            )

        if self.dim == 2:
            total_dst = self._remap_2d(src_data)
        else:
            total_dst = self._remap_3d(src_data)
        return total_dst / self.dst_volumes
