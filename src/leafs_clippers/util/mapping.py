import numpy as np
from numba import njit, prange
import itertools


@njit
def _compute_overlap_matrix(src_edges, dst_edges):
    """
    Compute geometric overlap lengths between two rectilinear grids.
    src_edges, dst_edges : 1D arrays of cell edges.
    Returns L[dst, src] = physical overlap length between dst and src cells.
    """
    n_src = len(src_edges) - 1
    n_dst = len(dst_edges) - 1
    L = np.zeros((n_dst, n_src), dtype=np.float64)
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


@njit
def _merge_2d(src_mass, dst_shape, merge):
    DST_H = dst_shape[0]
    DST_W = dst_shape[1]
    MERGE_H = merge[0]
    MERGE_W = merge[1]

    total_dst = np.zeros(dst_shape, dtype=np.float64)
    for W in prange(DST_W):
        for H in prange(DST_H):
            sum_val = 0.0
            for MW in range(MERGE_W):
                for MH in range(MERGE_H):
                    src_mass_row = W * MERGE_H + MH
                    src_mass_col = H * MERGE_W + MW

                    sum_val += src_mass[src_mass_row, src_mass_col]

            total_dst[W, H] = sum_val
    return total_dst


@njit(parallel=True)
def _merge_3d(src_mass, dst_shape, merge):
    DST_X = dst_shape[0]
    DST_Y = dst_shape[1]
    DST_Z = dst_shape[2]
    MERGE_X = merge[0]
    MERGE_Y = merge[1]
    MERGE_Z = merge[2]

    total_dst = np.zeros(dst_shape, dtype=np.float64)
    for Z in prange(DST_Z):
        for Y in prange(DST_Y):
            for X in prange(DST_X):
                sum_val = 0.0
                for MZ in range(MERGE_Z):
                    for MY in range(MERGE_Y):
                        for MX in range(MERGE_X):
                            src_mass_x = X * MERGE_X + MX
                            src_mass_y = Y * MERGE_Y + MY
                            src_mass_z = Z * MERGE_Z + MZ

                            sum_val += src_mass[src_mass_x, src_mass_y, src_mass_z]

                total_dst[X, Y, Z] = sum_val
    return total_dst


class ConservativeRemap:
    def __init__(self, src_edges, dst_shape, method="direct"):
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
        method : str, optional
            If "direct", use direct geometric overlap computation. If "indirect",
            first remap to an intermediate grid, merging cells to reduce computation.
        """

        self.dim = len(src_edges)
        if self.dim not in (2, 3):
            raise ValueError("Only 2D and 3D grids are supported.")
        if len(dst_shape) != self.dim:
            raise ValueError(
                "dst_shape must have the same number of dimensions as src_edges."
            )

        if method not in ("direct", "indirect"):
            raise ValueError('method must be either "direct" or "indirect".')

        if method == "indirect":
            _dst_shape = self._get_intermediate_shape(src_edges, dst_shape)
            self.indirect_mapper = SimpleConservativeRemap(src_edges, _dst_shape)
            self.src_edges = self.indirect_mapper.dst_edges
        else:
            self.indirect_mapper = None
            self.src_edges = src_edges

        self.dst_shape = dst_shape
        self.dst_edges, self.dst_volumes = self._get_dst_edges()

        self.L_mats = [
            _compute_overlap_matrix(self.src_edges[i], self.dst_edges[i])
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

    def _get_intermediate_shape(self, src_edges, dst_shape):
        _dst_shape = []
        for i in range(self.dim):
            src_n = len(src_edges[i]) - 1
            dst_n = src_n
            while dst_n // 2 >= dst_shape[i]:
                dst_n //= 2
            _dst_shape.append(dst_n)
        return tuple(_dst_shape)

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

        if self.indirect_mapper is not None:
            _src_data = self.indirect_mapper.remap(src_data)
        else:
            _src_data = src_data.astype(np.float64)

        if self.dim == 2:
            total_dst = self._remap_2d(_src_data)
        else:
            total_dst = self._remap_3d(_src_data)
        return total_dst / self.dst_volumes


class SimpleConservativeRemap:
    def __init__(self, src_edges, dst_shape):
        """
        A simple conservative remapper that simply merges neighboring cells.
        Input resolution must be an integer multiple of output resolution.

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

        if np.any(
            [(len(src_edges[i]) - 1) % dst_shape[i] != 0 for i in range(self.dim)]
        ):
            print([len(src_edges[i]) - 1 for i in range(self.dim)])
            print(dst_shape)
            raise ValueError(
                "Fine resolution must be an integer multiple of coarse resolution."
            )

        self.src_edges = src_edges
        self.dst_shape = dst_shape
        self.merge = [(len(src_edges[i]) - 1) // dst_shape[i] for i in range(self.dim)]
        self.dst_edges, self.dst_volumes = self._get_dst_edges()
        self.src_volumes = self._get_src_volumes()

    def _get_dst_edges(self):
        # Compute edges
        dst_edges = np.empty(self.dim, dtype=object)
        for i in range(self.dim):
            dst_edges[i] = self.src_edges[i][:: self.merge[i]]

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

    def _get_src_volumes(self):
        if self.dim == 2:
            return (
                np.diff(self.src_edges[0])[:, None]
                * np.diff(self.src_edges[1])[None, :]
            )
        else:
            return (
                np.diff(self.src_edges[0])[:, None, None]
                * np.diff(self.src_edges[1])[None, :, None]
                * np.diff(self.src_edges[2])[None, None, :]
            )

    def remap(self, src_data):
        if src_data.ndim != self.dim:
            raise ValueError(
                "src_data must have the same number of dimensions as src_edges."
            )

        _src_data = src_data.astype(np.float64)

        src_mass = _src_data * self.src_volumes

        if self.dim == 2:
            total_dst = self._remap_2d(src_mass)
        else:
            total_dst = self._remap_3d(src_mass)

        return total_dst / self.dst_volumes

    def _remap_2d(self, src_mass):
        total_dst = _merge_2d(src_mass, self.dst_shape, self.merge)
        return total_dst

    def _remap_3d(self, src_mass):
        total_dst = _merge_3d(src_mass, self.dst_shape, self.merge)

        return total_dst
