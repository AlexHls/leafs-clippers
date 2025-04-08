import os

import numpy as np
from scipy.io import FortranFile

from leafs_clippers.leafs import leafs as lc


def read_inipos(filename):
    data = {}
    f = FortranFile(filename, "r")

    data["n_bub"] = f.read_ints(np.int32)[0]
    data["r_bub"] = f.read_reals()[0]
    data["x"] = f.read_reals()
    data["y"] = f.read_reals()
    data["z"] = f.read_reals()

    f.close()

    return data


class LeafsXdmf3Writer:
    def __init__(self, snapshot: lc.LeafsSnapshot, subgrid_size: tuple = None) -> None:
        """
        Initialize the XDMF writer for the given snapshot.

        Parameters
        ----------
        snapshot : lc.LeafsSnapshot
            The snapshot to write to XDMF. Has to be a HDF5 snapshot,
            not the legacy Fortran binary format.
        subgrid_size : tuple, optional
            The size of each sub-grid (nx, ny, nz). If None, the entire grid is treated as a single grid.
        """
        self.snapshot = snapshot
        self.filename = os.path.basename(snapshot.filename)
        self.grid_shape = (snapshot.gnx + 1, snapshot.gny + 1, snapshot.gnz + 1)
        self.attribute_shape = (snapshot.gnx, snapshot.gny, snapshot.gnz)
        self.outname = snapshot.basename + ".xdmf"
        self.subgrid_size = subgrid_size or self.grid_shape
        # In case of 2D grid, set the z dimension to 1
        if snapshot.gnz == 1:
            self.subgrid_size = (self.subgrid_size[0], self.subgrid_size[1], 1)

    @property
    def _ignore_keys(self):
        return [
            "time",
            "gnx",
            "gny",
            "gnz",
            "fgx",
            "fgy",
            "fgz",
            "geomx",
            "geomy",
            "geomz",
            "edgez",
            "edgey",
            "edgex",
            "ncells",
            "rad_wd",
            "rad_fl",
            "idx_wd",
            "idx_fl",
            "simulation_type",
        ]

    def _split_grid(self):
        """
        Generate sub-grid offsets and dimensions based on the subgrid_size.
        """
        nx, ny, nz = self.grid_shape
        sx, sy, sz = self.subgrid_size
        for x in range(0, nx - 1, sx):
            for y in range(0, ny - 1, sy):
                for z in range(0, nz - 1, sz):
                    yield (
                        (x, y, z),
                        (
                            min(sx + 1, nx - x),
                            min(sy + 1, ny - y),
                            min(sz + 1, nz - z),
                        ),
                    )

    def _write_header(self, f):
        f.write('<?xml version="1.0" ?>\n')
        f.write('<Xdmf Version="3.0">\n')
        f.write("<Domain>\n")
        # Add CollectionType="Spatial" to the Grid element
        f.write(
            '<Grid Name="3DStructuredGrid" GridType="Collection" CollectionType="Spatial">\n'
        )

    def _write_footer(self, f):
        f.write("</Grid>\n")
        f.write("</Domain>\n")
        f.write("</Xdmf>\n")

    def _write_time(self, f, time: float) -> None:
        f.write(f'<Time Value="{time}" />\n')

    def _write_topology(self, f, grid_shape: tuple, offset: tuple) -> None:
        f.write(
            '<Topology TopologyType="3DRectMesh" Dimensions="{} {} {}">\n'.format(
                grid_shape[2], grid_shape[1], grid_shape[0]
            )
        )
        f.write("</Topology>\n")

    def _write_geometry(
        self, f, grid_shape: tuple, offset: tuple, filename: str
    ) -> None:
        f.write('<Geometry GeometryType="VXVYVZ">\n')
        for dim, edge, start in zip(grid_shape, ["edgex", "edgey", "edgez"], offset):
            f.write('<DataItem ItemType="HyperSlab" Dimensions="{}">\n'.format(dim))
            f.write('<DataItem Dimensions="3 1" NumberType="Int" Format="XML">\n')
            f.write(f"{start} 1 {dim}\n")  # Start, Stride, Count
            f.write("</DataItem>\n")
            f.write(
                '<DataItem Dimensions="{}" NumberType="Float" Precision="8" Format="HDF">\n'.format(
                    dim
                )
            )
            f.write(f"{filename}:/{edge}\n")
            f.write("</DataItem>\n")
            f.write("</DataItem>\n")
        f.write("</Geometry>\n")

    def _write_attribute(
        self, f, label: str, grid_shape: tuple, offset: tuple, filename: str
    ) -> None:
        f.write(f'<Attribute Name="{label}" AttributeType="Scalar" Center="Cell">\n')
        f.write(
            '<DataItem ItemType="HyperSlab" Dimensions="{} {} {}">\n'.format(
                grid_shape[2] - 1, grid_shape[1] - 1, grid_shape[0] - 1
            )
        )
        f.write('<DataItem Dimensions="3 3" NumberType="Int" Format="XML">\n')
        f.write(
            f"{offset[2]} {offset[1]} {offset[0]}\n1 1 1\n{grid_shape[2] - 1} {grid_shape[1] - 1} {grid_shape[0] - 1}\n"
        )  # Start, Stride, Count
        f.write("</DataItem>\n")
        f.write(
            '<DataItem Dimensions="{} {} {}" NumberType="Float" Precision="8" Format="HDF">\n'.format(
                *self.attribute_shape
            )
        )
        f.write(f"{filename}:/{label}\n")
        f.write("</DataItem>\n")
        f.write("</DataItem>\n")
        f.write("</Attribute>\n")

    def write(self) -> str:
        """
        Write the XDMF file for the current snapshot, splitting the grid into sub-grids if specified.
        """
        outname = self.snapshot.basename + ".xdmf"
        with open(outname, "w") as f:
            self._write_header(f)
            self._write_time(f, self.snapshot.time)
            for offset, subgrid_shape in self._split_grid():
                f.write(f'<Grid Name="SubGrid_{offset}" GridType="Uniform">\n')
                self._write_topology(f, subgrid_shape, offset)
                self._write_geometry(f, subgrid_shape, offset, self.filename)
                for label in self.snapshot.keys:
                    if label in self._ignore_keys:
                        continue
                    self._write_attribute(
                        f, label, subgrid_shape, offset, self.filename
                    )
                f.write("</Grid>\n")
            self._write_footer(f)
        return outname
