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
    def __init__(self, snapshot: lc.LeafsSnapshot) -> None:
        """
        Initialize the XDMF writer for the given snapshot.

        Parameters
        ----------
        snapshot : lc.LeafsSnapshot
            The snapshot to write to XDMF. Has to be a HDF5 snapshot,
            not the lecacy Fortran binary format.
        """
        self.snapshot = snapshot
        self.filename = os.path.basename(snapshot.filename)
        self.grid_shape = (snapshot.gnx + 1, snapshot.gny + 1, snapshot.gnz + 1)
        self.attribute_shape = (snapshot.gnx, snapshot.gny, snapshot.gnz)
        self.outname = snapshot.basename + ".xdmf"

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

    def _write_header(self, f):
        f.write('<?xml version="1.0" ?>\n')
        f.write('<Xdmf Version="3.0">\n')
        f.write("<Domain>\n")
        f.write('<Grid Name="3DStructuredGrid" GridType="Uniform">\n')

    def _write_footer(self, f):
        f.write("</Grid>\n")
        f.write("</Domain>\n")
        f.write("</Xdmf>\n")

    def _write_time(self, f, time: float) -> None:
        f.write(f'<Time Value="{time}" />\n')

    def _write_topology(self, f, grid_shape: tuple) -> None:
        f.write(
            '<Topology TopologyType="3DRectMesh" Dimensions="{} {} {}">\n'.format(
                *grid_shape
            )
        )
        f.write("</Topology>\n")

    def _write_geometry(self, f, grid_shape: tuple, filename: str) -> None:
        f.write('<Geometry GeometryType="VXVYVZ">\n')
        f.write(
            f'<DataItem Dimensions="{grid_shape[0]}" NumberType="Float" Precision="8" Format="HDF">\n'
        )
        f.write(f"{filename}:/edgex\n")
        f.write("</DataItem>\n")
        f.write(
            f'<DataItem Dimensions="{grid_shape[1]}" NumberType="Float" Precision="8" Format="HDF">\n'
        )
        f.write(f"{filename}:/edgey\n")
        f.write("</DataItem>\n")
        f.write(
            f'<DataItem Dimensions="{grid_shape[2]}" NumberType="Float" Precision="8" Format="HDF">\n'
        )
        f.write(f"{filename}:/edgez\n")
        f.write("</DataItem>\n")
        f.write("</Geometry>\n")

    def _write_attribute(self, f, label: str, grid_shape: tuple, filename: str) -> None:
        f.write(f'<Attribute Name="{label}" AttributeType="Scalar" Center="Cell">\n')
        f.write(
            '<DataItem Dimensions="{} {} {}" NumberType="Float" Precision="8" Format="HDF">\n'.format(
                *grid_shape
            )
        )
        f.write(f"{filename}:/{label}\n")
        f.write("</DataItem>\n")
        f.write("</Attribute>\n")

    def write(self) -> str:
        """
        Write the XDMF file for the current snapshot.
        """
        outname = self.snapshot.basename + ".xdmf"
        with open(outname, "w") as f:
            self._write_header(f)
            self._write_time(f, self.snapshot.time)
            self._write_topology(f, self.grid_shape)
            self._write_geometry(f, self.grid_shape, self.filename)
            for label in self.snapshot.keys:
                if label in self._ignore_keys:
                    continue
                self._write_attribute(f, label, self.attribute_shape, self.filename)
            self._write_footer(f)
