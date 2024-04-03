import os
import re
import glob
import struct

import h5py
import numpy as np


def get_snaplist(model, snappath="./"):
    # check parallel files first
    outfiles = glob.glob(model + "o[0-9][0-9][0-9].hdf5", root_dir=snappath)
    snaplist = []
    for name in outfiles:
        snaplist.append(int(re.sub(model + r"o([0-9]{3}).*", r"\1", name)))

    return sorted(snaplist)


def readsnap(ind, model, snappath="./", simulation_type="ONeDef", quiet=False):
    return LeafsSnapshot(
        os.path.join(snappath, "{:s}o{:03d}".format(model, int(ind))),
        simulation_type=simulation_type,
        quiet=quiet,
    )


class LeafsSnapshot:
    def __init__(self, filename, simulation_type="ONeDef", quiet=False):
        self.quiet = quiet
        self.data = {}
        return

    def __getattr__(self, __name: str):
        """enable access via object attributes to data dict entries"""
        if __name in self.data:
            return self.data[__name]
        else:
            raise AttributeError("{} has no attribute '{}'.".format(type(self), __name))

    def get_density_in_radius(self, center, radius):
        """return the density in a sphere of radius around center"""
        assert len(center) == 3, "Center must be a 3D point"
        assert isinstance(radius, (int, float)), "Radius must be a number"
        x = self.geomx
        y = self.geomy
        z = self.geomz
        xpos, ypos, zpos = np.meshgrid(x, y, z, indexing="ij")
        # Not sure about the indexing, i.e. C vs Fortran.
        # Shouldn't matter for spherical density
        rad = np.sqrt(
            (xpos - center[0]) ** 2 + (ypos - center[1]) ** 2 + (zpos - center[2]) ** 2
        )
        return self.density[rad < radius]

    def get_bound_material(self, include_internal_energy=False):
        """
        Returns a boolean mask for the bound material.
        If include_internal_energy is True, the internal energy
        is included in the bound criterion.

        Bound material is defined as material with a negative
        total energy:
            E_kin + E_grav + E_int < 0
        """

        total_energy = self.egrav + self.ekin
        if include_internal_energy:
            total_energy += self.eintkin

        return total_energy < 0

    def get_bound_mass(self, include_internal_energy=False):
        """
        Returns the mass of the bound material.
        If include_internal_energy is True, the internal energy
        is included in the bound criterion.

        Bound material is defined as material with a negative
        total energy:
            E_kin + E_grav + E_int < 0
        """

        bound_mask = self.get_bound_material(include_internal_energy)
        return np.sum(self.density[bound_mask] * self.vol[bound_mask])
