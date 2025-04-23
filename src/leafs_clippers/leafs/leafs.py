import os
import re
import glob
import struct
import itertools

import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import binned_statistic
from scipy.io import FortranFile

try:
    from singularity_eos import Helmholtz
except ImportError:
    pass

from leafs_clippers.util import utilities as util
from leafs_clippers.util import const as const


def get_snaplist(model, snappath="./", legacy=False, reduced_output=False):
    file_base = "redo" if reduced_output else "o"
    glob_pattern = f"{model}{file_base}[0-9][0-9][0-9]"

    # check parallel files first
    if legacy:
        outfiles = glob.glob(glob_pattern + ".000", root_dir=snappath)
        # check output from serial run if nothing was found
        if len(outfiles) == 0:
            outfiles = glob.glob(glob_pattern, root_dir=snappath)
    else:
        outfiles = glob.glob(glob_pattern + ".hdf5", root_dir=snappath)
    snaplist = []
    for name in outfiles:
        snaplist.append(
            int(re.sub(f"{model}{file_base}" + r"([0-9]{3}).*", r"\1", name))
        )

    return sorted(snaplist)


def readsnap(
    ind,
    model,
    snappath="./",
    simulation_type="ONeDef",
    quiet=False,
    legacy=False,
    little_endian=True,
    write_derived=True,
    ignore_cache=False,
    helm_table="helm_table.dat",
    reduced_output=False,
    remnant_threshold=1e4,
):
    file_base = "redo" if reduced_output else "o"
    if legacy:
        return LeafsLegacySnapshot(
            os.path.join(snappath, "{:s}{:s}{:03d}".format(model, file_base, int(ind))),
            simulation_type=simulation_type,
            quiet=quiet,
            little_endian=little_endian,
        )
    else:
        return LeafsSnapshot(
            os.path.join(
                snappath, "{:s}{:s}{:03d}.hdf5".format(model, file_base, int(ind))
            ),
            quiet=quiet,
            helm_table="helm_table.dat",
            write_derived=write_derived,
            ignore_cache=ignore_cache,
            remnant_threshold=remnant_threshold,
        )


def readprotocol(
    model,
    snappath="./",
    simulation_type="ONeDef",
    quiet=False,
):
    """
    Read the protocol file of a LEAFS simulation.

    Parameters
    ----------
    model : str
        The model name.
    snappath : str, optional
        The path to the snapshot files.
    simulation_type : str, optional
        The simulation type.
    quiet : bool, optional
        If True, suppress output.

    Returns
    -------
    LeafsProtocol
        The protocol object.
    """
    return LeafsProtocol(
        model=model,
        snappath=snappath,
        simulation_type=simulation_type,
        quiet=quiet,
    )


def readflameprotocol(
    model,
    snappath="./",
):
    """
    Read the flame protocol file of a LEAFS simulation.

    Parameters
    ----------
    model : str
        The model name.
    snappath : str, optional
        The path to the snapshot files.

    Returns
    -------
    FlameProtocol
        The protocol object.
    """
    return FlameProtocol(
        model=model,
        snappath=snappath,
    )


class LeafsSnapshot:
    def __init__(
        self,
        filename,
        quiet=False,
        helm_table="helm_table.dat",
        write_derived=True,
        ignore_cache=False,
        remnant_threshold=1e4,
    ):
        """
        Read a LEAFS snapshot from an HDF5 file.

        Parameters
        ----------
        filename : str
            The filename of the snapshot.
        quiet : bool, optional
            If True, suppress output.
        helm_table : str, optional
            The filename of the EOS table. Needed for functions that require the EOS.
        write_derived : bool, optional
            If True, derived quantities will be written to a npy file in the same directory
            as the snapshot. Unless called with 'ignore_cache=True', the derived
            quantities will be read from the cache file when available.
        ignore_cache : bool, optional
            If True, the cache file will be ignored and derived quantities will be recomputed.
        remnant_threshold : float, optional
            Threshold for what densities are considered to be part of the remnant.
            This only makes sense for snapshots at around 100s post explosion.
        """
        self.quiet = quiet
        self.filename = filename
        self.basename = filename.replace(".hdf5", "")
        self.helm_table = helm_table
        self.write_derived = write_derived
        self.ignore_cache = ignore_cache
        self.remnant_threshold = remnant_threshold

        try:
            f = h5py.File(filename, "r")
        except FileNotFoundError:
            raise FileNotFoundError("File {} not found.".format(filename))

        try:
            if os.path.exists(helm_table):
                self.eos = Helmholtz(helm_table)
            else:
                self.eos = None
        except NameError:
            self.eos = None

        # Read meta data
        try:
            self.time = f.attrs["time"]
            self.gnx = f.attrs["gnx"]
            self.gny = f.attrs["gny"]
            self.gnz = f.attrs["gnz"]
            self.ncells = f.attrs["ncells"]
            self.rad_wd = f.attrs["rad_wd"]
            self.rad_fl = f.attrs["rad_fl"]
            self.idx_wd = f.attrs["idx_wd"]
            self.idx_fl = f.attrs["idx_fl"]
            self.simulation_type = f.attrs["simulation_type"]

            # Create LazyDict
            self.keys = list(f.keys())
            self.data = util.LazyDict(filename, self.keys)
        finally:
            f.close()
        return

    def __getattr__(self, __name: str):
        """enable access via object attributes to data dict entries"""
        if __name in self.data:
            return self.data[__name]
        elif __name == "vel_abs":
            return self.get_abs_velocity()
        elif __name == "c_sound":
            return self.get_c_sound()
        elif __name == "mach":
            return self.get_mach()
        else:
            raise AttributeError("{} has no attribute '{}'.".format(type(self), __name))

    @classmethod
    def schwab_flamespeed(cls, rho, ye):
        """
        Computes the flame speed using the Schwab et al. 2020, ApJ 891:5 formula.
        Result is in cm/s.
        """

        vlam = 16.0e5 * rho / 1e9 * (1 + 96.88 * (0.5 - ye))
        return vlam

    @classmethod
    def timmes_flamespeed(cls, rho, xfuel):
        """
        Computes the flame speed using the formula of Timmes & Woosley 1992, ApJ 396:649-667
        This is only exact if the burn density and the fuel fraction are provided.
        These quantities are not stored in default snapshots.
        Result is in cm/s.
        """

        vlam = 51.8e5 * (rho / 6e9) ** 1.06 * (xfuel / 0.6) ** 0.688
        return vlam

    @property
    def mass(self):
        return np.array(self.density * self.vol, dtype=np.float64)

    @property
    def vel_abs(self):
        _ = self.get_abs_velocity()
        return self.data["vel_abs"]

    @property
    def radius(self):
        return self.data["geomx"][self.gnx // 2 :]

    @property
    def c_sound(self):
        _ = self.get_c_sound()
        return self.data["c_sound"]

    @property
    def mach(self):
        _ = self.get_mach()
        return self.data["mach"]

    @property
    def lset_dist(self):
        _ = self.get_lset_dist()
        return self.data["lset_dist"]

    @property
    def has_remnant(self):
        remnant_density = np.ma.masked_array(
            self.density, mask=np.logical_not(self.density > self.remnant_threshold)
        )
        return len(remnant_density.compressed()) > 0

    def _load_derived(self, field):
        """
        Load derived quantity from cache hdf5 file.

        Parameters
        ----------
        field : str
            The name of the derived quantity.

        Returns
        -------
        bool
            True if the field was found in the cache file, False otherwise.
        """
        cache_filename = self.basename + "_derived.hdf5"
        try:
            with h5py.File(cache_filename, "r") as f:
                if field in f:
                    self.data[field] = f[field][...]
                    return True
                else:
                    return False
        except FileNotFoundError:
            return False

    def _write_derived(self, field):
        """
        Write derived quantity to cache hdf5 file.

        Parameters
        ----------
        field : str
            The name of the derived quantity.
        """
        cache_filename = self.basename + "_derived.hdf5"
        try:
            with h5py.File(cache_filename, "a") as f:
                if field not in f:
                    f.create_dataset(field, data=self.data[field])
                else:
                    f[field][...] = self.data[field]
        except PermissionError:
            print(
                f"No permission to write in snapshot directory, {field} not cached..."
            )

        return

    def export_to_vtk(self):
        """
        Export the snapshot to a VTK file. Requires pyevtk.

        Parameters
        ----------
        filename : str
            The filename of the VTK file.
        fields : list, optional
            The list of fields to export. If None, all fields will be exported.
        """
        try:
            from pyevtk.hl import gridToVTK
        except ImportError:
            raise ImportError("pyevtk is required to export to VTK")

        vtk_filename = self.basename + ".vtk"

        print("Exporting to VTK file: {}".format(vtk_filename))

        # Generate mesh
        x, y, z = np.meshgrid(self.edgex, self.edgey, self.edgez)

        cell_data = {}
        for key in self.keys:
            # If array isn't 3D, skip
            if self.data[key].ndim != 3:
                continue
            cell_data[key] = self.data[key]

        field_data = {}
        field_data["time"] = np.array([self.time])
        field_data["gnx"] = np.array([self.gnx])
        field_data["gny"] = np.array([self.gny])
        field_data["gnz"] = np.array([self.gnz])
        field_data["ncells"] = np.array([self.ncells])
        field_data["rad_wd"] = np.array([self.rad_wd])
        field_data["rad_fl"] = np.array([self.rad_fl])
        field_data["idx_wd"] = np.array([self.idx_wd])
        field_data["idx_fl"] = np.array([self.idx_fl])

        gridToVTK(vtk_filename, x, y, z, cellData=cell_data, fieldData=field_data)

        return

    def get_abs_velocity(self):
        if not self.ignore_cache:
            if self._load_derived("vel_abs"):
                return

        self.data["vel_abs"] = np.sqrt(
            self.data["velx"] ** 2 + self.data["vely"] ** 2 + self.data["velz"] ** 2
        )

        if self.write_derived:
            self._write_derived("vel_abs")

        return self.data["vel_abs"]

    def get_c_sound(self):
        if self.eos is None:
            if not self.quiet:
                print("No EOS available, setting sound speed to zero.")
            return np.zeros_like(self.density)

        if not self.ignore_cache:
            if self._load_derived("c_sound"):
                return

        self.data["c_sound"] = np.zeros_like(self.density)
        abar = np.array(self.Amean.flatten(), dtype=np.float64)
        zbar = abar * np.array(self.ye.flatten(), dtype=np.float64)
        bmods = np.zeros_like(abar, dtype=np.float64)
        self.eos.BulkModulusFromDensityTemperature(
            np.array(self.density.flatten(), dtype=np.float64),
            np.array(self.temp.flatten(), dtype=np.float64),
            bmods,
            len(bmods),
            np.array([abar, zbar, np.log10(self.temp.flatten())], dtype=np.float64).T,
        )
        self.data["c_sound"] = np.array(
            np.sqrt(bmods / self.density.flatten()).reshape(
                self.gnx, self.gny, self.gnz
            ),
            dtype=np.float64,
        )

        if self.write_derived:
            self._write_derived("c_sound")

        return self.data["c_sound"]

    def get_mach(self):
        self.get_abs_velocity()
        self.get_c_sound()

        self.data["mach"] = np.zeros_like(self.density)
        self.data["mach"] = self.vel_abs / self.c_sound

        return self.data["mach"]

    def get_lset_dist(self, border=4):
        """
        Get the distance to the isosurface described by the level set function.
        Due to the lack of ghost-cells, this is only defined in the interior
        and the edges are clipped to the maximum distance.
        WARNING: For now this only works on the first levelset.
        Also probably only works in 3D.

        Parameters
        ----------
        border : int, optional
            The number of cells for which to compute the distance in
            each direction.
        """
        if not self.ignore_cache:
            if self._load_derived("lset_dist"):
                return

        assert self.lset1 is not None, "No level set function available"

        # Base distance to which everything gets clipped
        xlen = self.edgex[:-1] - self.edgex[1:]
        ylen = self.edgey[:-1] - self.edgey[1:]
        zlen = self.edgez[:-1] - self.edgez[1:]
        ref_len = np.min([xlen, ylen, zlen])
        dist = np.ones_like(self.density)
        for i in range(self.gnx):
            for j in range(self.gny):
                for k in range(self.gnz):
                    dist[i, j, k] = (
                        20 * 4 * max([xlen[i], ylen[j], zlen[k], ref_len])
                    ) ** 2
        # x-dir
        for i in range(self.gnx - 1):
            for j in range(self.gny):
                for k in range(self.gnz):
                    if (self.lset1[i, j, k] * self.lset1[i + 1, j, k]) < 0:
                        crx = self.geomx[i] + np.abs(self.lset1[i, j, k]) / np.abs(
                            self.lset1[i + 1, j, k] - self.lset1[i, j, k]
                        ) * (self.geomx[i + 1] - self.geomx[i])
                        for i2 in range(
                            np.max([0, i - border]), np.min([self.gnx, i + border + 1])
                        ):
                            for k2 in range(
                                np.max([0, k - border]), np.min([self.gnz, k + border])
                            ):
                                for j2 in range(
                                    np.max([0, j - border]),
                                    np.min([self.gny, j + border]),
                                ):
                                    dist[i2, j2, k2] = min(
                                        dist[i2, j2, k2],
                                        np.abs(
                                            (self.geomx[i2] - crx) ** 2
                                            + (self.geomy[j2] - self.geomy[j]) ** 2
                                            + (self.geomz[k2] - self.geomz[k]) ** 2
                                        ),
                                    )

        # y-dir
        for i in range(self.gnx):
            for j in range(self.gny - 1):
                for k in range(self.gnz):
                    if (self.lset1[i, j, k] * self.lset1[i, j + 1, k]) < 0:
                        cry = self.geomy[j] + np.abs(self.lset1[i, j, k]) / np.abs(
                            self.lset1[i, j + 1, k] - self.lset1[i, j, k]
                        ) * (self.geomy[j + 1] - self.geomy[j])
                        for i2 in range(
                            np.max([0, i - border]), np.min([self.gnx, i + border])
                        ):
                            for k2 in range(
                                np.max([0, k - border]),
                                np.min([self.gnz, k + border + 1]),
                            ):
                                for j2 in range(
                                    np.max([0, j - border]),
                                    np.min([self.gny, j + border]),
                                ):
                                    dist[i2, j2, k2] = min(
                                        dist[i2, j2, k2],
                                        np.abs(
                                            (self.geomx[i2] - self.geomx[i]) ** 2
                                            + (self.geomy[j2] - cry) ** 2
                                            + (self.geomz[k2] - self.geomz[k]) ** 2
                                        ),
                                    )

        # z-dir
        for i in range(self.gnx):
            for j in range(self.gny):
                for k in range(self.gnz - 1):
                    if (self.lset1[i, j, k] * self.lset1[i, j, k + 1]) < 0:
                        crz = self.geomz[k] + np.abs(self.lset1[i, j, k]) / np.abs(
                            self.lset1[i, j, k + 1] - self.lset1[i, j, k]
                        ) * (self.geomz[k + 1] - self.geomz[k])
                        for i2 in range(
                            np.max([0, i - border]), np.min([self.gnx, i + border])
                        ):
                            for k2 in range(
                                np.max([0, k - border]), np.min([self.gnz, k + border])
                            ):
                                for j2 in range(
                                    np.max([0, j - border]),
                                    np.min([self.gny, j + border + 1]),
                                ):
                                    dist[i2, j2, k2] = min(
                                        dist[i2, j2, k2],
                                        np.abs(
                                            (self.geomx[i2] - self.geomx[i]) ** 2
                                            + (self.geomy[j2] - self.geomy[j]) ** 2
                                            + (self.geomz[k2] - crz) ** 2
                                        ),
                                    )

        dist = np.sqrt(dist)
        self.data["lset_dist"] = dist

        if self.write_derived:
            self._write_derived("lset_dist")

        return self.data["lset_dist"]

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

    def get_internal_energy_from_eos(self):
        """
        Compute the internal energy from the EOS.
        """
        # Attempt to load internal energy from cache
        if not self.ignore_cache:
            if self._load_derived("e_internal"):
                return

        if not self.quiet:
            print("Computing internal energy from EOS...")

        self.data["e_internal"] = np.zeros_like(self.density)
        if self.eos is None:
            print("No EOS available, setting internal energy to zero.")
        for i in tqdm(range(self.gnx)):
            for j in range(self.gny):
                for k in range(self.gnz):
                    # Ye = zbar / abar => zbar = Ye * abar
                    abar = self.Amean[i, j, k]
                    zbar = abar * self.ye[i, j, k]
                    self.e_internal[i, j, k] = (
                        self.eos.InternalEnergyFromDensityTemperature(
                            self.density[i, j, k],
                            self.temp[i, j, k],
                            np.array([abar, zbar, np.log10(self.temp[i, j, k])]),
                        )
                    )

        # Write internal energy to cache
        if self.write_derived:
            self._write_derived("e_internal")

        return

    def _get_remnant_velocity(self):
        """
        Get the velocity of the dense remnant.
        """
        if not self.has_remnant:
            return 0.0, 0.0, 0.0

        remnant_mask = self.density > self.remnant_threshold
        remnant_mass = np.sum(self.mass[remnant_mask])
        remnant_velx = (
            np.sum(self.velx[remnant_mask] * self.mass[remnant_mask]) / remnant_mass
        )
        remnant_vely = (
            np.sum(self.vely[remnant_mask] * self.mass[remnant_mask]) / remnant_mass
        )
        remnant_velz = (
            np.sum(self.velz[remnant_mask] * self.mass[remnant_mask]) / remnant_mass
        )

        return remnant_velx, remnant_vely, remnant_velz

    def get_bound_material(
        self,
        include_internal_energy=True,
        eint_from_eos=False,
        vacuum_threshold=1e-4,
        check_remnant=True,
    ):
        """
        Returns a boolean mask for the bound material.
        If include_internal_energy is True, the internal energy
        is included in the bound criterion.

        Bound material is defined as material with a negative
        total energy:
            E_kin + E_grav + E_int < 0

        Parameters
        ----------
        include_internal_energy : bool, optional
            If True, the internal energy is included in the bound criterion.
        eint_from_eos : bool, optional
            If True, the internal energy is computed from the EOS.
        vacuum_threshold : float, optional
            The threshold for densities to be considered vacuum.
        check_remnant : bool, optional
            If True, the velocity of the dense remnant is subtracted in the
            bound criterion.
        """

        if not self.ignore_cache:
            if self._load_derived("bound_mask"):
                return self.data["bound_mask"]

        if eint_from_eos:
            self.get_internal_energy_from_eos()

        remnant_velx, remnant_vely, remnant_velz = 0.0, 0.0, 0.0
        if check_remnant:
            remnant_velx, remnant_vely, remnant_velz = self._get_remnant_velocity()

        ekin = 0.5 * (
            (self.velx - remnant_velx) ** 2
            + (self.vely - remnant_vely) ** 2
            + (self.velz - remnant_velz) ** 2
        )

        # Note that quantities are stored per unit mass
        total_energy = (self.egrav + ekin) * self.mass
        if include_internal_energy:
            if eint_from_eos:
                total_energy += self.e_internal
            else:
                total_energy += self.energy * self.mass

        self.data["bound_mask"] = np.logical_and(
            total_energy < 0, self.density > vacuum_threshold
        )
        if self.write_derived:
            self._write_derived("bound_mask")

        return self.data["bound_mask"]

    def get_bound_mass(
        self,
        include_internal_energy=True,
        eint_from_eos=False,
        use_dmass=True,
        vacuum_threshold=1e-4,
        check_remnant=True,
    ):
        """
        Returns the mass of the bound material.
        If include_internal_energy is True, the internal energy
        is included in the bound criterion.

        Bound material is defined as material with a negative
        total energy:
            E_kin + E_grav + E_int < 0
        """

        bound_mask = self.get_bound_material(
            include_internal_energy,
            eint_from_eos=eint_from_eos,
            vacuum_threshold=vacuum_threshold,
            check_remnant=check_remnant,
        )
        if use_dmass:
            return np.sum(self.dmass[bound_mask])
        return np.sum(self.density[bound_mask] * self.vol[bound_mask])

    def get_kick_velocity(
        self,
        include_internal_energy=True,
        eint_from_eos=False,
        return_dirs=False,
        check_remnant=True,
        vacuum_threshold=1e-4,
    ):
        """
        Returns the kick velocity of the bound material.
        If include_internal_energy is True, the internal energy
        is included in the bound criterion.

        Bound material is defined as material with a negative
        total energy:
            E_kin + E_grav + E_int < 0

        Parameters
        ----------
        include_internal_energy : bool, optional
            If True, the internal energy is included in the bound criterion.
        eint_from_eos : bool, optional
            If True, the internal energy is computed from the EOS.
        return_dirs : bool, optional
            If True, the kick velocity components are returned.
        check_remnant : bool, optional
            If True, the velocity of the dense remnant is subtracted in the
            bound criterion.
        vacuum_threshold : float, optional
            The threshold for densities to be considered vacuum.
        """

        bound_mask = self.get_bound_material(
            include_internal_energy,
            eint_from_eos=eint_from_eos,
            vacuum_threshold=vacuum_threshold,
            check_remnant=check_remnant,
        )
        vx = np.sum(
            (
                np.array(self.velx, dtype=np.float64)
                * np.array(self.dmass, dtype=np.float64)
            )[bound_mask]
            / const.M_SOL
        )
        vy = np.sum(
            (
                np.array(self.vely, dtype=np.float64)
                * np.array(self.dmass, dtype=np.float64)
            )[bound_mask]
            / const.M_SOL
        )
        vz = np.sum(
            (
                np.array(self.velz, dtype=np.float64)
                * np.array(self.dmass, dtype=np.float64)
            )[bound_mask]
            / const.M_SOL
        )

        vel_kick = np.sqrt(vx**2 + vy**2 + vz**2)

        if return_dirs:
            return vel_kick, vx, vy, vz
        return vel_kick

    def get_remnant_info(self, include_internal_energy=True, eint_from_eos=False):
        """
        Returns the remnant mass and kick velocity.
        If include_internal_energy is True, the internal energy
        is included in the bound criterion.
        Quantities are returned in solar masses and km/s.

        Bound material is defined as material with a negative
        total energy:
            E_kin + E_grav + E_int < 0
        """

        bound_mask = self.get_bound_material(
            include_internal_energy, eint_from_eos=eint_from_eos
        )
        bound_mass = self.get_bound_mass(
            include_internal_energy, eint_from_eos=eint_from_eos
        )
        vel_kick, vx, vy, vz = self.get_kick_velocity(
            include_internal_energy, eint_from_eos=eint_from_eos, return_dirs=True
        )
        ige = np.sum((self.dmass * self.xnuc05)[bound_mask]) / const.M_SOL
        nifs = np.sum((self.dmass * self.nifs)[bound_mask]) / const.M_SOL
        ime = np.sum((self.dmass * self.xnuc04)[bound_mask]) / const.M_SOL
        rhomax = np.max(self.density[bound_mask])

        return {
            "bound_mass": bound_mass / const.M_SOL,
            "vel_kick": vel_kick / 1e5,
            "vx": vx / 1e5,
            "vy": vy / 1e5,
            "vz": vz / 1e5,
            "ige": ige,
            "nifs": nifs,
            "ime": ime,
            "rhomax": rhomax,
        }

    def get_central_value(self, value):
        """
        Returns the central value of a quantity.
        """
        assert isinstance(value, str), "Value must be a string"
        mid = int(self.gnx / 2)
        return np.mean(
            self.data[value][mid - 1 : mid + 2, mid - 1 : mid + 2, mid - 1 : mid + 2]
        )

    def get_rad_profile(
        self,
        value,
        res=None,
        statistic="sum",
        extensive=False,
        min_radius=0,
        max_radius=None,
        return_edges=False,
    ):
        """
        Returns the (1D) radial averaged profile of a quantity.
        Uses binned_statistic to bin the data.
        If res is not None, the data will be binned into original resolution.

        Parameters
        ----------
        value : str
            The name of the quantity to bin. If value == "density", the density
            will be computed from the mass and volume of each bin.
        res : int, optional
            The resolution to bin the data into.
        statistic : str, optional
            The statistic to compute in each bin. Passed to binned_statistic.
        extensive : bool, optional
            If True, the quantity is treated as extensive. This sets the
            statistic to 'sum'. Intensive quantities will be weighted by the
            mass of each cell.
        min_radius : float, optional
            The minimum radius to consider.
        max_radius : float, optional
            The maximum radius to consider.
        return_edges : bool, optional
            If True, the bin edges are returned as well.

        Returns
        -------
        bin_centers : np.ndarray
            The bin centers.
        bin_values : np.ndarray
            The binned values.
        bin_edges : np.ndarray
            The bin edges (if return_edges is True).
        """
        assert isinstance(value, str), "Value must be a string"

        if extensive or (value == "density"):
            # We treat density as an extensive quantity since we only need
            # to average the mass and recompute the density from that.
            statistic = "sum"
            extensive = True

        if res is None:
            res = int(self.gnx // 2)

        xx, yy, zz = np.meshgrid(self.geomx, self.geomy, self.geomz)
        r = np.sqrt(xx**2 + yy**2 + zz**2)

        r_flat = r.flatten()
        value_flat = self.data[value].flatten()

        if max_radius is None:
            max_radius = np.max(r_flat)

        assert min_radius < max_radius, "min_radius must be smaller than max_radius"

        if (not extensive) or (value == "density"):
            # Careful here: usually a weighted arithmetic mean is computed
            # as sum(value * weight) / sum(weight). Here we use
            # 1/n * sum(value * weight) / (1/n * sum(weight)) = sum(value * weight) / sum(weight)
            # This is equivalent to the weighted arithmetic mean.
            # The reason for this is so we can also use the median as a statistic.
            # Otherwise 'sum' would be used in every case.
            mass_flat = self.mass.flatten()
            bin_values, bin_edges, _ = util.binned_statistic_weighted(
                r_flat,
                value_flat,
                weights=mass_flat,
                bins=res,
                statistic=statistic,
                range=(min_radius, max_radius),
            )
            # Special case: for the density we recompute the density from the mass and volume
            if value == "density":
                mass_binned, _, _ = binned_statistic(
                    r_flat,
                    mass_flat,
                    bins=res,
                    statistic="sum",
                    range=(min_radius, max_radius),
                )
                vol = 4 / 3 * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_values = mass_binned / vol

                if return_edges:
                    return bin_centers, bin_values, bin_edges

                return bin_centers, bin_values
        else:
            bin_values, bin_edges, _ = binned_statistic(
                r_flat,
                value_flat,
                bins=res,
                statistic=statistic,
                range=(min_radius, max_radius),
            )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if return_edges:
            return bin_centers, bin_values, bin_edges

        return bin_centers, bin_values

    def _get_label_from_key(self, key):
        """
        Get a label for a quantity from its key.

        Parameters
        ----------
        key : str
            The key of the quantity.

        Returns
        -------
        label : str
            The label of the quantity.
        """
        try:
            label = const.KEY_TO_LABEL_DICT[key]
        except KeyError:
            label = key
        return label

    def _get_fmt_from_key(self, key):
        """
        Get a format function for a quantity from its key.

        Parameters
        ----------
        key : str
            The key of the quantity.

        Returns
        -------
        fmt : function
            The format function of the quantity.
        """

        try:
            fmt = const.KEY_TO_FMT_DICT[key]
        except KeyError:
            fmt = const.FLOAT_FMT
        return fmt

    def _get_cmap_from_key(self, key):
        """
        Get a colormap for a quantity from its key.

        Parameters
        ----------
        key : str
            The key of the quantity.

        Returns
        -------
        cmap : str
            The colormap of the quantity.
        """
        try:
            cmap = const.KEY_TO_CMAP_DICT[key]
        except KeyError:
            cmap = "cubehelix"
        return cmap

    def _get_box_min_max(self, axis="z", boxsize=None, center_offset=0):
        """
        Get the min and max indices of a box centered in the domain.

        Parameters
        ----------
        axis : str
            The axis along which to slice.
        boxsize : int
            The size of the box in cell counts.
        center_offset : int, optional
            Offset from the center of the domain in cell counts.

        Returns
        -------
        box_min : int
            The minimum index of the box.
        box_max : int
            The maximum index of the box.
        """
        if axis == "z":
            length = self.gnz
        elif axis == "y":
            length = self.gny
        elif axis == "x":
            length = self.gnx

        if boxsize is None:
            boxsize = length

        if boxsize > length:
            raise ValueError("Boxsize must be smaller than the domain size.")

        n = boxsize // 2
        box_min = int(length // 2 - n)
        box_max = int(length // 2 + n)

        if box_min - center_offset < 0:
            raise ValueError("Box is out of bounds.")
        if box_max + center_offset > length:
            raise ValueError("Box is out of bounds.")

        box_min -= center_offset
        box_max -= center_offset

        return box_min, box_max

    def _get_slice(self, data, axis, index, boxsize=None, center_offset=[0, 0]):
        """
        Get a 2D slice of a quantity.

        Parameters
        ----------
        data : np.ndarray
            The data to slice.
        axis : str
            The axis along which to slice.
        index : int
            The index of the slice.
        boxsize : int, optional
            The size of the box in cell counts. If None, the whole domain is used.
        center_offset : list of int, optional
            Offset from the center of the domain in cell counts.

        Returns
        -------
        np.ndarray
            The 2D slice.
        """
        assert axis in ["x", "y", "z"], "Axis must be 'x', 'y', or 'z'"
        assert isinstance(index, int), "Index must be an integer"

        if axis == "z":
            axes = ["x", "y"]
        elif axis == "y":
            axes = ["x", "z"]
        elif axis == "x":
            axes = ["y", "z"]

        box_min_0, box_max_0 = self._get_box_min_max(axes[0], boxsize, center_offset[0])
        box_min_1, box_max_1 = self._get_box_min_max(axes[1], boxsize, center_offset[1])

        if axis == "z":
            return data[box_min_0:box_max_0, box_min_1:box_max_1, index]
        elif axis == "y":
            return data[box_min_0:box_max_0, index, box_min_1:box_max_1]
        elif axis == "x":
            return data[index, box_min_0:box_max_0, box_min_1:box_max_1]

    def get_slice(self, key, axis, index, boxsize=None, center_offset=[0, 0]):
        """
        Get a 2D slice of a quantity.

        Parameters
        ----------
        key : str
            The name of the quantity to slice.
        axis : str
            The axis along which to slice.
        index : int
            The index of the slice.
        boxsize : int, optional
            The size of the box in cell counts. If None, the whole domain is used.
        center_offset : list of int, optional
            Offset from the center of the domain in cell counts.

        Returns
        -------
        np.ndarray
            The 2D slice.
        """
        assert axis in ["x", "y", "z"], "Axis must be 'x', 'y', or 'z'"
        assert isinstance(index, int), "Index must be an integer"

        if key not in self.data:
            raise KeyError("No quantity named {:s} in data dictionary".format(key))
        return self._get_slice(self.data[key], axis, index, boxsize, center_offset)

    def _get_edge_slice(self, axis, boxsize=None, center_offset=0):
        """
        Get slice of cell edge coordinates.

        Parameters
        ----------
        axis : str
            The axis along which to slice.
        boxsize : int, optional
            The size of the box in cell counts. If None, the whole domain is used.
        center_offset : list of int, optional
            Offset from the center of the domain in cell counts.

        Returns
        -------
        slice_0 : np.ndarray
            The slice of the edge coordinates.
        slice_1 : np.ndarray
            The slice of the edge coordinates.

        """
        assert axis in ["x", "y", "z"], "Axis must be 'x', 'y', or 'z'"
        if len(center_offset) < 2:
            return ValueError("Center offset must be a list of two integers")

        if axis == "z":
            axes = ["x", "y"]
        elif axis == "y":
            axes = ["x", "z"]
        elif axis == "x":
            axes = ["y", "z"]

        box_min_0, box_max_0 = self._get_box_min_max(axes[0], boxsize, center_offset[0])
        box_min_1, box_max_1 = self._get_box_min_max(axes[1], boxsize, center_offset[1])

        if axis == "z":
            slice_0 = self.data["edgex"][box_min_0 : box_max_0 + 1]
            slice_1 = self.data["edgey"][box_min_1 : box_max_1 + 1]
        elif axis == "y":
            slice_0 = self.data["edgex"][box_min_0 : box_max_0 + 1]
            slice_1 = self.data["edgez"][box_min_1 : box_max_1 + 1]
        elif axis == "x":
            slice_0 = self.data["edgey"][box_min_0 : box_max_0 + 1]
            slice_1 = self.data["edgez"][box_min_1 : box_max_1 + 1]
        return slice_0, slice_1

    def _get_geom_slice(self, axis, boxsize=None, center_offset=0):
        """
        Get slice of cell geom coordinates.

        Parameters
        ----------
        axis : str
            The axis along which to slice.
        boxsize : int, optional
            The size of the box in cell counts. If None, the whole domain is used.
        center_offset : list of int, optional
            Offset from the center of the domain in cell counts.

        Returns
        -------
        slice_0 : np.ndarray
            The slice of the geom coordinates.
        slice_1 : np.ndarray
            The slice of the geom coordinates.

        """
        assert axis in ["x", "y", "z"], "Axis must be 'x', 'y', or 'z'"
        if len(center_offset) < 2:
            return ValueError("Center offset must be a list of two integers")

        if axis == "z":
            axes = ["x", "y"]
        elif axis == "y":
            axes = ["x", "z"]
        elif axis == "x":
            axes = ["y", "z"]

        box_min_0, box_max_0 = self._get_box_min_max(axes[0], boxsize, center_offset[0])
        box_min_1, box_max_1 = self._get_box_min_max(axes[1], boxsize, center_offset[1])

        if axis == "z":
            slice_0 = self.data["edgex"][box_min_0:box_max_0]
            slice_1 = self.data["edgey"][box_min_1:box_max_1]
        elif axis == "y":
            slice_0 = self.data["edgex"][box_min_0:box_max_0]
            slice_1 = self.data["edgez"][box_min_1:box_max_1]
        elif axis == "x":
            slice_0 = self.data["edgey"][box_min_0:box_max_0]
            slice_1 = self.data["edgez"][box_min_1:box_max_1]

        return slice_0, slice_1

    def plot_grid_lines(
        self, ax=None, axis="z", boxsize=None, center_offset=[0, 0], linecolor="black"
    ):
        """
        Plot grid lines in a 2D slice.

        Parameters
        ----------
        ax : matplotlib.axes, optional
            The axes to plot on. If None, a new figure is created.
        axis : str, optional
            The axis along which to slice.
        boxsize : int, optional
            The size of the box in cell counts. If None, the whole domain is used.
        center_offset : list of int, optional
            Offset from the center of the domain in cell counts.
        linecolor : str, optional
            The color of the grid lines.

        Returns
        -------
        ax : matplotlib.axes
            The axes containing the plot.
        """
        if ax is None:
            ax = plt.figure().add_subplot(111)

        slice_0, slice_1 = self._get_edge_slice(axis, boxsize, center_offset)

        for i in range(len(slice_0)):
            ax.axvline(slice_0[i], color=linecolor, lw=0.5)
        for i in range(len(slice_1)):
            ax.axhline(slice_1[i], color=linecolor, lw=0.5)

        return ax

    def plot_slice(
        self,
        key,
        ax=None,
        log=False,
        Min=None,
        Max=None,
        boxsize=None,
        axis="z",
        center_offset=[0, 0],
        index=None,
        cmap=None,
        show_lsets=True,
        lsets_colors=["white"],
        lsets_styles=["solid"],
        rasterized=True,
        show_cbar=True,
        show_time=True,
        cbar_label="from_key",
        plot_grid_lines=False,
        plot_gl_color="black",
        mask=None,
        cax=None,
        vmin=None,
        vmax=None,
    ):
        """Plot a 2D slice through the simulation data, showing one particular
        quantity and, if desired, the location of the level set(s).

        Parameters
        ----------
        key : str or list of str
            identifier for the quantity to be displayed; the key must be in
            data dictionary. If a list is provided, the quantities are summed
            up before plotting
        ax: matplotlib.axes or None
            axes object into which the plot is placed; if None a new figure
            with a 111 subplot is created and used (default None)
        log: bool
            display quantity linearly or logarithmically (default False)
        Min: float or None
            minimum value displayed in the pcolormesh plot (goes into vmin).
            Everything below will be clipped. If log is True, the logarithm
            will be automatically calculated before handing it to pcolormesh.
            If None, the minimum value of the quantity is used (default None)
        Max : float or None
            analogous to Min but for the maximum value in the pcolormesh plot
            (goes into vmax); default None
        boxsize : int or None
            size of the box (in cell counts) in which the slice is taken;
            if None, the whole domain is used (default None)
        axis : str
            axis along which the slice is taken; must be 'x', 'y', or 'z'
            (default 'z')
        center_offset : list of int
            offset from the center of the domain in cell counts. The first
            two entries are used, any further entries are ignored (default [0, 0])
        index : int or None
            index of the slice along the axis; if None, the middle slice is
            taken (default None)
        cmap : str or matplotlib.cm object
            color map to be used in pcolormesh (default 'None')
        show_lsets : bool
            show contours of the zero level set(s) (default True)
        lsets_colors : list of str
            list of colors to be used for the display of the level set(s). If
            number of list entries if smaller than the number of level sets
            shown, the list entries are recycled (default ["white"])
        lsets_styles : list of str
            list of line styles for the display of the level set(s). Works in
            the same way as lsets_colors (default ["solid"]).
        rasterized : bool
            passed directly to pcolormesh (default True)
        show_cbar : bool
            whether to show a colorbar. If so, space is always taken from ax
            (default True).
        show_time : bool
            whether to display the time of the snapshot (in s) as the plot
            title (default True)
        cbar_label : str
            label to be put onto the colorbar; if 'from_key' is used, the key
            is displayed, if log is set to True, the cbar_label is prepended by
            'log', if an empty string is provided no label will be shown
            (default 'from_key')
        plot_grid_lines : bool
            whether to plot grid lines (default False)
        plot_gl_color : str
            color of the grid lines (default 'black')
        mask : np.ndarray or None
            mask to be applied to the data before plotting; if None, no mask
            is applied (default None)
        cax : matplotlib.axes or None
            axes object into which the colorbar is placed; if None, the colorbar
            is placed into the same axes as the plot (default None)
        vmin : float or None
            minimum value displayed in the pcolormesh plot (goes into vmin).
            Everything below will be clipped. If log is True, the logarithm
            will be automatically calculated before handing it to pcolormesh.
        vmax : float or None
            analogous to Min but for the maximum value in the pcolormesh plot
            (goes into vmax); default None

        Returns
        -------
        ax : matplotlib.axes
            axes object containing the Slice
        """

        if index is None:
            index = int(self.gnz // 2)

        assert isinstance(key, (str, list)), "Key must be a string or a list of strings"

        mask_target_shape = (
            self.data[key].shape if isinstance(key, str) else self.data[key[0]].shape
        )

        if mask is not None and mask.shape != mask_target_shape:
            raise ValueError("Mask shape must match data shape.")

        G2 = None

        try:
            if isinstance(key, list):
                Z = np.zeros(self.data[key[0]].shape)
                for k in key:
                    Z += self.data[k].T
                Z = self._get_slice(Z, axis, index, boxsize, center_offset)
            elif isinstance(key, str):
                Z = self.get_slice(
                    key, axis, index, boxsize=boxsize, center_offset=center_offset
                ).T
        except KeyError:
            raise ValueError("No quantity named {:s} in data dictionary".format(key))

        if mask is not None:
            mask = self._get_slice(mask, axis, index, boxsize, center_offset)
            Z = np.ma.masked_array(Z, mask=np.logical_not(mask))

        if show_lsets:
            G1 = self.get_slice(
                "lset1",
                axis,
                index,
                boxsize=boxsize,
                center_offset=center_offset,
            ).T
            try:
                G2 = self.get_slice(
                    "lset2",
                    axis,
                    index,
                    boxsize=boxsize,
                    center_offset=center_offset,
                ).T
            except KeyError:
                pass

        if Min is None:
            Min = np.min(Z) if vmin is None else vmin
        if Max is None:
            Max = np.max(Z) if vmax is None else vmax
        if log:
            Z = np.log10(Z)
            Min = np.log10(Min)
            Max = np.log10(Max)

        grid_0, grid_1 = self._get_edge_slice(
            axis, boxsize=boxsize, center_offset=center_offset
        )
        geom_0, geom_1 = self._get_geom_slice(
            axis, boxsize=boxsize, center_offset=center_offset
        )

        lsets_colors = itertools.cycle(lsets_colors)
        lsets_styles = itertools.cycle(lsets_styles)

        if cmap is None:
            cmap = (
                self._get_cmap_from_key(key)
                if isinstance(key, str)
                else self._get_cmap_from_key(key[0])
            )
        if ax is None:
            ax = plt.figure().add_subplot(111)
        im = ax.pcolormesh(
            grid_0,
            grid_1,
            Z,
            cmap=cmap,
            vmin=Min,
            vmax=Max,
            rasterized=rasterized,
        )
        if show_lsets:
            ax.contour(
                geom_0,
                geom_1,
                G1,
                [0],
                colors=next(lsets_colors),
                linestyles=next(lsets_styles),
            )
            if G2 is not None:
                ax.contour(
                    geom_0,
                    geom_1,
                    G2,
                    [0],
                    colors=next(lsets_colors),
                    linestyles=next(lsets_styles),
                )
        ax.axis("image")
        if axis == "z":
            xlabel = r"$x$ (cm)"
            ylabel = r"$y$ (cm)"
        elif axis == "y":
            xlabel = r"$x$ (cm)"
            ylabel = r"$z$ (cm)"
        elif axis == "x":
            xlabel = r"$y$ (cm)"
            ylabel = r"$z$ (cm)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        axfmt = util.ScalarFormatterForceFormat()
        axfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(axfmt)
        ax.yaxis.set_major_formatter(axfmt)
        if show_time:
            ax.set_title(r"$t = {:>8.4f}\,\mathrm{{s}}$".format(self.time))
        if show_cbar:
            fmt = self._get_fmt_from_key(key)
            cbar = plt.colorbar(im, cax=cax, format=FuncFormatter(fmt))
            if cbar_label != "":
                if cbar_label == "from_key":
                    cbar_label = (
                        self._get_label_from_key(key)
                        if isinstance(key, str)
                        else self._get_label_from_key(key[0])
                    )
                if log:
                    cbar_label = r"$\log\,$" + cbar_label
                cbar.set_label(cbar_label)

        if plot_grid_lines:
            self.plot_grid_lines(
                ax, axis, boxsize, center_offset, linecolor=plot_gl_color
            )
        return im


class LeafsLegacySnapshot(LeafsSnapshot):
    def __init__(
        self, filename, simulation_type="ONeDef", quiet=False, little_endian=True
    ):
        self.data = {}
        assert simulation_type in [
            "CODef",
            "ONeDef",
            "HeDet",
        ], "Unrecognized simulation type"
        self.simulation_type = simulation_type

        if little_endian:
            self.endian = "<"
        else:
            self.endian = ">"

        filecount = 1
        if os.path.exists(filename):
            files = [filename]
        elif os.path.exists(filename + ".000"):
            files = [filename + ".000"]
            while os.path.exists(filename + ".%03d" % filecount):
                files += [filename + ".%03d" % filecount]
                filecount += 1
        else:
            raise FileNotFoundError(
                "Neither %s nor %s.000 exists." % (filename, filename)
            )
            return

        count = 0
        fileid = 0
        while fileid < filecount:
            if filecount > 1:
                count = count + 1
                if not quiet:
                    print("Reading file %d of %d." % (count, filecount))

            f = open(files[fileid], "rb")

            s = f.read(20)
            while len(s) > 0:
                _, name, length, _ = struct.unpack(f"{self.endian}i8sii", s)
                name = name.decode("ascii").strip()

                if name == "HEAD":
                    f.seek(4, os.SEEK_CUR)

                    # snap header
                    s = f.read(44)
                    (
                        time,
                        gnx,
                        gny,
                        gnz,
                        num_files,
                        rad_wd,
                        rad_fl,
                        idx_wd,
                        idx_fl,
                    ) = struct.unpack(f"{self.endian}fiiiiddii", s)

                    s = f.read(256 - 48)

                    self.time = time
                    self.gnx = gnx
                    self.gny = gny
                    self.gnz = gnz
                    self.ncells = gnx * gny * gnz
                    self.num_files = num_files
                    self.rad_wd = rad_wd
                    self.rad_fl = rad_fl
                    self.idx_wd = idx_wd
                    self.idx_fl = idx_fl

                    f.seek(4, os.SEEK_CUR)

                    if filecount > num_files:
                        if not quiet:
                            print(
                                "Found %d files, but need only %d files."
                                % (filecount, num_files)
                            )
                        filecount = num_files

                    if filecount < num_files:
                        if not quiet:
                            print(
                                "Need %d files, but found only %d files, stopping."
                                % (num_files, filecount)
                            )
                        f.close()
                        return

                elif name.startswith("GEOM"):
                    f.seek(4, os.SEEK_CUR)

                    label = name.lower()
                    if label not in self.data:
                        self.data[label] = np.fromfile(
                            f, dtype=f"{self.endian}f", count=length
                        )
                    else:
                        f.seek(length * 4, os.SEEK_CUR)

                    f.seek(4, os.SEEK_CUR)
                elif name.startswith("EDGE"):
                    f.seek(4, os.SEEK_CUR)

                    label = name.lower()
                    if label not in self.data:
                        self.data[label] = np.fromfile(
                            f, dtype=f"{self.endian}f", count=length
                        )
                    else:
                        f.seek(length * 4, os.SEEK_CUR)

                    f.seek(4, os.SEEK_CUR)

                else:
                    label = name.lower()
                    if label not in self.data:
                        self.data[label] = np.zeros(
                            (self.gnx, self.gny, self.gnz), dtype=f"{self.endian}f"
                        )

                    for i in range(length):
                        f.seek(4, os.SEEK_CUR)

                        s = f.read(24)
                        x0, x1, y0, y1, z0, z1 = struct.unpack(f"{self.endian}6i", s)
                        nx = x1 - x0 + 1
                        ny = y1 - y0 + 1
                        nz = z1 - z0 + 1

                        # reshape and transpose due to fortran data ordering
                        data = np.transpose(
                            np.fromfile(
                                f, dtype=f"{self.endian}f", count=nx * ny * nz
                            ).reshape((nz, ny, nx))
                        )
                        self.data[label][x0 - 1 : x1, y0 - 1 : y1, z0 - 1 : z1] = data

                        f.seek(4, os.SEEK_CUR)

                s = f.read(20)

            f.close()
            fileid += 1

        # compute edges for uniform grid if not present
        if "edgex" not in self.data:
            deltax = np.diff(self.data["geomx"])
            if np.std(deltax) / np.mean(deltax) < 1e-3:
                # almost uniform
                deltax = np.mean(deltax)
                self.data["edgex"] = np.zeros(len(self.data["geomx"]) + 1)
                self.data["edgex"][0] = self.data["geomx"][0] - deltax / 2.0
                self.data["edgex"][1:] = self.data["geomx"] + deltax / 2.0
        if "edgey" not in self.data:
            deltay = np.diff(self.data["geomy"])
            if np.std(deltay) / np.mean(deltay) < 1e-3:
                # almost uniform
                deltay = np.mean(deltay)
                self.data["edgey"] = np.zeros(len(self.data["geomy"]) + 1)
                self.data["edgey"][0] = self.data["geomy"][0] - deltay / 2.0
                self.data["edgey"][1:] = self.data["geomy"] + deltay / 2.0
        if "edgez" not in self.data:
            deltaz = np.diff(self.data["geomz"])
            if np.std(deltaz) / np.mean(deltaz) < 1e-3:
                # almost uniform
                deltaz = np.mean(deltaz)
                self.data["edgez"] = np.zeros(len(self.data["geomz"]) + 1)
                self.data["edgez"][0] = self.data["geomz"][0] - deltaz / 2.0
                self.data["edgez"][1:] = self.data["geomz"] + deltaz / 2.0

        if len(self.data["geomz"]) == 1:  # 2D
            self.vol = (
                np.diff(self.data["edgex"])[:, None, None]
                * np.diff(self.data["edgey"])[None, :, None]
            )
            self.vol[:, :, 0] = (
                self.vol[:, :, 0].T * self.data["geomx"] * 2.0 * np.pi
            ).T
        else:  # 3D
            self.vol = (
                np.diff(self.data["edgex"])[:, None, None]
                * np.diff(self.data["edgey"])[None, :, None]
                * np.diff(self.data["edgez"])[None, None, :]
            )
        self.data["vol"] = self.vol

        Amean = None
        xtot = None
        if "xnuc01" in self.data:
            Amean = (
                self.data["xnuc01"] / 12.0  # Carbon
                + self.data["xnuc02"] / 16.0  # Oxygen
                + self.data["xnuc03"] / 4.0  # alpha
                + self.data["xnuc04"] / 30.0  # medium
                + self.data["xnuc05"] / 56.0  # Nickel
            )
            xtot = (
                self.data["xnuc01"]
                + self.data["xnuc02"]
                + self.data["xnuc03"]
                + self.data["xnuc04"]
                + self.data["xnuc05"]
            )
        if "xnuc06" in self.data:
            if simulation_type == "ONeDef":
                Amean += self.data["xnuc06"] / 20.0  # Neon
            elif simulation_type == "HeDet":
                Amean += self.data["xnuc06"] / 4.0  # Helium
            else:
                raise ValueError("This simulaiton type should not have 6 species")
            xtot += self.data["xnuc06"]

        # Conflicts with the property in the parent class
        # self.mass = np.sum(self.data["density"] * self.vol)
        if "nifs" in self.data:
            self.mass_nifs = np.sum(self.data["nifs"] * self.data["density"] * self.vol)
        if "xnuc01" in self.data:
            self.mass_carb = np.sum(
                self.data["xnuc01"] * self.data["density"] * self.vol
            )
            self.mass_oxy = np.sum(
                self.data["xnuc02"] * self.data["density"] * self.vol
            )
            self.mass_alpha = np.sum(
                self.data["xnuc03"] * self.data["density"] * self.vol
            )
            self.mass_ime = np.sum(
                self.data["xnuc04"] * self.data["density"] * self.vol
            )
            self.mass_ige = np.sum(
                self.data["xnuc05"] * self.data["density"] * self.vol
            )

        if Amean is not None:
            self.data["Amean"] = xtot / Amean
        return

    def __getattr__(self, __name: str):
        """enable access via object attributes to data dict entries"""
        if __name in self.data:
            return self.data[__name]
        else:
            raise AttributeError("{} has no attribute '{}'.".format(type(self), __name))

    def convert_to_hdf5(self, filename, overwrite=False):
        """
        Convert the snapshot to an HDF5 file.
        Only works for 3D snapshots.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required to convert to HDF5")

        meta = [
            self.time,
            self.gnx,
            self.gny,
            self.gnz,
            self.ncells,
            self.rad_wd,
            self.rad_fl,
            self.idx_wd,
            self.idx_fl,
            self.simulation_type,
        ]
        meta_label = [
            "time",
            "gnx",
            "gny",
            "gnz",
            "ncells",
            "rad_wd",
            "rad_fl",
            "idx_wd",
            "idx_fl",
            "simulation_type",
        ]

        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                "File {} exists. Use overwrite=True to overwrite.".format(filename)
            )

        with h5py.File(filename, "w") as f:
            for i, label in enumerate(meta_label):
                f.attrs[label] = meta[i]

            for k in list(self.data.keys()):
                dset = f.create_dataset(
                    k, self.data[k].shape, dtype=str(self.data[k].dtype)
                )
                dset[...] = self.data[k]

        return


class LeafsProtocol:
    def __init__(self, model, snappath="./", quiet=False, simulation_type="ONeDef"):
        self.model = model
        self.snappath = snappath
        self.quiet = quiet

        assert simulation_type in [
            "CODef",
            "ONeDef",
            "HeDet",
        ], "Unrecognized simulation type"
        self.simulation_type = simulation_type

        self.keys = [
            "time",
            "mass",
            "ene",
            "ekin",
            "enuc",
            "etur",
            "egrav",
            "dpole1",
            "dpole2",
            "qpole1",
            "qpole2",
            "prod",
            "diss",
            "diff",
            "mass_xn",
            "mass_nifs",
        ]
        self.keylen = [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5, 1, 1, 1, 5, 1]

        if simulation_type == "ONeDef":
            self.keylen = [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5, 1, 1, 1, 6, 1]
            self.keys.extend(["min_ye", "max_rho"])
            self.keylen.extend([1, 1])

        self.proto = {}
        self._read_protocol()

    def _read_protocol(self):
        protocol_files = glob.glob(os.path.join(self.snappath, self.model + "*.bprot"))
        if len(protocol_files) == 0:
            raise FileNotFoundError("No protocol files found.")

        j = 99999  # Large number to find starting index of protocol files
        for filename in protocol_files:
            j = min(j, int(filename[-9:-6]))

        filename = os.path.join(self.snappath, self.model + "%03d.bprot" % j)
        while os.path.exists(filename):
            f = open(filename, "rb")

            matrix = util.readmatrix(
                f,
                sum(self.keylen),
                dtype="d",
                completerecord=True,
            )
            while isinstance(matrix, np.ndarray):
                c = 0
                for i in range(len(self.keys)):
                    key = self.keys[i]
                    if key not in self.proto:
                        if self.keylen[i] > 1:
                            self.proto[key] = np.reshape(
                                matrix[c : c + self.keylen[i]], [self.keylen[i], 1]
                            )
                        else:
                            self.proto[key] = matrix[c : c + self.keylen[i]]
                    else:
                        if self.keylen[i] > 1:
                            self.proto[key] = np.append(
                                self.proto[key],
                                np.reshape(
                                    matrix[c : c + self.keylen[i]], [self.keylen[i], 1]
                                ),
                                axis=1,
                            )
                        else:
                            self.proto[key] = np.append(
                                self.proto[key], matrix[c : c + self.keylen[i]], axis=0
                            )

                    c = c + self.keylen[i]

                matrix = util.readmatrix(
                    f, sum(self.keylen), dtype="d", completerecord=True
                )

            f.close()
            j += 1
            filename = os.path.join(self.snappath, self.model + "%03d.bprot" % j)

        if j == 0:
            raise FileNotFoundError("No protocol files found.")


class FlameHistogram:
    def __init__(self, nbins, ncrit, data_grid, data_binsum) -> None:
        self.nbins = nbins
        self.ncrit = ncrit
        self.data_bin_weights = (
            np.array(data_binsum[1:], dtype=float) / ncrit
        )  # First element is always zero
        self.data_bin_edges = np.array(data_grid)
        self.data_bin_values = 0.5 * (
            np.array(data_grid[1:]) + np.array(data_grid[:-1])
        )

    def percentile(self, p):
        if len(self.data_bin_values) == 0:
            return np.nan
        cumsum = np.cumsum(self.data_bin_weights)
        cumsum /= cumsum[-1]
        return np.interp(p, cumsum, self.data_bin_values)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.data_bin_values, bins=self.nbins, weights=self.data_bin_weights)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        return ax


class FlameProtocol:
    def __init__(self, model, snappath=".") -> None:
        time, mach_rise, v_turb_mach, v_burn_mach = np.genfromtxt(
            f"{snappath}/{model}.half", unpack=True
        )
        self.n_tsteps = len(time)
        self.time = time
        self.mach_rise_half = mach_rise
        self.v_turb_half_mach = v_turb_mach
        self.v_burn_half_mach = v_burn_mach

        self.model = model
        self.snappath = snappath
        self._loaded_data = False

        self.percentiles = None

    def _load_hstgr(self, label):
        histgrs = []
        with FortranFile(
            os.path.join(self.snappath, f"{self.model}_{label}.hstgr"), "r"
        ) as f:
            for i in tqdm(range(self.n_tsteps)):
                _ = f.read_reals("<f8")
                nb, nc = f.read_ints()
                data_grid = f.read_reals("<f8")
                data_binsum = f.read_ints("<i4")
                histgrs.append(FlameHistogram(nb, nc, data_grid, data_binsum))
        return histgrs

    def load_vburn_hstgr(self):
        print("Loading vburn histograms...")
        self.vburn_histgrs = self._load_hstgr("vburn")

    def load_vturb_hstgr(self):
        print("Loading vturb histograms...")
        self.vturb_histgrs = self._load_hstgr("vturb")

    def load_machrise_hstgr(self):
        print("Loading mach_rise histograms...")
        self.machrise_histgrs = self._load_hstgr("mach_rise")

    def load_vburn_mach_hstgr(self):
        print("Loading vburn (mach) histograms...")
        self.vburn_mach_histgrs = self._load_hstgr("vburn_mach")

    def load_vturb_mach_hstgr(self):
        print("Loading vturb (mach) histograms...")
        self.vturb_mach_histgrs = self._load_hstgr("vturb_mach")

    def load_all(self):
        self.load_vburn_hstgr()
        self.load_vturb_hstgr()
        self.load_machrise_hstgr()
        self.load_vburn_mach_hstgr()
        self.load_vturb_mach_hstgr()
        self._loaded_data = True

    def get_percentiles(self):
        if self.percentiles is not None:
            return self.percentiles

        if not self._loaded_data:
            self.load_all()
            self._loaded_data = True
        percentiles = {
            "time": self.time,
        }
        for p in [0.13, 50.0, 99.87]:
            percentiles[f"vburn_{p}"] = []
            percentiles[f"vturb_{p}"] = []
            percentiles[f"machrise_{p}"] = []
            percentiles[f"vburn_mach_{p}"] = []
            percentiles[f"vturb_mach_{p}"] = []

        for i in range(self.n_tsteps):
            for p in [0.13, 50.0, 99.87]:
                percentiles[f"vburn_{p}"].append(
                    self.vburn_histgrs[i].percentile(p / 100)
                )
                percentiles[f"vturb_{p}"].append(
                    self.vturb_histgrs[i].percentile(p / 100)
                )
                percentiles[f"machrise_{p}"].append(
                    self.machrise_histgrs[i].percentile(p / 100)
                )
                percentiles[f"vburn_mach_{p}"].append(
                    self.vburn_mach_histgrs[i].percentile(p / 100)
                )
                percentiles[f"vturb_mach_{p}"].append(
                    self.vturb_mach_histgrs[i].percentile(p / 100)
                )

        self.percentiles = pd.DataFrame(percentiles)

        return self.percentiles

    def plot_percentiles(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        percentiles = self.get_percentiles()

        ax1.plot(
            percentiles["time"],
            percentiles["vburn_mach_50.0"],
            label=r"$\langle v_{lam}\rangle$",
            ls="-.",
            color="tab:blue",
        )
        ax1.plot(
            percentiles["time"],
            percentiles["vturb_mach_50.0"],
            label=r"$\langle v_{turb}\rangle$",
            ls="-.",
            color="tab:orange",
        )
        ax1.fill_between(
            percentiles["time"],
            percentiles["vburn_mach_0.13"],
            percentiles["vburn_mach_99.87"],
            alpha=0.5,
            color="tab:blue",
        )
        ax1.fill_between(
            percentiles["time"],
            percentiles["vturb_mach_0.13"],
            percentiles["vturb_mach_99.87"],
            alpha=0.5,
            color="tab:orange",
        )

        ax2.plot(
            percentiles["time"],
            percentiles["vburn_50.0"],
            label=r"$\langle v_{lam}\rangle$",
            ls="-.",
            color="tab:blue",
        )
        ax2.plot(
            percentiles["time"],
            percentiles["vturb_50.0"],
            label=r"$\langle v_{turb}\rangle$",
            ls="-.",
            color="tab:orange",
        )
        ax2.fill_between(
            percentiles["time"],
            percentiles["vburn_0.13"],
            percentiles["vburn_99.87"],
            alpha=0.5,
            color="tab:blue",
        )
        ax2.fill_between(
            percentiles["time"],
            percentiles["vturb_0.13"],
            percentiles["vturb_99.87"],
            alpha=0.5,
            color="tab:orange",
        )

        ax1.set_ylabel(r"$v / c_s$")
        ax1.set_yscale("log")
        ax1.legend()

        ax2.set_ylabel(r"$v$ (cm s$^{-1}$)")
        ax2.set_xlabel("time")
        ax2.set_yscale("log")
        ax2.legend()

        return fig, (ax1, ax2)
