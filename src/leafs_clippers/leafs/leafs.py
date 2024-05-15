import os
import re
import glob
import struct

import h5py
from tqdm import tqdm
import numpy as np
from scipy.stats import binned_statistic

try:
    from singularity_eos import Helmholtz
except ImportError:
    pass

from leafs_clippers.util import utilities as util


def get_snaplist(model, snappath="./", legacy=False):
    # check parallel files first
    if legacy:
        outfiles = glob.glob(model + "o[0-9][0-9][0-9].000", root_dir=snappath)
        # check output from serial run if nothing was found
        if len(outfiles) == 0:
            outfiles = glob.glob(model + "o[0-9][0-9][0-9]", root_dir=snappath)
    else:
        outfiles = glob.glob(model + "o[0-9][0-9][0-9].hdf5", root_dir=snappath)
    snaplist = []
    for name in outfiles:
        snaplist.append(int(re.sub(model + r"o([0-9]{3}).*", r"\1", name)))

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
):
    if legacy:
        return LeafsLegacySnapshot(
            os.path.join(snappath, "{:s}o{:03d}".format(model, int(ind))),
            simulation_type=simulation_type,
            quiet=quiet,
            little_endian=little_endian,
        )
    else:
        return LeafsSnapshot(
            os.path.join(snappath, "{:s}o{:03d}.hdf5".format(model, int(ind))),
            quiet=quiet,
            helm_table="helm_table.dat",
            write_derived=write_derived,
            ignore_cache=ignore_cache,
        )


class LeafsSnapshot:
    def __init__(
        self,
        filename,
        quiet=False,
        helm_table="helm_table.dat",
        write_derived=True,
        ignore_cache=False,
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
        """
        self.quiet = quiet
        self.filename = filename
        self.basename = filename.replace(".hdf5", "")
        self.helm_table = helm_table
        self.write_derived = write_derived
        self.ignore_cache = ignore_cache

        try:
            f = h5py.File(filename, "r")
        except FileNotFoundError:
            raise FileNotFoundError("File {} not found.".format(filename))

        try:
            if os.path.exists(helm_table):
                self.eos = Helmholtz(helm_table)
            else:
                print("No EOS table found, setting eos to None.")
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
        else:
            raise AttributeError("{} has no attribute '{}'.".format(type(self), __name))

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
        with h5py.File(cache_filename, "a") as f:
            if field not in f:
                f.create_dataset(field, data=self.data[field])
            else:
                f[field][...] = self.data[field]

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
                    # Use arbitrary abar = 16, only needed to recalculate Ye internally
                    abar = 16
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

    def get_bound_material(
        self, include_internal_energy=True, eint_from_eos=False, vacuum_threshold=1e-5
    ):
        """
        Returns a boolean mask for the bound material.
        If include_internal_energy is True, the internal energy
        is included in the bound criterion.

        Bound material is defined as material with a negative
        total energy:
            E_kin + E_grav + E_int < 0
        """

        if not self.ignore_cache:
            if self._load_derived("bound_mask"):
                return self.data["bound_mask"]

        if eint_from_eos:
            self.get_internal_energy_from_eos()

        # Note that quantities are stored per unit mass
        total_energy = (self.egrav + self.ekin) * self.mass
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

    def get_bound_mass(self, include_internal_energy=True, eint_from_eos=False):
        """
        Returns the mass of the bound material.
        If include_internal_energy is True, the internal energy
        is included in the bound criterion.

        Bound material is defined as material with a negative
        total energy:
            E_kin + E_grav + E_int < 0
        """

        bound_mask = self.get_bound_material(
            include_internal_energy, eint_from_eos=eint_from_eos
        )
        return np.sum(self.density[bound_mask] * self.vol[bound_mask])

    def get_central_value(self, value):
        """
        Returns the central value of a quantity.
        """
        assert isinstance(value, str), "Value must be a string"
        mid = int(self.gnx / 2)
        return np.mean(
            self.data[value][mid - 1 : mid + 2, mid - 1 : mid + 2, mid - 1 : mid + 2]
        )

    def get_rad_profile(self, value, res=None, statistic="mean"):
        """
        Returns the (1D) radial averaged profile of a quantity.
        Uses binned_statistic to bin the data.
        If res is not None, the data will be binned into original resolution.

        Parameters
        ----------
        value : str
            The name of the quantity to bin.
        res : int, optional
            The resolution to bin the data into.
        statistic : str, optional
            The statistic to compute in each bin. Passed to binned_statistic.

        Returns
        -------
        bin_centers : np.ndarray
            The bin centers.
        bin_values : np.ndarray
            The binned values.
        """
        assert isinstance(value, str), "Value must be a string"

        xx, yy, zz = np.meshgrid(self.geomx, self.geomy, self.geomz)
        r = np.sqrt(xx**2 + yy**2 + zz**2)

        r_flat = r.flatten()
        value_flat = self.data[value].flatten()

        # Sort by radius
        idx = np.argsort(r_flat)
        r_sorted = r_flat[idx]
        value_sorted = value_flat[idx]

        # Bin back into original resolution
        if res is None:
            res = len(self.geomx) // 2

        bin_values, bin_edges, _ = binned_statistic(
            r_sorted, value_sorted, bins=res, statistic=statistic
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, bin_values


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
                        self.data[label] = np.fromfile(f, dtype="f", count=length)
                    else:
                        f.seek(length * 4, os.SEEK_CUR)

                    f.seek(4, os.SEEK_CUR)
                elif name.startswith("EDGE"):
                    f.seek(4, os.SEEK_CUR)

                    label = name.lower()
                    if label not in self.data:
                        self.data[label] = np.fromfile(f, dtype="f", count=length)
                    else:
                        f.seek(length * 4, os.SEEK_CUR)

                    f.seek(4, os.SEEK_CUR)

                else:
                    label = name.lower()
                    if label not in self.data:
                        self.data[label] = np.zeros(
                            (self.gnx, self.gny, self.gnz), dtype="f"
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
                            np.fromfile(f, dtype="f", count=nx * ny * nz).reshape(
                                (nz, ny, nx)
                            )
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
            # self.keys.extend(["min_ye", "max_rho"])
            # self.keylen.extend([1, 1])

        self.proto = {}
        self._read_protocol()

    def _read_protocol(self):
        j = 0
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
