import os
import re
import glob

import h5py
import numpy as np

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
    ind, model, snappath="./", simulation_type="ONeDef", quiet=False, legacy=False
):
    if legacy:
        return LeafsLegacySnapshot(
            os.path.join(snappath, "{:s}o{:03d}".format(model, int(ind))),
            simulation_type=simulation_type,
            quiet=quiet,
        )
    else:
        return LeafsSnapshot(
            os.path.join(snappath, "{:s}o{:03d}".format(model, int(ind))),
            quiet=quiet,
        )


class LeafsSnapshot:
    def __init__(self, filename, quiet=False):
        self.quiet = quiet
        self.filename = filename
        f = h5py.File(filename, "r")

        # Read meta data
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

        f.close()
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


class LeafsLegacySnapshot(LeafsSnapshot):
    def __init__(self, filename, simulation_type="ONeDef", quiet=False):
        self.data = {}
        assert simulation_type in [
            "CODef",
            "ONeDef",
            "HeDet",
        ], "Unrecognized simulation type"
        self.simulation_type = simulation_type

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
                _, name, length, _ = struct.unpack("<i8sii", s)
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
                    ) = struct.unpack("<fiiiiddii", s)

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
                        x0, x1, y0, y1, z0, z1 = struct.unpack("<6i", s)
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

        self.mass = np.sum(self.data["density"] * self.vol)
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
