import os
import struct

import numpy as np

from leafs_clippers.util import utilities


class LeafsTracer:
    def __init__(self, model, snappath="./", npart=0, file="", vartracer=True):
        """vartracer: set to use new mass header"""

        self.name = model
        self.snappath = snappath

        # default values
        self.npart = npart
        self.mass = 0.0
        self.vartracer = vartracer

        self.files = []
        self.starttimes = []

        if len(file) > 0:
            self.nfiles = 1
            self.files = [file]

            f = open(file, "rb")
            if not self.readheader(f):
                print("File %s must not be empty." % os.path.join(snappath, self.name))
                return

            dum = f.read(4)
            if len(dum) == 4:
                (time,) = struct.unpack("<d", f.read(8))
                self.starttimes = [time]

        else:
            i = 0
            filename = os.path.join(snappath, self.name + "%03d.trace" % i)
            while os.path.exists(filename):
                f = open(filename, "rb")
                if i == 0:
                    if not self.readheader(f):
                        print("File %s must not be empty." % filename)
                        return
                print("Valid tracer file found.")

                # if file is not empty => read starttime
                dum1 = f.read(4)
                if len(dum1) == 4:
                    (time,) = struct.unpack("<d", f.read(8))

                    self.files.append(filename)
                    self.starttimes.append(time)

                f.close()
                i += 1
                filename = os.path.join(snappath, self.name + "%03d.trace" % i)

            self.nfiles = len(self.files)

        print("Starttimes: ", self.starttimes)
        return

    def readheader(self, infile):
        infile.seek(0, 0)  # start at the beginning

        header = infile.read(4)
        if len(header) != 4:
            print("Error!")
            return False

        self.headerlen = 0
        self.noHeader = False

        (headerlen,) = struct.unpack("<i", header)
        if headerlen == 12:
            (self.npart,) = struct.unpack("<i", infile.read(4))
            (self.mass,) = struct.unpack("<d", infile.read(8))
            print("%d particles of mass %g." % (self.npart, self.mass))
        elif headerlen == 4:
            (self.npart,) = struct.unpack("<i", infile.read(4))
            print("%d particles." % self.npart)
        else:
            print("Tracerfile has no header.")
            self.noHeader = True

        if self.noHeader:
            # get nvalues
            self.nvalues = np.int32((headerlen - 8) / (4 * self.npart) + 1)
            print("nvalues of %d detected." % self.nvalues)
            infile.seek(-len(header), 1)
        else:
            _ = infile.read(4)

            self.headerlen += headerlen + 8

            # look for variable mass tracers
            header = infile.read(4)
            if len(header) == 4:
                (headerlen,) = struct.unpack("<i", header)
                if headerlen == 0:
                    self.npart = 1000000
                    self.headerlen += 8
                    _ = infile.read(4)
                elif headerlen == self.npart * 8:
                    self.tmass = np.fromfile(infile, dtype="float64", count=self.npart)
                    _ = infile.read(4)
                    self.vartracer = True
                    print("Tracer particles have variable masses.")
                    print("Mass min/max: %g/%g" % (self.tmass.min(), self.tmass.max()))

                    self.headerlen += headerlen + 8
                else:
                    infile.seek(-len(header), 1)  # go back

            # get nvalues
            header = infile.read(4)
            if len(header) == 4:
                (headerlen,) = struct.unpack("<i", header)
                self.nvalues = int((headerlen - 8) / (4 * self.npart) + 1)
                print(
                    "nvalues of %d detected, headerlen=%d, fpos=%d, npart=%d."
                    % (self.nvalues, headerlen, infile.tell(), self.npart)
                )
            infile.seek(-len(header), 1)  # go back

        return True

    def initial(self, two_d=False):
        """set two_d to avoid getting wrong dictionary entries

        quantities: time, xpos, ypos, rho, tmp, ene, tye, txn, tbd (burning
                    density), tls

        unbound star diag: time, xpos, ypos, rho, tmp, ene, tye, txn, tbd,
                           tls, tgpot, vx, vy"""

        f = open(self.files[0], "rb")
        f.seek(self.headerlen, 0)

        _ = f.read(4)
        (time,) = struct.unpack("<d", f.read(8))
        data = np.fromfile(
            f, dtype="float32", count=self.npart * (self.nvalues - 1)
        ).reshape(self.nvalues - 1, self.npart)

        if self.nvalues == 7 and not two_d:
            # incorrect for two_d case as pos has only two dimensions then
            return utilities.dict2obj(
                {
                    "pos": data[0:3, :],
                    "rho": data[3, :],
                    "temp": data[4, :],
                    "ene": data[5, :],
                    "nvalues": self.nvalues,
                    "time": time,
                    "data": data,
                }
            )
        else:
            return utilities.dict2obj(
                {"data": data, "nvalues": self.nvalues, "time": time}
            )


class LeafsTracerUtil(object):
    """Utility class for leafs_tracer

    This helper class interprets the tracer_block returned by leafs_tracer
    access methods according leafs/code/tracer/tracer2d.F90. For this purpose,
    it is important that all parameters determining the size and the contents
    of the tracer data block are properly provided.

    Parameters
    ----------
    tracer_block : object
        tracer data block, as returned by e.g. leafs_tracer.initial,
        leafs_tracer.final, etc.
    threed : bool
        is the tracer block from a three-dimensional run (default True)
    unbound_star_diag : bool
        is the tracer block from a run with unbound star diagnostics
        (default False)
    ext_tracer : bool
        is the tracer block from a run with extended tracer output
        (default True)
    qn : int
        number of reduced species (default 6)
    nlsets : int
        number of level sets (default 1)

    """

    def __init__(
        self,
        tracer_block,
        threed=True,
        unbound_star_diag=False,
        qn=6,
        nlsets=1,
        ext_tracer=True,
        simulation_type="ONeDef",
    ):
        assert simulation_type in [
            "CODef",
            "ONeDef",
            "HeDet",
        ], "Unrecognized simulation type"
        self._unbound_star_diag = unbound_star_diag
        self._qn = qn
        self._nlsets = nlsets
        self._threed = threed
        self._ext_tracer = ext_tracer
        self._simulation_type = simulation_type

        try:
            self._raw_data = tracer_block.data
            self._time = tracer_block.time
        except AttributeError:
            print(
                "Tracer block does not have expected form; "
                "provide object returned from leafs_tracer.last(), "
                "leafs_tracer.initial() or leafs_tracer.attime()"
            )
            raise ValueError

        self._extract()

    @property
    def unbound_star_diag(self):
        return self._unbound_star_diag

    @property
    def qn(self):
        return self._qn

    @property
    def nlsets(self):
        return self._nlsets

    @property
    def threed(self):
        return self._threed

    @property
    def ext_tracer(self):
        return self._ext_tracer

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def time(self):
        return self._time

    @property
    def Xr_C(self):
        """reduced carbon abundance"""
        return self.X[0, :]

    @property
    def Xr_O(self):
        """reduced oxygen abundance"""
        return self.X[1, :]

    @property
    def Xr_alpha(self):
        """reduced alpha abundance"""
        return self.X[2, :]

    @property
    def Xr_ime(self):
        """reduced IME abundance"""
        return self.X[3, :]

    @property
    def Xr_IGE(self):
        """reduced IGE abundance"""
        return self.X[4, :]

    @property
    def Xr_Add(self):
        """reduced helium or neon abundance"""
        if self.qn == 6:
            return self.X[5, :]
        else:
            return None

    def _extract(self):
        """Parse tracer block. The parsing process follows the storage scheme
        laid out in tracer2d.F90/ tracer3d.F90"""
        if self.ext_tracer:
            expected_nvals = 7 + self.qn + 2 * self.nlsets
            if self.threed:
                expected_nvals += 1
            if self.unbound_star_diag:
                expected_nvals += 3
                if self.threed:
                    expected_nvals += 1
        else:
            if self.threed:
                expected_nvals = 7
            else:
                expected_nvals = 6

        if self.raw_data.shape[0] != expected_nvals:
            print(
                "tracer block stores a different number of properties than "
                "expected for the setup: "
            )
            print("3D :: ", self.threed)
            print("ext_tracer :: ", self.ext_tracer)
            print("Unbound Star Diagnostics :: ", self.unbound_star_diag)
            print("# of level sets (nlsets) :: ", self.nlsets)
            print("# of species (qn) :: ", self.qn)
            print("# of values :: ", self.raw_data.shape[0])
            print("# of expected values :: ", expected_nvals)
            print("Check that these choices are appropriate for this run")

            raise ValueError

        if self.threed:
            k = 1
        else:
            k = 0

        self.x = self.raw_data[0, :]
        self.y = self.raw_data[1, :]
        if self.threed:
            self.z = self.raw_data[2, :]
        self.rho = self.raw_data[k + 2, :]
        self.T = self.raw_data[k + 3, :]
        self.e = self.raw_data[k + 4, :]
        ind = k + 4

        if self.ext_tracer:
            self.Ye = self.raw_data[k + 5, :]

            self.X = self.raw_data[k + 6 : k + 6 + self.qn, :]
            # has burndens been added recently to tracer writeout? If so, this
            # has to be caught and handled separately to ensure backwards
            # compatibility
            self.burndens = self.raw_data[
                k + 6 + self.qn : k + 6 + self.qn + self.nlsets, :
            ]
            self.lset = self.raw_data[
                k + 6 + self.qn + self.nlsets : k + 6 + self.qn + 2 * self.nlsets, :
            ]
            ind = k + 6 + self.qn + 2 * self.nlsets
            if self.unbound_star_diag:
                self.gpot = self.raw_data[ind, :]
                self.vx = self.raw_data[ind + 1, :]
                self.vy = self.raw_data[ind + 2, :]
                ind = ind + 3
                if not self.threed:
                    self.vz = self.raw_data[ind, :]
                    ind += 1
