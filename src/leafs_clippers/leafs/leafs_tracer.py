import os
import struct

import numpy as np
from tqdm import tqdm

from leafs_clippers import util
from leafs_clippers.util import utilities


def read_tracer(model, snappath="./", npart=0, file="", vartracer=True):
    """Read tracer files

    Parameters
    ----------
    model : str
        model name
    snappath : str
        path to snapshot files
    npart : int
        number of particles
    file : str
        name of the tracer file
    vartracer : bool
        set to use new mass header

    Returns
    -------
    LeafsTracer
        tracer object

    """
    return LeafsTracer(model, snappath, npart, file, vartracer)


def get_bound_unbound_ids(
    model="one_def",
    snappath="./output",
    ignore_cache=False,
    remnant_velocity=[0, 0, 0],
    qn=6,
    nlset=1,
    writeout=True,
):
    if not ignore_cache:
        try:
            unbound = np.genfromtxt(f"{snappath}/unbound.txt", dtype=int)
            bound = np.genfromtxt(f"{snappath}/bound.txt", dtype=int)
            print("Found bound and unbound particles ids in output directory...")
            return bound, unbound
        except FileNotFoundError:
            pass

    rx, ry, rz = remnant_velocity

    print("Calculating bound and unbound particles ids...")

    tp = read_tracer(model=model, snappath=snappath)
    at = tp.last()

    bound = []
    unbound = []
    off = 1 + 6 + qn + 2 * nlset

    for i in tqdm(range(tp.npart)):
        idx = i + 1
        eint = at.data[5, i]
        egrav = at.data[off, i]
        ekin = 0.5 * (
            (at.data[off + 1, i] - rx) ** 2
            + (at.data[off + 2, i] - ry) ** 2
            + (at.data[off + 3, i] - rz) ** 2
        )
        etot = eint + egrav + ekin
        if etot >= 0:
            unbound.append(idx)
        if etot < 0:
            bound.append(idx)

    if writeout:
        np.savetxt(f"{snappath}/unbound.txt", unbound, fmt="%d")
        np.savetxt(f"{snappath}/bound.txt", bound, fmt="%d")

    print("All done.")

    return np.array(bound), np.array(unbound)


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
            assert self.nfiles > 0, "No tracer files found!"

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
                {
                    "data": data,
                    "nvalues": self.nvalues,
                    "time": time,
                }
            )

    def get_times(self, quiet=False):
        if not quiet:
            print("Getting the number of tracer timesteps")
        ntimesteps = 0
        times = []
        for i in range(self.nfiles):
            if not quiet:
                print("Doing file " + self.files[i])

            f = open(self.files[i], "rb")
            if i == 0:
                f.seek(self.headerlen, 0)

            dum1 = f.read(4)
            while len(dum1) > 0:
                (t,) = struct.unpack("<d", f.read(8))
                times.append(t)
                f.seek(int(4 * self.npart * (self.nvalues - 1)), 1)

                if (i == self.nfiles - 1) or (t < self.starttimes[i + 1]):
                    ntimesteps += 1
                else:
                    break

                if ntimesteps % 10000 == 0 and not quiet:
                    print("Timesteps currently: %d" % ntimesteps)

                _ = f.read(4)
                dum1 = f.read(4)
            f.close()

        if not quiet:
            print("Total number of timesteps: ", len(times))
            print("Time runs from ", times[0], " to ", times[-1])
        return np.array(times)

    def attimestep(self, tstp, two_d=False, usefile=None, quiet=False):
        # set two_d to avoid getting wrong dictionary entries

        index = tstp
        if usefile is None:
            for fileid in range(self.nfiles):
                count = 0
                f = open(self.files[fileid], "rb")
                if fileid == 0:
                    f.seek(self.headerlen, 0)

                dum1 = f.read(4)
                while len(dum1) > 0:
                    (time,) = struct.unpack("<d", f.read(8))
                    f.seek(4 * self.npart * (self.nvalues - 1), 1)

                    if (fileid == self.nfiles - 1) or (
                        time < self.starttimes[fileid + 1]
                    ):
                        count += 1
                    else:
                        break

                    _ = f.read(4)
                    dum1 = f.read(4)
                f.close()

                if index >= count:
                    index -= count
                else:
                    break
        else:
            fileid = usefile

        f = open(self.files[fileid], "rb")
        if fileid == 0:
            f.seek(self.headerlen, 0)

        f.seek((self.npart * (self.nvalues - 1) * 4 + 8 + 8) * index, 1)

        _ = f.read(4)
        (ttime,) = struct.unpack("<d", f.read(8))

        if not quiet:
            print(
                "Loading data of timestep %d at index %d of file %d at time %g."
                % (tstp, index, fileid, ttime)
            )

        data = np.fromfile(
            f, dtype="float32", count=self.npart * (self.nvalues - 1)
        ).reshape(self.nvalues - 1, self.npart)
        f.close()

        if self.nvalues == 7 and not two_d:
            # incorrect for two_d case as pos has only two dimensions then
            return utilities.dict2obj(
                {
                    "pos": data[0:3, :],
                    "rho": data[3, :],
                    "temp": data[4, :],
                    "ene": data[5, :],
                    "nvalues": self.nvalues,
                    "time": ttime,
                }
            )
        else:
            return utilities.dict2obj(
                {
                    "data": data,
                    "nvalues": self.nvalues,
                    "time": ttime,
                }
            )

        return False

    def attime(self, time, two_d=False):
        # set two_d to avoid getting wrong dictionary entries

        fileid = 0
        while (fileid + 1 < self.nfiles) and (self.starttimes[fileid + 1] < time):
            fileid += 1

        f = open(self.files[fileid], "rb")
        if fileid == 0:
            f.seek(self.headerlen, 0)

        header = f.read(4)
        while len(header) == 4:
            (ttime,) = struct.unpack("<d", f.read(8))
            if ttime >= time:
                print("Loading data at time %g." % ttime)
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
                        }
                    )
                else:
                    return utilities.dict2obj({"data": data, "nvalues": self.nvalues})

            else:
                f.seek(self.npart * (self.nvalues - 1) * 4 + 4, 1)

            header = f.read(4)

        return False

    def get_timestep(
        self,
        tstep,
        chunksize=None,
        read_count=0,
        usefile=None,
        threed=True,
        quiet=False,
    ):
        index = tstep
        fileid = 0
        if usefile is None:
            for fileid in range(self.nfiles):
                count = 0
                f = open(self.files[fileid], "rb")
                if fileid == 0:
                    f.seek(self.headerlen, 0)

                dum1 = f.read(4)
                while len(dum1) > 0:
                    (time,) = struct.unpack("<d", f.read(8))
                    f.seek(int(4 * self.npart * (self.nvalues - 1)), 1)
                    if (fileid == self.nfiles - 1) or (
                        time < self.starttimes[fileid + 1]
                    ):
                        count += 1
                    else:
                        break

                    _ = f.read(4)
                    dum1 = f.read(4)
                f.close()

                if index >= count:
                    index -= count
                else:
                    break
        else:
            fileid = usefile

        f = open(self.files[fileid], "rb")
        if fileid == 0:
            f.seek(self.headerlen, 0)

        f.seek((self.npart * (self.nvalues - 1) * 4 + 8 + 8) * index, 1)

        _ = f.read(4)
        (ttime,) = struct.unpack("<d", f.read(8))

        if not quiet:
            print(
                "Loading data of timestep %d at index %d of file %d at time %g."
                % (tstep, index, fileid, ttime)
            )
        if chunksize != self.npart and chunksize is not None:
            assert threed, "Chunking only works in 3D!"
            assert chunksize + read_count <= self.npart, (
                "Chunsize bigger than number of particles!"
            )
            data = np.zeros((chunksize, 6))
            offset = f.tell()
            for i in range(6):
                f.seek(offset + (self.npart * 4 * i) + read_count * 4, 0)
                data[:, i] = np.fromfile(f, dtype="float32", count=chunksize)
            data = data.T
        else:
            data = np.fromfile(
                f, dtype="float32", count=self.npart * (self.nvalues - 1)
            ).reshape(self.nvalues - 1, self.npart)
        f.close()

        if threed:
            return data[:6, :], ttime
        else:
            return data[:5, :], ttime

    def mergefiles(self, timeoffsets=[], outname=None):
        if len(timeoffsets) == 0:
            timeoffsets = np.zeros(self.nfiles)
        else:
            assert len(timeoffsets) != self.nfiles, (
                "timeoffsets {0} has to be of the same size as the number of files {1}.".format(
                    len(timeoffsets), self.nfiles
                )
            )

        if not outname:
            outname = "{0}_merged000.trace".format(self.name)

        fout = open(outname, "wb")

        for fid in range(self.nfiles):
            fin = open(self.files[fid], "rb")

            if fid == 0:
                fout.write(fin.read(self.headerlen))

            datastart = fin.tell()
            fin.seek(0, 2)  # end
            dataend = fin.tell()
            fin.seek(datastart, 0)

            blocksize = self.npart * (self.nvalues - 1) * 4 + 16
            blockcount = int((dataend - datastart) / blocksize)

            if (dataend - datastart) % blocksize != 0:
                print(
                    "Tracer file {0} is inconsistent: headersize={1}, blocksize={2}, modulo={3}.".format(
                        fid, datastart, blocksize, (dataend - datastart) % blocksize
                    )
                )
                fin.close()
                fout.close()
                raise ValueError()

            useblock = np.zeros(blockcount, dtype=np.bool)
            blocktimes = np.zeros(blockcount)

            count = 0

            dum1 = fin.read(4)
            while len(dum1) > 0:
                (time,) = struct.unpack("d", fin.read(8))
                time += timeoffsets[fid]

                if fid < self.nfiles - 1:
                    if time >= self.starttimes[fid + 1] + timeoffsets[fid + 1]:
                        break

                blocktimes[count] = time
                useblock[count] = True

                backiter = count - 1
                while backiter >= 0:
                    if time <= blocktimes[backiter]:
                        useblock[backiter] = False
                        backiter -= 1
                    else:
                        break

                count += 1
                fin.seek(int(self.npart * (self.nvalues - 1) * 4 + 4), 1)
                dum1 = fin.read(4)

            count = 0
            for iblock in range(blockcount):
                if useblock[iblock]:
                    fin.seek(int(datastart + blocksize * iblock))

                    block = fin.read(int(blocksize))
                    block = (
                        block[:4] + struct.pack("d", blocktimes[iblock]) + block[12:]
                    )

                    if len(block) != blocksize:
                        print(
                            "Fail: len(block)={0}, blocksize={1}.".format(
                                len(block), blocksize
                            )
                        )
                        fin.close()
                        fout.close()
                        raise ValueError()

                    fout.write(block)

            fin.close()

        fout.close()
        return outname

    def last(self, two_d=False, countfromend=1):
        """set two_d to avoid getting wrong dictionary entries
        use countfromend to access timesteps before the last one"""

        f = open(self.files[-1], "rb")
        f.seek(0, 2)  # seek end of file

        f.seek(
            -4 - self.npart * (self.nvalues - 1) * 4 - 8, 1
        )  # go back one dataset from current position
        for i in range(countfromend - 1):
            f.seek(
                -4 - 4 - self.npart * (self.nvalues - 1) * 4 - 8, 1
            )  # go back one dataset from current position

        (time,) = struct.unpack("<d", f.read(8))
        print("Time: ", time)
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
                {
                    "data": data,
                    "nvalues": self.nvalues,
                    "time": time,
                }
            )

    def count_timesteps(self, quiet=False):
        times = self.get_times(quiet=quiet)
        timestepcount = len(times)

        if not quiet:
            print("Found %d timesteps." % timestepcount)
        return timestepcount

    def loadalltracers(self, two_d=False, quiet=False, stride=1):
        """
        Load all data from all tracers - CAUTION: takes a lot of memory!

        Parameters
        ----------
        two_d : bool
            set to avoid getting wrong dictionary entries
        quiet : bool
            set to suppress output
        stride : int
            read only every stride-th tracer

        Returns
        -------
        data : dict2obj
            data array
            time, xpos, ypos, zpos, rho, tmp, ene, tye, txn, tbd, tls, tgpot, vx, vy, vz
        """
        if not quiet:
            print("npart: %d, nvalues: %d" % (self.npart, self.nvalues))

        timestepcount = 0

        nsteps = self.count_timesteps(quiet=quiet)
        if stride > 1:
            nparts = self.npart // stride + 1
        else:
            nparts = self.npart
        values = np.zeros((nsteps, nparts, self.nvalues), dtype="float32")

        for i in range(self.nfiles):
            if not quiet:
                print("Doing file " + self.files[i])
            f = open(self.files[i], "rb")

            if i == 0:
                f.seek(self.headerlen, 0)

            dum1 = f.read(4)
            while len(dum1) > 0:
                (time,) = struct.unpack("<d", f.read(8))

                if (i == self.nfiles - 1) or (time < self.starttimes[i + 1]):
                    values[timestepcount, :, 0] = time
                    data = np.fromfile(
                        f, dtype="float32", count=self.npart * (self.nvalues - 1)
                    ).reshape(self.nvalues - 1, self.npart)
                    values[timestepcount, :, 1:] = data[:, ::stride].T

                    timestepcount += 1
                else:
                    break

                _ = f.read(4)
                dum1 = f.read(4)

            f.close()

        values = values.astype("float64")

        if self.nvalues == 7 and not two_d:
            # incorrect for two_d case as pos has only two dimensions then
            return utilities.dict2obj(
                {
                    "time": values[:, :, 0],
                    "pos": values[:, :, 1:4],
                    "rho": values[:, :, 4],
                    "temp": values[:, :, 5],
                    "ene": values[:, :, 6],
                    "tsteps": timestepcount,
                    "nvalues": self.nvalues,
                }
            )
        else:
            return utilities.dict2obj(
                {
                    "time": values[:, :, 0],
                    "data": values[:, :, 1:],
                    "tsteps": timestepcount,
                    "nvalues": self.nvalues,
                }
            )

    def loadtracer(
        self, id, two_d=False, quiet=False, nsteps=None, return_float64=False
    ):
        """
        Load data from a single tracer

        Parameters
        ----------
        id : int or list of int
            ID(s) of the tracer(s) to be loaded
        two_d : bool
            set to avoid getting wrong dictionary entries
        quiet : bool
            set to suppress output
        nsteps : None or int
            If given, preallocate for nsteps timesteps, otherwise
            all files have to be scanned which can take some time.
        return_float64 : bool
            If True, return data as float64 (default False)

        Returns
        -------
        data : dict2obj
            data array
            time, xpos, ypos, zpos, rho, tmp, ene, tye, txn, tbd, tls, tgpot, vx, vy, vz

        """
        # set two_d to avoid getting wrong dictionary entries

        if isinstance(id, int):
            id = np.array([id])

        npart = len(id)

        if not quiet:
            print("npart: %d, nvalues: %d" % (self.npart, self.nvalues))
            print(f"Getting {npart} tracers...")
            print("IDs: ", id)

        timestepcount = 0

        if nsteps is None:
            nsteps = self.count_timesteps(quiet=quiet)
        values = np.zeros((nsteps, npart, self.nvalues), dtype="float32")

        for i in range(self.nfiles):
            if not quiet:
                print("Doing file " + self.files[i])
            f = open(self.files[i], "rb")

            if i == 0:
                f.seek(self.headerlen, 0)

            dum1 = f.read(4)
            while len(dum1) > 0:
                (time,) = struct.unpack("<d", f.read(8))

                if (i == self.nfiles - 1) or (time < self.starttimes[i + 1]):
                    values[timestepcount, :, 0] = time
                    data = np.fromfile(
                        f, dtype="float32", count=self.npart * (self.nvalues - 1)
                    ).reshape(self.nvalues - 1, self.npart)
                    values[timestepcount, :, 1:] = data[:, id].T

                    timestepcount += 1
                else:
                    break

                _ = f.read(4)
                dum1 = f.read(4)

            f.close()

        if return_float64:
            values = values.astype("float64")

        if self.nvalues == 7 and not two_d:
            # incorrect for two_d case as pos has only two dimensions then
            return utilities.dict2obj(
                {
                    "time": values[:timestepcount, :, 0],
                    "pos": values[:timestepcount, :, 1:4],
                    "rho": values[:timestepcount, :, 4],
                    "temp": values[:timestepcount, :, 5],
                    "ene": values[:timestepcount, :, 6],
                    "tsteps": timestepcount,
                    "nvalues": self.nvalues,
                }
            )
        else:
            return utilities.dict2obj(
                {
                    "time": values[:timestepcount, :, 0],
                    "data": values[:timestepcount, :, 1:],
                    "tsteps": timestepcount,
                    "nvalues": self.nvalues,
                }
            )

    def extracttracers(
        self,
        npart,
        startid=None,
        mask=None,
        noutvalues=0,
        writeHeader=False,
        outfile="",
        nstep=1,
        endAtTime=-1,
    ):
        """
        Read tracer files, extract certain tracers and write them to a new file suitable for
        postprocessing.

        Parameters
        ----------

        npart : int
            Number of tracers to be written
        startid : int
            First tracer to be written
        mask : list of bool
            Provide a bool mask to specifiy tracers to extract
        noutvalues : int
            Number of values to be written per tracer (including time).
            If 0, write first 7 values (time, xpos, ypos, zpos, rho, tmp, ene).
        writeHeader : bool
            Write header to output file
        outfile : str
            Name of output file
        nstep : int
            How many timesteps to skip when writing tracers
        endAtTime : float
            Stop writing tracers at this time (default: -1, i.e. write all timesteps)
        """

        if startid is None and mask is None:
            raise ValueError("Either startid or mask has to be provided!")

        if noutvalues == 0:
            noutvalues = 7

        print("noutvalues set to %d." % noutvalues)

        if nstep > 1 and self.tmass.min() != self.tmass.max():
            raise ValueError(
                "This only works for equal mass tracers."
                f" Found min/max mass: {self.tmass.min()}/{self.tmass.max()}."
            )

        timestepcount = 0

        if outfile == "":
            fout = open(self.snappath + self.name + ".trace", "wb")
        else:
            fout = open(outfile, "wb")

        if writeHeader:
            fout.write(struct.pack("<iii", 4, npart // nstep, 4))
            if self.vartracer:
                fout.write(struct.pack("<i", 8 * npart // nstep))
                if mask is None:
                    (self.tmass[startid : startid + npart : nstep] * nstep).tofile(fout)
                else:
                    (self.tmass[mask] * nstep).tofile(fout)
                fout.write(struct.pack("<i", 8 * npart // nstep))

        for i in range(self.nfiles):
            print("Doing file " + self.files[i])

            f = open(self.files[i], "rb")
            if i == 0:
                f.seek(self.headerlen, 0)

            dum1 = f.read(4)
            while len(dum1) > 0:
                (time,) = struct.unpack("<d", f.read(8))

                if (endAtTime >= 0) and (time > endAtTime):
                    break

                data = np.fromfile(
                    f, dtype="float32", count=self.npart * (self.nvalues - 1)
                ).reshape(self.nvalues - 1, self.npart)

                if (i == self.nfiles - 1) or (time < self.starttimes[i + 1]):
                    fout.write(
                        struct.pack(
                            "<i",
                            (8 + npart // nstep * (noutvalues - 1) * 4),
                        )
                    )
                    fout.write(struct.pack("<d", (time)))
                    if mask is None:
                        data[
                            : noutvalues - 1, startid : startid + npart : nstep
                        ].tofile(fout)
                    else:
                        data[: noutvalues - 1, mask].tofile(fout)
                    fout.write(
                        struct.pack(
                            "<i",
                            (8 + npart // nstep * (noutvalues - 1) * 4),
                        )
                    )

                    timestepcount += 1
                else:
                    break

                _ = f.read(4)

                dum1 = f.read(4)

            f.close()

        fout.close()

        print("Wrote %d timesteps." % timestepcount)
        return


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
                if self.threed:
                    self.vz = self.raw_data[ind, :]
                    ind += 1

        self.esc = self.raw_data[ind, :]
