import os
import gc
import time
import struct
import argparse

import numpy as np

from leafs_clippers.util.const import M_SOL
from leafs_clippers import leafs_tracer


def main(model, outfile, snappath="./", chunksize=2.5, verbose=False):
    fin = leafs_tracer.LeafsTracer(model, snappath=snappath)

    # Check if file is in one single file
    # If not, merge into one file and transpose that file
    if fin.nfiles > 1:
        print("Multiple tracer files detected, merging them into one file...")
        if os.path.exists(os.path.join(snappath, "%s_merged000.trace" % model)):
            print("Merged file already exists, loading existing one...")
            mergedfile = "%s_merged000.trace" % model
        else:
            mergedfile = fin.mergefiles()
        fin = leafs_tracer.LeafsTracer(model, snappath=snappath, file=mergedfile)

    fout = open(outfile, "wb")

    ntracer = fin.npart
    masses = fin.tmass

    print("NTracer: ", ntracer)
    print("Total mass: ", masses.sum() / M_SOL)
    # Find the number of timesteps
    start = time.time()
    times = fin.get_times()
    ntimesteps = len(times)
    print("Getting timesteps took %ds" % (time.time() - start))
    print("Found %d tracers with %d timesteps" % (ntracer, ntimesteps))

    fout.write(struct.pack("iiiii", ntracer, 1, ntracer, ntimesteps, 6))
    masses.tofile(fout)
    times.tofile(fout)

    # Set buffer size
    read_chunk = int((chunksize * 8e9) // (6 * ntimesteps * 10 * 8))
    print("Chunksize = %.1fGB" % chunksize)
    print("Reading ", read_chunk, " tracers per chunk.")

    read_count = 0
    while read_count < ntracer:
        read_chunk = min(read_chunk, ntracer - read_count)

        data = np.zeros((read_chunk, ntimesteps, 6), dtype="f4")
        start = time.time()
        print("Reading chunk size %d starting from %d" % (read_chunk, read_count))

        for itstp in range(ntimesteps):
            if itstp % 1000 == 0:
                print("Now reading timestep %d/%d" % (itstp, ntimesteps))

            chunk, _ = fin.get_timestep(
                itstp, read_count=read_count, chunksize=read_chunk, quiet=verbose
            )
            for ival in range(6):
                data[:, itstp, ival] = chunk[ival, :]

        print("Writing chunk size %d starting from %d." % (read_chunk, read_count))
        for itracer in range(read_chunk):
            if verbose:
                print(
                    "Tracer %6d: rho= %g - %g"
                    % (
                        itracer + read_count,
                        data[itracer, :, 3].min(),
                        data[itracer, :, 3].max(),
                    )
                )
            data[itracer, :, :].T.tofile(fout)

        print(
            "Writing done, reading and writing of chunk took %ds."
            % (time.time() - start)
        )

        read_count += read_chunk

        del data
        gc.collect()

    fout.close()

    return


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        help="Name of the model for which to re-write the tracers.",
    )
    parser.add_argument(
        "outfile",
        help="Name of the re-written output file",
    )
    parser.add_argument(
        "-s",
        "--snappath",
        help="Path to the output directory where snapshots are stored",
        default="./",
    )
    parser.add_argument(
        "-c",
        "--chunksize",
        help="Chunksize of tracers to be transposed per loop in GB",
        type=float,
        default=2.5,
    )
    parser.add_argument("--verbose", help="Enables verbose output", action="store_true")

    args = parser.parse_args()

    main(
        args.model,
        args.outfile,
        snappath=args.snappath,
        chunksize=args.chunksize,
        verbose=args.verbose,
    )
    return


if __name__ == "__main__":
    cli()
