import os
import gc
import time
import struct
import argparse

import numpy as np

from leafs_clippers.util.const import M_SOL
from leafs_clippers import leafs_tracer


def main(model, outfile, snappath="./"):
    fin = leafs_tracer.LeafsTracer(model, snappath=snappath)
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

    fout.write(struct.pack("iiiii", ntracer, 1, ntracer, ntimesteps, 6))
    masses.tofile(fout)
    times.tofile(fout)

    # 1GB Buffer
    read_chunk = 2 * 10**11 // (6 * ntimesteps * 10 * 8)
    print("Reading ", read_chunk, " tracers per chunk.")

    read_count = 0
    while read_count < ntracer:
        read_chunk = min(read_chunk, ntracer - read_count)

        data = np.zeros((read_chunk, ntimesteps, 6), dtype="f4")
        start = time.time()
        print("Reading chunk size %d starting from %d" % (read_chunk, read_count))

        for itstp in range(ntimesteps):
            if itstp % 10000 == 0:
                print("Now reading timestep %d/%d" % (itstp, ntimesteps))

            chunk, _ = fin.get_timestep(
                itstp, read_count=read_count, chunksize=read_chunk, quiet=True
            )
            for ival in range(6):
                data[:, itstp, ival] = chunk[ival, :]

        print("Writing chunk size %d starting from %d." % (read_chunk, read_count))
        for itracer in range(read_chunk):
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

    args = parser.parse_args()

    main(
        args.model,
        args.outfile,
        snappath=args.snappath,
    )
    return


if __name__ == "__main__":
    cli()
