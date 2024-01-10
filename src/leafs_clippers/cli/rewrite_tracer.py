import os
import gc
import time
import struct
import argparse

import numpy as np

from leafs_clippers.util.const import M_SOL
from leafs_clippers.leafs import leafs_tracer


def main(
    model,
    outfile,
    snappath="./",
    chunksize=2.5,
    verbose=False,
    max_tracers=-1,
    tracer_number=[],
):
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

    if len(tracer_number) > 0:
        fout.write(
            struct.pack(
                "iiiii", len(tracer_number), 1, len(tracer_number), ntimesteps, 6
            )
        )
        masses[tracer_number].tofile(fout)
    if max_tracers > 0:
        write_out_tracers = np.random.choice(ntracer, size=max_tracers, replace=False)
        fout.write(struct.pack("iiiii", max_tracers, 1, max_tracers, ntimesteps, 6))
        masses[write_out_tracers].tofile(fout)
    else:
        fout.write(struct.pack("iiiii", ntracer, 1, ntracer, ntimesteps, 6))
        masses.tofile(fout)
    np.array(times, dtype="f4").tofile(fout)

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
                itstp,
                read_count=read_count,
                usefile=0,
                chunksize=read_chunk,
                quiet=(not verbose),
            )
            for ival in range(6):
                data[:, itstp, ival] = chunk[ival, :]

        print("Writing chunk size %d starting from %d." % (read_chunk, read_count))
        for itracer in range(read_chunk):
            if len(tracer_number) > 0:
                if itracer + read_count not in tracer_number:
                    continue
                else:
                    data[itracer, :, :].T.tofile(fout)
            elif max_tracers > 0 and (itracer + read_count) in write_out_tracers:
                data[itracer, :, :].T.tofile(fout)
            elif max_tracers < 0:
                data[itracer, :, :].T.tofile(fout)
            else:
                continue
            if verbose:
                print(
                    "Tracer %6d: rho= %g - %g"
                    % (
                        itracer + read_count,
                        data[itracer, :, 3].min(),
                        data[itracer, :, 3].max(),
                    )
                )

        print(
            "Writing done, reading and writing of chunk took %ds."
            % (time.time() - start)
        )

        read_count += read_chunk

        del data
        gc.collect()

    fout.close()

    if max_tracers > 0:
        np.savetxt(
            outfile + ".tracers",
            write_out_tracers,
            fmt="%d",
            header="Tracers written out: %d" % max_tracers,
        )

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
    parser.add_argument(
        "-m",
        "--max_tracers",
        help="Maximum number of tracers to be transposed",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--tracer_number",
        help="Specific tracers to be transposed",
        type=int,
        nargs="+",
        default=[],
    )
    parser.add_argument("--verbose", help="Enables verbose output", action="store_true")

    args = parser.parse_args()

    main(
        args.model,
        args.outfile,
        snappath=args.snappath,
        chunksize=args.chunksize,
        verbose=args.verbose,
        max_tracers=args.max_tracers,
        tracer_number=args.tracer_number,
    )
    return


if __name__ == "__main__":
    cli()
