import os
import argparse

from leafs_clippers.leafs import leafs as lc
from leafs_clippers.leafs import utils as lu


def remove_snap(
    snapshot, snap, model, directory, no_confirm=False, reduced_output=False
):
    file_base = "redo" if reduced_output else "o"
    files = [f"{model}{file_base}{snap:03d}.{x:03d}" for x in range(snapshot.num_files)]
    for f in files:
        f = os.path.join(directory, f)
        if os.path.exists(f):
            if no_confirm:
                os.remove(f)
            else:
                confirm = input(f"Delete {f}? [y/N] ")
                if confirm.lower() == "y":
                    os.remove(f)
                    print(f"Deleted {f}")
                else:
                    print(f"Skipping {f}")
        else:
            print(f"File {f} does not exist")


def main(
    directory,
    model,
    overwrite=False,
    replace=False,
    no_confirm=False,
    simulation_type="ONeDef",
    big_endian=False,
    write_xdmf=True,
    reduced_output=False,
    subgrid_size=(64, 64, 64),
):
    file_base = "redo" if reduced_output else "o"
    snaplist = lc.get_snaplist(
        model, snappath=directory, legacy=True, reduced_output=reduced_output
    )
    for snap in snaplist:
        s = lc.readsnap(
            snap,
            model,
            snappath=directory,
            legacy=True,
            simulation_type=simulation_type,
            little_endian=not big_endian,
            reduced_output=reduced_output,
        )
        outfile = f"{model}{file_base}{snap:03d}.hdf5"
        s.convert_to_hdf5(os.path.join(directory, outfile), overwrite=overwrite)
        if replace:
            remove_snap(
                s,
                snap,
                model,
                directory,
                no_confirm=no_confirm,
                reduced_output=reduced_output,
            )

        print(f"Converted snapshot {snap} to {outfile}")

        if write_xdmf:
            # Load new snapshot, writing XDMF files is only supported for HDF5 snapshots
            s = lc.readsnap(
                snap,
                model,
                snappath=directory,
                legacy=False,
                simulation_type=simulation_type,
                little_endian=not big_endian,
                reduced_output=reduced_output,
            )
            writer = lu.LeafsXdmf3Writer(s, subgrid_size=subgrid_size)
            writer.write()
            print(f"Wrote XDMF files for snapshot {snap}")


def cli():
    parser = argparse.ArgumentParser(description="Convert leafs snapshots to HDF5")

    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing leafs snapshots",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Base model name of the snapshots",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 file",
    )
    parser.add_argument(
        "-r",
        "--replace",
        action="store_true",
        help="Replace existing snapshots",
    )
    parser.add_argument(
        "--no_confirm",
        action="store_true",
        help="Do not ask for confirmation before deleting snapshots",
    )
    parser.add_argument(
        "--simulation_type",
        type=str,
        default="ONeDef",
        help="Simulation type",
    )
    parser.add_argument(
        "--big_endian",
        action="store_true",
        help="Big endian",
    )
    parser.add_argument(
        "--no_xdmf",
        action="store_true",
        help="Do not write XDMF files",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Use reduced output files",
    )

    args = parser.parse_args()
    main(
        args.directory,
        args.model,
        overwrite=args.overwrite,
        replace=args.replace,
        no_confirm=args.no_confirm,
        simulation_type=args.simulation_type,
        big_endian=args.big_endian,
        write_xdmf=not args.no_xdmf,
        reduced_output=args.redo,
    )


if __name__ == "__main__":
    cli()
