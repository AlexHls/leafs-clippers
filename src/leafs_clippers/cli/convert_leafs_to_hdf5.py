import os
import argparse

from leafs_clippers.leafs import leafs as lc


def remove_snap(snapshot, snap, model, directory, no_confirm=False):
    files = [f"{model}o{snap:03d}.{x:03d}" for x in range(snapshot.num_files)]
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
):
    snaplist = lc.get_snaplist(model, snappath=directory, legacy=True)
    for snap in snaplist:
        s = lc.readsnap(
            snap,
            model,
            snappath=directory,
            legacy=True,
            simulation_type=simulation_type,
        )
        outfile = f"{model}o{snap:03d}.hdf5"
        s.convert_to_hdf5(os.path.join(directory, outfile), overwrite=overwrite)
        if replace:
            remove_snap(s, snap, model, directory)

        print(f"Converted snapshot {snap} to {outfile}")


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

    args = parser.parse_args()
    main(
        args.directory,
        args.model,
        overwrite=args.overwrite,
        replace=args.replace,
        no_confirm=args.no_confirm,
        simulation_type=args.simulation_type,
    )


if __name__ == "__main__":
    cli()
