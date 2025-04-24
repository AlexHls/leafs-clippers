import os
import argparse
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from parallel_decorators import is_master, vectorize_parallel, mpi_barrier

from leafs_clippers.leafs import leafs as lc


@vectorize_parallel(method="MPI", use_progressbar=True)
def calculate_flamespeeds(
    ind,
    snappath="output",
    model="one_def",
    ignore_cache=False,
    flamespeed_type="schwab",
    normalize="mach",
):
    """
    Calculate the flamespeed for a given snapshot.
    Careful, if this is run in parallel, IO issues may occur.
    """

    s = lc.readsnap(ind, model, snappath=snappath, ignore_cache=ignore_cache)

    schwab_flamespeed = True if flamespeed_type == "schwab" else False

    if normalize == "mach" or normalize == "both":
        s.get_flame_speed_statistic(mach=True, schwab_flamespeed=schwab_flamespeed)
    if normalize == "none" or normalize == "both":
        s.get_flame_speed_statistic(mach=False, schwab_flamespeed=schwab_flamespeed)


def main_flamespeeds(
    path=".",
    model="one_def",
    simulation_type="ONeDef",
    ignore_cache=False,
    flamespeed_type="schwab",
    normalize="mach",
):
    assert simulation_type in [
        "CODef",
        "ONeDef",
        "HeDet",
    ], "Unrecognized simulation type"

    snaps = lc.get_snaplist(model, snappath=path)

    mpi_barrier()

    if is_master():
        print("Calculating flamespeeds...")

    calculate_flamespeeds(
        snaps,
        snappath=path,
        model=model,
        ignore_cache=ignore_cache,
        flamespeed_type=flamespeed_type,
        normalize=normalize,
    )

    mpi_barrier()

    if is_master():
        print("Done!")


def cli():
    parser = argparse.ArgumentParser()

    # This flag selects what to process
    parser.add_argument(
        "mode",
        help="Processing mode. Default: 'flamespeeds'",
        default="flamespeeds",
        choices=["flamespeeds"],
    )

    # General options
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the directory containing the simulation. Default: '.'",
        default=".",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model to use. Default: 'one_def'",
        default="one_def",
    )
    parser.add_argument(
        "-s",
        "--simulation_type",
        help="Simulation type. Default: 'ONeDef'",
        default="ONeDef",
        choices=["ONeDef", "CODef", "HeDet"],
    )
    parser.add_argument(
        "-c",
        "--ignore_cache",
        help="Ignore cache and reprocess all snapshots. Default: False",
        action="store_true",
        default=False,
    )

    # Mode specific arguments
    ## Flamespeeds
    parser.add_argument(
        "--flamespeed_type",
        help="Flamespeed type. Default: 'schwab'",
        default="schwab",
        choices=["schwab", "timmes"],
    )
    parser.add_argument(
        "--normalize",
        help="Normalize flamespeed to e.g. mach number. Default: 'mach'",
        default="mach",
        choices=["mach", "none", "both"],
    )

    args = parser.parse_args()

    if args.mode == "flamespeeds":
        main_flamespeeds(
            path=args.path,
            model=args.model,
            simulation_type=args.simulation_type,
            ignore_cache=args.ignore_cache,
            flamespeed_type=args.flamespeed_type,
            normalize=args.normalize,
        )
    else:
        raise ValueError("Unrecognized mode: %s" % args.mode)


if __name__ == "__main__":
    cli()
