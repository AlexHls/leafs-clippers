import argparse

from leafs_clippers.leafs import leafs_mapping as lm


def main(
    snappath,
    tppnppath,
    one_dim=False,
    model="one_def",
    simulation_type="ONeDef",
    res=100,
    vacuum_threshold=1e-4,
    max_vel=0.0,
    decay_time=0.0,
):
    map = lm.LeafsMapping(
        snappath=snappath,
        tppnppath=tppnppath,
        model=model,
        simulation_type=simulation_type,
    )

    if one_dim:
        map.map1D(
            res=res,
            vacuum_threshold=vacuum_threshold,
            max_vel=max_vel,
            decay_time=decay_time,
        )
    else:
        map.map3D(res=res, vacuum_threshold=vacuum_threshold, max_vel=max_vel)

    return


def cli():
    parser = argparse.ArgumentParser(description="Map leafs simulations to ARTIS")

    parser.add_argument(
        "snappath",
        type=str,
        help="Path to the snapshot directory",
    )
    parser.add_argument(
        "tppnppath",
        type=str,
        help="Path to the TPPNP output directory",
    )
    parser.add_argument(
        "--one_dim",
        action="store_true",
        help="Map to 1D, else 3D",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=100,
        help="Resolution of the mapped output",
    )
    parser.add_argument(
        "--vacuum_threshold",
        type=float,
        default=1e-4,
        help="Threshold for vacuum",
    )
    parser.add_argument(
        "--max_vel",
        type=float,
        default=0.0,
        help="Maximum velocity for the mapping",
    )
    parser.add_argument(
        "--decay_time",
        type=float,
        default=0.0,
        help="Decay time for radioactives. Only used in 1D mapping",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="one_def",
        help="The LEAFS model to use. Default is 'one_def'",
    )
    parser.add_argument(
        "--simulation_type",
        type=str,
        default="ONeDef",
        help="The type of LEAFS simulation. Default is 'ONeDef'",
    )

    args = parser.parse_args()

    main(
        args.snappath,
        args.tppnppath,
        one_dim=args.one_dim,
        res=args.resolution,
        vacuum_threshold=args.vacuum_threshold,
        max_vel=args.max_vel,
        decay_time=args.decay_time,
        model=args.model,
        simulation_type=args.simulation_type,
    )

    return


if __name__ == "__main__":
    cli()
