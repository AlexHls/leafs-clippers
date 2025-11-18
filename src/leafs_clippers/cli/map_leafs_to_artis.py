import argparse
import json
import importlib.resources as r

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
    overwrite=False,
    remove_bound_core=True,
    center_expansion=False,
    replace_bound_region=True,
    map1D=True,
    sph_method="arepo",
    normalize_abundances=True,
    radioactives="default",
):
    with (
        r.files("leafs_clippers.data").joinpath("artis_radio_isos.json").open("r") as f
    ):
        radioactives_list = json.load(f)[radioactives]

    map = lm.LeafsMapping(
        snappath=snappath,
        tppnppath=tppnppath,
        model=model,
        simulation_type=simulation_type,
        remove_bound_core=remove_bound_core,
    )

    if one_dim:
        map.map1D(
            res=res,
            vacuum_threshold=vacuum_threshold,
            max_vel=max_vel,
            decay_time=decay_time,
            overwrite=overwrite,
            center_expansion=center_expansion,
            radioactives=radioactives_list,
        )
    else:
        map.map3D(
            res=res,
            vacuum_threshold=vacuum_threshold,
            max_vel=max_vel,
            center_expansion=center_expansion,
            overwrite=overwrite,
            replace_bound_region=replace_bound_region,
            decay_time=decay_time,
            sph_method=sph_method,
            normalize_abundances=normalize_abundances,
            map1D=map1D,
            radioactives=radioactives_list,
        )

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
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing mapped files",
    )
    parser.add_argument(
        "--no_remove_bound_core",
        action="store_false",
        help="Do not remove the bound core from the mapping",
    )
    parser.add_argument(
        "-c",
        "--center_expansion",
        action="store_true",
        help="Center the box around the center of expansion",
    )
    parser.add_argument(
        "--no_replace_bound_region",
        action="store_false",
        help="Do not replace the bound region with low density material in 3D mapping",
    )
    parser.add_argument(
        "--no_map1D",
        action="store_false",
        help="Do not perform 1D mapping as part of the 3D mapping",
    )
    parser.add_argument(
        "--sph_method",
        type=str,
        default="arepo",
        choices=["arepo", "snsb"],
        help="Which tracer mapping method to use. 'snsb' is not recommended and only for testing.",
    )
    parser.add_argument(
        "--no_normalize_abundances",
        action="store_false",
        help="Do not normalize abundances after mapping in 3D mapping. Only for sph_method 'snsb'.",
    )
    parser.add_argument(
        "--radioactives",
        type=str,
        default="default",
        choices=["default", "tecsne"],
        help="Which set of radioactives to use. Sets are contained in data/artis_radio_isos.json.",
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
        overwrite=args.overwrite,
        remove_bound_core=args.no_remove_bound_core,
        center_expansion=args.center_expansion,
        replace_bound_region=args.no_replace_bound_region,
        sph_method=args.sph_method,
        map1D=args.no_map1D,
        normalize_abundances=args.no_normalize_abundances,
        radioactives=args.radioactives,
    )

    return


if __name__ == "__main__":
    cli()
