import os
import struct
import argparse

import numpy as np

from leafs_clippers import leafs_tracer


def prepare_yann_tracerinitialcomposition(
    model,
    snappath=".",
    outname="tracerinitialcomposition.dat",
    overwrite=False,
    simulation_type="ONeDef",
):
    assert simulation_type in [
        "CODef",
        "ONeDef",
        "HeDet",
    ], "Unsupported simulation type"
    data = leafs_tracer.LeafsTracer(model, snappath=snappath)
    ini = leafs_tracer.LeafsTracerUtil(data.initial())
    XC = ini.X[0, :]
    XO = ini.X[1, :]
    if len(ini.X) == 6:
        if simulation_type == "ONeDef":
            XHe = np.zeros_like(XC)
            XNe = ini.X[5, :]
        else:
            XHe = ini.X[5, :]
            XNe = np.zeros_like(XC)
    else:
        XHe = np.zeros_like(XC)
        XNe = np.zeros_like(XC)

    ntrace = XC.shape[0]

    if os.path.exists(outname):
        print("Target file '{:s}' already exists".format(outname))
        if not overwrite:
            print("Overwrite flag not set; aborting")
            return False
        else:
            print("Overwrite flag set; overwriting")

    f = open(outname, "wb")
    if simulation_type == "ONeDef":
        f.write(struct.pack("{:d}d".format(ntrace), *XHe.astype(np.float64)))
        f.write(struct.pack("{:d}d".format(ntrace), *XC.astype(np.float64)))
        f.write(
            struct.pack("{:d}d".format(ntrace), *np.zeros_like(XC).astype(np.float64))
        )  # N14
        f.write(struct.pack("{:d}d".format(ntrace), *XO.astype(np.float64)))
        f.write(struct.pack("{:d}d".format(ntrace), *XNe.astype(np.float64)))
    else:
        f.write(struct.pack("{:d}d".format(ntrace), *XHe.astype(np.float64)))
        f.write(struct.pack("{:d}d".format(ntrace), *XC.astype(np.float64)))
        f.write(struct.pack("{:d}d".format(ntrace), *XO.astype(np.float64)))
    f.close()

    return True


def main(args):
    if args.network == "yann":
        prepare_yann_tracerinitialcomposition(
            args.model,
            snappath=args.snappath,
            outname=args.outname,
            overwrite=args.overwrite,
            simulation_type=args.simulation_type,
        )
    else:
        raise ValueError("Network not implemented")

    return None


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        help="Name of the model for which to prepare postprocessing files",
    )
    parser.add_argument(
        "-n",
        "--network",
        help="Specifies the network for which to prepare postprocessing files",
        choices=["yann"],
        default="yann",
    )
    parser.add_argument(
        "-s",
        "--snappath",
        help="Path to the output directory where snapshots are stored",
        default="./",
    )
    parser.add_argument(
        "-o",
        "--outname",
        help="Name of the generated output file",
        default="tracerinitialcomposition.dat",
    )
    parser.add_argument(
        "--overwrite",
        help="If flag is given, any existing output files will be overwritten",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--simulation_type",
        help="Which type of simulation to process",
        choices=["ONeDef", "CODef", "HeDet"],
        default="ONeDef",
    )

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli()
