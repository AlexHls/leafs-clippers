import os
import argparse
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from parallel_decorators import is_master, vectorize_parallel, mpi_barrier

from leafs_clippers.leafs import leafs as lc


@vectorize_parallel(method="MPI", use_progressbar=True)
def make_plot(ind, snappath="output", model="one_def", ye_min=0.25, plotdir="plots"):
    s = lc.readsnap(ind, model, snappath=snappath)

    bs = s.gnx // 2
    res_min, res_max = np.min(np.diff(s.edgex)) / 1e5, np.max(np.diff(s.edgex)) / 1e5

    fig = plt.figure(figsize=(25, 10.5))
    outer_grid = fig.add_gridspec(2, 4)
    axes = []
    caxes = []
    for i in range(2):
        for j in range(4):
            inner_grid = outer_grid[i, j].subgridspec(
                1, 2, width_ratios=[1, 0.05], wspace=0, hspace=0
            )
            ax = fig.add_subplot(inner_grid[0, 0])
            cax = fig.add_subplot(inner_grid[0, 1])
            axes.append(ax)
            caxes.append(cax)

    s.plot_slice(
        "density", ax=axes[0], cax=caxes[0], boxsize=bs, log=True, show_time=False
    )
    s.plot_slice(
        "energy", ax=axes[1], cax=caxes[1], boxsize=bs, log=True, show_time=False
    )
    s.plot_slice(
        "ye",
        ax=axes[2],
        cax=caxes[2],
        boxsize=bs,
        log=False,
        show_time=False,
        Min=ye_min,
        Max=0.5,
    )
    s.plot_slice(
        "Amean",
        ax=axes[3],
        cax=caxes[3],
        boxsize=bs,
        log=False,
        show_time=False,
        Min=1,
        Max=56,
    )

    s.plot_slice("density", ax=axes[4], log=True, show_time=False, cax=caxes[4])
    s.plot_slice("energy", ax=axes[5], log=True, show_time=False, cax=caxes[5])
    s.plot_slice(
        "ye", ax=axes[6], log=False, show_time=False, cax=caxes[6], Min=ye_min, Max=0.5
    )
    s.plot_slice(
        "Amean", ax=axes[7], log=False, show_time=False, cax=caxes[7], Min=1, Max=56
    )

    plt.suptitle(
        r"$\Delta x_{\rm min} = %.1f$ km, $\Delta x_{\rm max} = %.1f$ km, t = %.4f s"
        % (res_min, res_max, s.time)
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(plotdir, "slice_%04d.png" % ind),
        dpi=300,
        # bbox_inches='tight',
    )
    plt.close()


def main(path=".", model="one_def"):
    snappath = os.path.join(path, "output")

    if is_master():
        print("Reading protocol...")
    try:
        lp = lc.LeafsProtocol(model=model, snappath=snappath)
        ye_min = np.min(lp.proto["min_ye"])
    except Exception as e:
        print(e)
        print("Error reading protocol.")
        ye_min = 0.25

    snaps = lc.get_snaplist(model, snappath=snappath)

    plotdir = os.path.join(path, "plots")
    if is_master():
        if not os.path.exists(plotdir):
            print("Creating plots directory...")
            os.makedirs(plotdir)

    mpi_barrier()

    if is_master():
        print("Making plots...")

    make_plot(snaps, snappath=snappath, model=model, ye_min=ye_min, plotdir=plotdir)

    if is_master():
        try:
            ffmpeg_cmd = (
                "ffmpeg -y -r 5 -i %s/slice_%%04d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p %s/slices.mp4"
                % (plotdir, plotdir)
            )
            print("Creating movie...")
            subprocess.run(ffmpeg_cmd, shell=True)
        except Exception as e:
            print(e)
            print("Error creating movie.")

    if is_master():
        print("Done!")


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--path",
        help="Path to the directory containing the simulation. Default:'.'",
        default=".",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model to use. Default:'one_def'",
        default="one_def",
    )

    args = parser.parse_args()
    main(
        path=args.path,
        model=args.model,
    )


if __name__ == "__main__":
    cli()
