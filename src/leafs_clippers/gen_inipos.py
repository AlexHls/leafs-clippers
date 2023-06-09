import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile


def sphere(r, c):
    n = 100
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = c[0] + r * np.outer(np.cos(u), np.sin(v))
    y = c[1] + r * np.outer(np.sin(u), np.sin(v))
    z = c[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def plot_bubbles(x, y, z, r_bub, outname, wd_rad=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for i in range(len(x)):
        xs, ys, zs = sphere(r_bub / 1e5, [x[i] / 1e5, y[i] / 1e5, z[i] / 1e5])
        if i == 0:
            ax.plot_surface(xs, ys, zs, color="tab:red", alpha=0.75)
        else:
            ax.plot_surface(xs, ys, zs, color="tab:red", alpha=0.75)

    if wd_rad is not None:
        xs, ys, zs = sphere(wd_rad / 1e5, [0.0, 0.0, 0.0])
        ax.plot_surface(xs, ys, zs, color="tab:grey", alpha=0.1)

    # Set an equal aspect ratio
    ax.set_aspect("equal")
    # Set labels
    ax.set(
        xlabel="X (km)",
        ylabel="Y (km)",
        zlabel="Z (km)",
    )

    plt.savefig(
        outname + ".png",
        # bbox_inches="tight",
        dpi=300,
    )
    return


def create_uniform_bubbles(n_bub, r_cen, r_sphere):
    max_coord = r_sphere / np.sqrt(3)
    x_coords = np.linspace(-max_coord, max_coord, n_bub) + r_cen
    y_coords = np.linspace(-max_coord, max_coord, n_bub)
    z_coords = np.linspace(-max_coord, max_coord, n_bub)

    x = []
    y = []
    z = []

    for i in x_coords:
        for j in y_coords:
            for k in z_coords:
                x.append(i)
                y.append(j)
                z.append(k)

    return np.array(x), np.array(y), np.array(z)


def create_gaussian_bubbles(n_bub, r_cen, r_sphere):
    rng = np.random.default_rng()
    radius = rng.normal(0, r_sphere, size=n_bub)
    theta = rng.uniform(low=0, high=np.pi, size=n_bub)
    phi = rng.uniform(low=0, high=2 * np.pi, size=n_bub)

    # Take the absolute of the radius
    radius = np.abs(radius)

    x = radius * np.sin(theta) * np.cos(phi) + r_cen
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return x, y, z


def main(
    outname,
    n_bub=5,
    r_bub=5e6,
    r_cen=0.0,
    r_sphere=5e7,
    distribution_type="uniform",
    create_plot=False,
    wd_rad=None,
):
    print("Generating ignition bubbles...")
    if distribution_type == "uniform":
        x, y, z = create_uniform_bubbles(n_bub, r_cen, r_sphere)
    elif distribution_type == "gaussian":
        x, y, z = create_gaussian_bubbles(n_bub, r_cen, r_sphere)
    else:
        raise ValueError("Distribution type not implemented")

    if create_plot:
        print("Plotting bubbles...")
        plot_bubbles(x, y, z, r_bub, outname, wd_rad=wd_rad)

    print("Writing inipos files...")

    f = FortranFile(outname, "w")

    f.write_record(np.array([n_bub], dtype=np.int32))
    f.write_record(np.array([r_bub]))
    f.write_record(x.T)
    f.write_record(y.T)
    f.write_record(z.T)

    f.close()

    print("All done!")

    return


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "outname", help="Name of the output file where the data is written to"
    )
    parser.add_argument(
        "--n_bub",
        "-n",
        help="Number of bubbles to generate",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--r_bub",
        "-r",
        help="Size of each individual bubble",
        type=float,
        default=5e6,
    )
    parser.add_argument(
        "--r_cen",
        "-c",
        help="Center of the bubble distribution",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--r_sphere",
        "-s",
        help="Size of the bubble distribution sphere",
        type=float,
        default=5e7,
    )
    parser.add_argument(
        "--distribution_type",
        help="Type of bubble distribution",
        choices=["uniform", "gaussian"],
        default="uniform",
    )
    parser.add_argument(
        "--create_plot",
        help="Creates a plot visualizing the bubbles",
        action="store_true",
    )
    parser.add_argument(
        "--wd_rad",
        help="WD radius. Is used to overplot WD in bubble plot",
        type=float,
    )

    args = parser.parse_args()

    main(
        args.outname,
        args.n_bub,
        args.r_bub,
        args.r_cen,
        args.r_sphere,
        args.distribution_type,
        args.create_plot,
        args.wd_rad,
    )
    return


if __name__ == "__main__":
    cli()
