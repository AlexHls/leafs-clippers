import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import FortranFile
from plotly import graph_objects as go

matplotlib.use("Agg")


def sphere(r, c):
    n = 100
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = c[0] + r * np.outer(np.cos(u), np.sin(v))
    y = c[1] + r * np.outer(np.sin(u), np.sin(v))
    z = c[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def plot_bubbles(x, y, z, r_bub, outname, wd_rad=None, plot_type="plotly"):
    if plot_type == "matplotlib":
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
    elif plot_type == "plotly":
        data = []
        for i in range(len(x)):
            xs, ys, zs = sphere(r_bub / 1e5, [x[i] / 1e5, y[i] / 1e5, z[i] / 1e5])
            trace = go.Surface(x=xs, y=ys, z=zs, opacity=0.65)
            trace["surfacecolor"] = np.ones_like(xs)
            data.append(trace)

        if wd_rad is not None:
            xs, ys, zs = sphere(wd_rad / 1e5, [0.0, 0.0, 0.0])
            trace = go.Surface(x=xs, y=ys, z=zs, opacity=0.25, colorscale="Blues")
            trace["surfacecolor"] = np.ones_like(xs)
            data.append(trace)

        # Add scatter for the center of the wd_rad
        trace = go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker=dict(
                size=25,
                color="black",
                symbol="cross",
            ),
        )
        data.append(trace)

        fig = go.Figure(data=data)

        fig.update_layout(
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
            ),
        )

        fig.write_html(outname + ".html")

    return


def create_uniform_bubbles(
    n_bub, r_bub, r_cen, r_sphere, include_center=False, uniform_compresion=0.85
):
    max_coord = (r_sphere - 2 * r_bub) * uniform_compresion / 2
    x_coords = np.linspace(-max_coord, max_coord, n_bub) + r_cen
    y_coords = np.linspace(-max_coord, max_coord, n_bub)
    z_coords = np.linspace(-max_coord, max_coord, n_bub)

    x = []
    y = []
    z = []

    if include_center:
        x.append(r_cen)
        y.append(0.0)
        z.append(0.0)

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
    plot_type="plotly",
    include_center=False,
    uniform_compresion=0.85,
):
    print("Generating ignition bubbles...")
    assert distribution_type in ["uniform", "gaussian"], "Distribution type not valid"
    if distribution_type == "uniform":
        if n_bub == 1:
            x = np.array([r_cen])
            y = np.array([0.0])
            z = np.array([0.0])
        else:
            x, y, z = create_uniform_bubbles(
                n_bub,
                r_bub,
                r_cen,
                r_sphere,
                include_center=include_center,
                uniform_compresion=uniform_compresion,
            )
    elif distribution_type == "gaussian":
        x, y, z = create_gaussian_bubbles(n_bub, r_cen, r_sphere)

    if create_plot:
        print("Plotting bubbles...")
        plot_bubbles(x, y, z, r_bub, outname, wd_rad=wd_rad, plot_type=plot_type)

    print("Writing inipos files...")

    f = FortranFile(outname, "w")

    if distribution_type == "uniform":
        n_bub_out = n_bub**3 + 1 if include_center else n_bub**3
        f.write_record(np.array([n_bub_out], dtype=np.int32))
    elif distribution_type == "gaussian":
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
    parser.add_argument(
        "--plot_type",
        help="Type of plot to create",
        choices=["matplotlib", "plotly"],
        default="plotly",
    )
    parser.add_argument(
        "--include_center",
        help="Include a bubble at the center of the distribution. Only works for uniform distribution",
        action="store_true",
    )
    parser.add_argument(
        "--uniform_compresion",
        help="Compresion factor for the uniform distribution",
        type=float,
        default=0.85,
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
        args.plot_type,
        args.include_center,
    )
    return


if __name__ == "__main__":
    cli()
