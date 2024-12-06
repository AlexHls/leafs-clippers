import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from leafs_clippers.leafs import nuclide_chart as nc


class ParticleXiso:
    def __init__(self, a, z, xiso):
        self.a = a
        self.z = z
        self._xiso = xiso

        assert (
            len(a) == len(z) == len(xiso[0, :])
        ), f"Lengths of a({a.shape}), z({z.shape}), and xiso({xiso.shape}) do not match"

    def xiso(self, pnum=None):
        return self._xiso

    def dm(self, pnum=None):
        return np.array([1.0])

    def pnum(self):
        raise NotImplementedError(
            "This is a single trajectory, it does not have a pnum"
        )


class ParticleTrajectory:
    def __init__(self, file, setup_file):
        self.a, self.z, self.network_isos = ParticleTrajectory.read_networksetup(
            setup_file
        )
        self.data = ParticleTrajectory.read_trajectory_output(file)

        self.a, self.z = self._clean_isotopes()

    def _clean_isotopes(self):
        a_new = []
        z_new = []
        for i in range(len(self.data["iso_names"])):
            if self.data["iso_names"][i] in self.network_isos:
                a_new.append(self.a[self.network_isos.index(self.data["iso_names"][i])])
                z_new.append(self.z[self.network_isos.index(self.data["iso_names"][i])])
        return np.array(a_new), np.array(z_new)

    @classmethod
    def read_networksetup(self, file):
        with open(file, "r") as f:
            lines = f.readlines()

        lines = lines[5:]
        a = []
        z = []
        iso_names = []
        for line in lines:
            if "*****" in line:
                break
            if "OOOOO" in line:
                continue
            a_val, z_val, _ = line[26:].split()
            if int(a_val[:-1]) == 0:
                raise ValueError("A value is 0")
                continue
            a.append(int(a_val[:-1]))
            z.append(int(z_val[:-1]))
            name = line[15:20].rstrip()
            iso_names.append(name.replace(" ", ""))

        return np.array(a), np.array(z), iso_names

    @classmethod
    def read_trajectory_output(self, file):
        iso_names = []
        with open(file, "r") as f:
            # Read only the first line
            line = f.readline()
        line = line[68:]
        # Separate the string into 13 character long substrings and trim whitespace
        for i in range(0, len(line), 13):
            iso_names.append(line[i + 6 : i + 13].strip().replace(" ", ""))

        data = np.genfromtxt(file, skip_header=1)
        cycle = np.array(data[:, 0], dtype=int)
        time = data[:, 1]
        t9 = data[:, 2]
        rho = data[:, 3]
        ye = data[:, 5]
        xnuc = data[:, 6:]

        return {
            "iso_names": iso_names,
            "cycle": cycle,
            "time": time,
            "t9": t9,
            "rho": rho,
            "ye": ye,
            "xnuc": xnuc,
        }

    def get_traj_at_cycle(self, cycle):
        return ParticleXiso(
            self.a, self.z, self.data["xnuc"][self.data["cycle"] == cycle]
        )

    def get_time_at_cycle(self, cycle):
        time = self.data["time"][self.data["cycle"] == cycle]
        return time

    def plot_nuclide_chart(self, cycle, **kwargs):
        traj = self.get_traj_at_cycle(cycle)
        ncp = nc.NuclideChart(traj)
        ax = ncp.plot(
            pnum=kwargs.get("pnum", None),
            ax=kwargs.get("ax", None),
            log=kwargs.get("log", True),
            cutoff=kwargs.get("cutoff", 1e-15),
            max_n=kwargs.get("max_n", 70),
            max_z=kwargs.get("max_z", 70),
            cmap=kwargs.get("cmap", "Spectral_r"),
            plot_network=kwargs.get("plot_network", True),
            network_lw=kwargs.get("network_lw", 0.1),
            network_color=kwargs.get("network_color", "black"),
            network_alpha=kwargs.get("network_alpha", 0.5),
            include_ye=kwargs.get("include_ye", True),
            plot_magic_numbers=kwargs.get("plot_magic_numbers", False),
            plot_stable_isotopes=kwargs.get("plot_stable_isotopes", False),
            vmin=kwargs.get("vmin", None),
            vmax=kwargs.get("vmax", None),
        )
        ax.set_title(f"Cycle {cycle} - Time {self.get_time_at_cycle(cycle)[0]:.2e} s")

    def plot_time_evolution(
        self,
        outdir="plots",
        name_base="time_evolution",
        dpi=300,
        format="png",
        plot_title=True,
        **kwargs,
    ):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for i in tqdm(range(len(self.data["cycle"]))):
            cycle = self.data["cycle"][i]
            traj = self.get_traj_at_cycle(cycle)
            ncp = nc.NuclideChart(traj)
            ax = ncp.plot(
                pnum=kwargs.get("pnum", None),
                ax=kwargs.get("ax", None),
                log=kwargs.get("log", True),
                cutoff=kwargs.get("cutoff", 1e-15),
                max_n=kwargs.get("max_n", 70),
                max_z=kwargs.get("max_z", 70),
                cmap=kwargs.get("cmap", "Spectral_r"),
                plot_network=kwargs.get("plot_network", True),
                network_lw=kwargs.get("network_lw", 0.1),
                network_color=kwargs.get("network_color", "black"),
                network_alpha=kwargs.get("network_alpha", 0.5),
                include_ye=kwargs.get("include_ye", True),
                plot_magic_numbers=kwargs.get("plot_magic_numbers", False),
                plot_stable_isotopes=kwargs.get("plot_stable_isotopes", False),
                vmin=kwargs.get("vmin", None),
                vmax=kwargs.get("vmax", None),
            )
            if plot_title:
                ax.set_title(
                    f"Cycle {cycle} - Time {self.get_time_at_cycle(cycle)[0]:.2e} s"
                )

            fig = ax.get_figure()
            fig.savefig(
                f"{outdir}/{name_base}_{i:04d}.{format}", dpi=dpi, bbox_inches="tight"
            )
            plt.close(fig)
