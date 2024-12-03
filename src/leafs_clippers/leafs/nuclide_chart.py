import numpy as np
import matplotlib.pyplot as plt

# fmt: off
STABLE_Z = np.array([ 1,  1,  2,  2,  3,  3,  4,  5,  5,  6,  6,  7,  7,  8,  8,  8,  9,
       10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15, 16, 16, 16, 16, 17,
       17, 18, 18, 18, 19, 19, 20, 20, 20, 20, 20, 21, 22, 22, 22, 22, 22,
       23, 24, 24, 24, 25, 26, 26, 26, 26, 27, 28, 28, 28, 28, 28, 29, 29,
       30, 30, 30, 30, 31, 31, 32, 32, 32, 32, 32, 33, 34, 34, 34, 34, 34,
       35, 35, 36, 36, 36, 36, 36, 36, 37, 38, 38, 38, 38, 39, 40, 40, 40,
       40, 41, 42, 42, 42, 42, 42, 42, 44, 44, 44, 44, 44, 44, 44, 45, 46,
       46, 46, 46, 46, 46, 47, 47, 48, 48, 48, 48, 48, 48, 49, 50, 50, 50,
       50, 50, 50, 50, 50, 50, 50, 51, 51, 52, 52, 52, 52, 52, 53, 54, 54,
       54, 54, 54, 54, 55, 56, 56, 56, 56, 56, 56, 57, 58, 58, 59, 60, 60,
       60, 60, 60, 62, 62, 62, 62, 62, 63, 64, 64, 64, 64, 64, 64, 65, 66,
       66, 66, 66, 66, 66, 66, 67, 68, 68, 68, 68, 68, 68, 69, 70, 70, 70,
       70, 70, 70, 70, 71, 72, 72, 72, 72, 72, 73, 74, 74, 74, 75, 76, 76,
       76, 76, 76, 77, 77, 78, 78, 78, 78, 78, 79, 80, 80, 80, 80, 80, 80,
       80, 81, 81, 82, 82, 82])

STABLE_N = np.array([  0,   1,   1,   2,   3,   4,   5,   5,   6,   6,   7,   7,   8,
         8,   9,  10,  10,  10,  11,  12,  12,  12,  13,  14,  14,  14,
        15,  16,  16,  16,  17,  18,  20,  18,  20,  18,  20,  22,  20,
        22,  20,  22,  23,  24,  26,  24,  24,  25,  26,  27,  28,  28,
        28,  29,  30,  30,  28,  30,  31,  32,  32,  30,  32,  33,  34,
        36,  34,  36,  34,  36,  37,  38,  38,  40,  38,  40,  41,  42,
        44,  42,  40,  42,  43,  44,  46,  44,  46,  42,  44,  46,  47,
        48,  50,  48,  46,  48,  49,  50,  50,  50,  51,  52,  54,  52,
        50,  52,  53,  54,  55,  56,  52,  54,  55,  56,  57,  58,  60,
        58,  56,  58,  59,  60,  62,  64,  60,  62,  58,  60,  62,  63,
        64,  66,  64,  62,  64,  65,  66,  67,  68,  69,  70,  72,  74,
        70,  72,  68,  70,  72,  73,  74,  74,  72,  74,  75,  76,  77,
        78,  78,  74,  78,  79,  80,  81,  82,  82,  78,  82,  82,  82,
        83,  85,  86,  88,  82,  87,  88,  90,  92,  90,  90,  91,  92,
        93,  94,  96,  94,  90,  92,  94,  95,  96,  97,  98,  98,  94,
        96,  98,  99, 100, 102, 100,  98, 100, 101, 102, 103, 104, 106,
       104, 104, 105, 106, 107, 108, 108, 108, 110, 112, 110, 111, 112,
       113, 114, 116, 114, 116, 114, 116, 117, 118, 120, 118, 116, 118,
       119, 120, 121, 122, 124, 122, 124, 124, 125, 126])
# fmt: on


class NuclideChart:
    def __init__(self, p):
        self.p = p
        self.n = p.a - p.z
        self.z = p.z
        self.a = p.a

        self.n_unique = sorted(list(set(self.n)))
        self.z_unique = sorted(list(set(self.z)))

        self.zuq = np.unique(self.z)
        self.nuq = np.unique(self.n)

        self.n_grid = np.zeros(len(self.n_unique) + 1)
        self.z_grid = np.zeros(len(self.z_unique) + 1)
        for i in range(len(self.n_unique)):
            self.n_grid[i] = self.n_unique[i] - 0.5
        self.n_grid[-1] = self.n_unique[-1] + 0.5
        for i in range(len(self.z_unique)):
            self.z_grid[i] = self.z_unique[i] - 0.5
        self.z_grid[-1] = self.z_unique[-1] + 0.5

    def get_xiso(self, pnum=None, pnum_mask=None):
        if pnum_mask is not None:
            mask = np.isin(pnum_mask, self.p.pnum())
            pnum = pnum_mask[mask]
            xiso = np.array([self.p.xiso(p=i) for i in pnum])
            dm = np.array([self.p.dm(p=i) for i in pnum])
            sum = (xiso.T * dm).sum(axis=1)
            self.xiso = sum / dm.sum()
        elif pnum is not None:
            self.xiso = self.p.xiso(p=pnum)
        else:
            sum = (self.p.xiso().T * self.p.dm()).sum(axis=1)
            self.xiso = sum / self.p.dm().sum()

        return self.xiso

    def map_xiso_to_grid(self, xiso, cutoff=1e-15, log=True):
        """
        Map the xiso to a grid of n and z values
        """

        # Create the grid
        grid = np.zeros((len(self.n_unique), len(self.z_unique)))

        # Fill the grid
        for i in range(len(xiso)):
            n_idx = self.n_unique.index(self.n[i])
            z_idx = self.z_unique.index(self.z[i])
            grid[n_idx, z_idx] = xiso[i]

        # Apply threshold
        if cutoff is not None:
            grid_ma = np.ma.masked_where(grid < cutoff, grid)
            if log:
                grid_ma = np.log10(grid_ma)
            return grid_ma
        if log:
            grid = np.log10(grid)

        return grid

    def draw_network(self, ax, lw=0.1, color="black", alpha=0.5):
        for zz in self.zuq:
            ns = np.sort(self.n[np.where(self.z == zz)])
            seqs = np.split(ns, np.where(np.diff(ns) != 1)[0] + 1)
            for seq in seqs:
                y = zz
                x1 = np.min(seq)
                x2 = np.max(seq)
                ax.plot(
                    [x1 - 0.5, x2 + 0.5],
                    [y + 0.5, y + 0.5],
                    color=color,
                    lw=lw,
                    alpha=alpha,
                )
                ax.plot(
                    [x1 - 0.5, x2 + 0.5],
                    [y - 0.5, y - 0.5],
                    color=color,
                    lw=lw,
                    alpha=alpha,
                )

        # now the vertical lines
        for nn in self.nuq:
            zs = np.sort(self.z[np.where(self.n == nn)])
            seqs = np.split(zs, np.where(np.diff(zs) != 1)[0] + 1)
            for seq in seqs:
                x = nn
                y1 = np.min(seq)
                y2 = np.max(seq)
                ax.plot(
                    [x + 0.5, x + 0.5],
                    [y1 - 0.5, y2 + 0.5],
                    color=color,
                    lw=lw,
                    alpha=alpha,
                )
                ax.plot(
                    [x - 0.5, x - 0.5],
                    [y1 - 0.5, y2 + 0.5],
                    color=color,
                    lw=lw,
                    alpha=alpha,
                )

        return ax

    def plot_stable_isotopes(self, ax, network_lw=0.1, network_color="black"):
        for i in range(len(STABLE_Z)):
            z = STABLE_Z[i]
            n = STABLE_N[i]
            ax.pcolormesh(
                np.array([n - 0.5, n + 0.5]),
                np.array([z - 0.5, z + 0.5]),
                np.array([[0]]),
                color="black",
                alpha=1.0,
                lw=network_lw,
                edgecolor=network_color,
                facecolor="none",
            )

        return ax

    def get_ye(self):
        """
        Get the electron fraction
        """
        abar = 0
        zbar = 0
        xsum = 0
        for i in range(len(self.xiso)):
            ymass = self.xiso[i] / self.a[i]
            abar += ymass
            zbar += self.z[i] * ymass
            xsum += self.xiso[i]

        abar = xsum / abar
        zbar = zbar / xsum * abar

        return zbar / abar

    def plot_magic_numbers(self, ax, network_lw=0.1, network_color="black"):
        magic = [8, 20, 28, 50, 82, 126]
        for m in magic:
            ax.axvline(m - 0.5, ls=":", color=network_color, lw=network_lw)
            ax.axvline(m + 0.5, ls=":", color=network_color, lw=network_lw)
            ax.axhline(m - 0.5, ls=":", color=network_color, lw=network_lw)
            ax.axhline(m + 0.5, ls=":", color=network_color, lw=network_lw)

        return ax

    def plot(
        self,
        pnum=None,
        ax=None,
        log=True,
        cutoff=1e-15,
        max_n=70,
        max_z=70,
        cmap="viridis",
        plot_network=True,
        network_lw=0.1,
        network_color="black",
        network_alpha=0.5,
        include_ye=True,
        pnum_mask=None,
        plot_magic_numbers=False,
        plot_stable_isotopes=False,
    ):
        """
        Plot the network nuclide chart
        """

        self.get_xiso(pnum=pnum, pnum_mask=pnum_mask)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        grid = self.map_xiso_to_grid(self.xiso, cutoff=cutoff, log=log)

        pcm = ax.pcolormesh(self.n_grid, self.z_grid, grid.T, cmap=cmap)

        if plot_network:
            self.draw_network(
                ax, lw=network_lw, color=network_color, alpha=network_alpha
            )

        if plot_magic_numbers:
            self.plot_magic_numbers(ax, network_lw * 2, network_color)

        if plot_stable_isotopes:
            self.plot_stable_isotopes(ax, network_lw * 2, network_color)

        ax.set_xlim(0, max_n)
        ax.set_ylim(0, max_z)

        # Add colorbar
        cbar = fig.colorbar(pcm, ax=ax)
        if log:
            cbar.set_label(r"$\log_{10}(X)$")
        else:
            cbar.set_label(r"$X$")

        ax.set_xlabel("Number of neutrons")
        ax.set_ylabel("Number of protons")

        if include_ye:
            ye = self.get_ye()
            ax.text(0.1, 0.9, f"Y$_e$ = {ye:.4f}", transform=ax.transAxes)

        return ax
