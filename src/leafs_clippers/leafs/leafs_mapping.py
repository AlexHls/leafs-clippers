import os
import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import least_squares
from scipy.interpolate import NearestNDInterpolator
from tqdm import tqdm

from leafs_clippers.leafs import leafs as lc
from leafs_clippers.leafs import leafs_tracer as lt
from leafs_clippers.util import const as const
from leafs_clippers.util import utilities as util
from leafs_clippers.util import radioactivedecay as lrd


def read_1d_artis_model(root_dir=".", nradioactives=4, max_element=30):
    """
    Read the 1D ARTIS model from the given directory.

    Parameters
    ----------
    root_dir : str, optional
        The root directory. Default: ".".
    nradioactives : int, optional
        The number of radioactive isotopes. Default: 4.
    max_element : int, optional
        The maximum number of elements. Default: 30.

    Returns
    -------
    dict
        The ARTIS model.
    """

    model = {}
    model["data"] = {}
    model["abundances"] = {}
    model["radioactives"] = []

    with open(os.path.join(root_dir, "model_1D.txt"), "r") as f:
        lines = f.readlines()

    model["data"]["res"] = int(lines[0])
    model["data"]["time"] = float(lines[1])

    model["data"]["vel"] = np.zeros(model["data"]["res"])
    model["data"]["rho"] = np.zeros(model["data"]["res"])
    model["data"]["ige"] = np.zeros(model["data"]["res"])
    model["data"]["radioactives"] = np.zeros((model["data"]["res"], nradioactives))
    model["abundances"] = np.zeros((model["data"]["res"], max_element))

    for i in range(model["data"]["res"]):
        line = lines[i + 2].split()
        model["data"]["vel"][i] = float(line[1])
        logrho = float(line[2])
        if logrho != 0.0:
            model["data"]["rho"][i] = 10**logrho
        else:
            model["data"]["rho"][i] = 0.0
        model["data"]["ige"][i] = float(line[3])
        for j in range(nradioactives):
            model["data"]["radioactives"][i, j] = float(line[4 + j])

    with open(os.path.join(root_dir, "abundances_1D.txt"), "r") as f:
        lines = f.readlines()

    for i in range(model["data"]["res"]):
        line = lines[i].split()
        for j in range(max_element):
            model["abundances"][i, j] = float(line[j + 1])

    return model


class MappingTracer:
    def __init__(
        self,
        snapshot,
        trajectory,
        snappath,
        model="one_def",
        remove_bound_core=True,
        remnant_threshold=1e-4,
        ignore_tracers=[],
    ):
        self._model = model
        self._snapshot = snapshot
        self._trajectory = trajectory
        self._snappath = snappath
        self._remove_bound_core = remove_bound_core
        self._remnant_threshold = remnant_threshold
        self._ignore_tracers = ignore_tracers

        # Needed attributes
        self.fpos = self._trajectory.posfin()
        self.frad = np.sqrt(np.sum(self.fpos**2, axis=1))
        self.xiso = self._trajectory.xiso()
        self.isos = [self.convert_iso_string(s) for s in self._trajectory.isos]
        self.a = self._trajectory.a
        self.z = self._trajectory.z
        self.npart = self.fpos.shape[0]

        # Density and mass need to be loaded from the LEAFS snapshot
        tracer = lt.read_tracer(model=model, snappath=snappath)

        self.mass = tracer.tmass
        self.tracer_data = tracer.last().data
        self.rho = self.tracer_data[3, :]

        if self._remove_bound_core:
            self.remove_bound_core()

        # Convert the relevant items to double
        self.fpos = np.array(self.fpos, dtype="float64")
        self.mass = np.array(self.mass, dtype="float64")
        self.rho = np.array(self.rho, dtype="float64")
        self.xiso = np.array(self.xiso, dtype="float64")

    @classmethod
    def convert_iso_string(cls, s):
        """
        Converst the TPPNP iso string to the arepo-like strings
        """
        s = s.decode("utf-8")
        # all lower case
        s = s.lower()
        # remove spaces
        s = s.replace(" ", "")
        if s == "neut":
            s = "n"
        if s == "prot":
            s = "p"
        return s

    def remove_bound_core(self):
        remnant_velocity = self._snapshot._get_remnant_velocity()

        _, unbound = lt.get_bound_unbound_ids(
            model=self._model,
            snappath=self._snappath,
            ignore_cache=True,
            remnant_velocity=remnant_velocity,
            writeout=False,
        )

        # If there are failed tracers, we need to do things tracer by tracer
        # for the tppnp fields
        if len(self._ignore_tracers) > 0:
            pnum = self._trajectory.pnum()
            unbound_pnum, unbound_clean = [], []
            unbound_set, ignore_set = set(unbound), set(self._ignore_tracers)
            for i, idx in enumerate(tqdm(pnum, desc="Removing failed tracers")):
                if idx in unbound_set and idx not in ignore_set:
                    unbound_pnum.append(i)

            self.fpos = self.fpos[unbound_pnum]
            self.frad = self.frad[unbound_pnum]
            self.xiso = self.xiso[unbound_pnum]

            # Convert tppnp ids to array indices
            unbound = np.array(unbound_clean) - 1
        else:
            # Convert tppnp ids to array indices
            unbound = unbound - 1
            self.fpos = self.fpos[unbound]
            self.frad = self.frad[unbound]
            self.xiso = self.xiso[unbound]

        self.mass = self.mass[unbound]
        self.rho = self.rho[unbound]
        self.npart = len(unbound)

        return


class LeafsMapping:
    def __init__(
        self,
        snappath,
        tppnppath,
        model="one_def",
        simulation_type="ONeDef",
        remnant_threshold=1e4,
        remove_bound_core=True,
        quiet=False,
        max_element=30,
    ):
        """
        Mapping of tppnp abundances to a leafs model for e.g. mapping to ARTIS inputs.

        Parameters
        ----------
        snappath : str
            Path to the LEAFS output directory.
        tppnppath : str
            Path to the TPPNP output directory.
        model : str, optional
            The LEAFS model to use. Default: "one_def".
        simulation_type : str, optional
            The type of simulation. Default: "ONeDef".
        remnant_density : float, optional
            The density threshold for the remnant. Default: 1e4.
        remove_bound_core : bool, optional
            Remove the bound core from the LEAFS model. Default
        quiet : bool, optional
            Suppress all output. Default: False.
        max_element : int, optional
            The maximum number of elements to consider. Default: 30.
        """

        # Make sure the necessary modules are available
        try:
            import tppnp as t
        except ImportError:
            raise ImportError("The tppnp module is required.")

        self.snappath = snappath
        self.tppnppath = tppnppath
        self.model = model
        self.remnant_threshold = remnant_threshold
        self.remove_bound_core = remove_bound_core
        self.quiet = quiet
        self.max_element = max_element

        self.traj = t.particle_set()
        self.traj.load_final_abundances(tppnppath, sort=True)

        # Check if failures.txt file is present in tppnp path
        self.failures = False
        ignore_tracers = []
        if os.path.exists(os.path.join(tppnppath, "failures.txt")):
            if not self.quiet:
                print("Warning: failures.txt found in TPPNP path.")
            self.failures = True
            ignore_tracers = np.genfromtxt(
                os.path.join(tppnppath, "failures.txt"), dtype=int
            )

        snaps = lc.get_snaplist(snappath=snappath, model=model)
        self.s = lc.readsnap(
            snaps[-1],
            model=model,
            simulation_type=simulation_type,
            snappath=snappath,
            remnant_threshold=remnant_threshold,
        )

        self.tracer = MappingTracer(
            self.s,
            self.traj,
            snappath=snappath,
            model=model,
            remove_bound_core=remove_bound_core,
            remnant_threshold=remnant_threshold,
            ignore_tracers=ignore_tracers,
        )

        self.rhointp = None

    def _guess_boxsize(self, max_vel=0.0, vacuum_threshold=1e-4):
        if max_vel > 0:
            boxsize = max_vel * self.s.time
        else:
            rad, rho = self.s.get_rad_profile("density")
            boxsize = np.max(rad[rho > vacuum_threshold])

        if not self.quiet:
            print(
                f"Boxsize is {boxsize:e}cm, equivalent to {boxsize / self.s.time / 1e5:.2f} km/s"
            )

        return boxsize

    def _get_rhointp(self, res=200):
        """
        Get the density field for the LEAFS model.
        """

        if not self.quiet:
            print("Calculating density field from last hydro snapshot...")

        x = np.linspace(-self.boxsize, self.boxsize, res)
        y = np.linspace(-self.boxsize, self.boxsize, res)
        z = np.linspace(-self.boxsize, self.boxsize, res)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        XX, YY, ZZ = np.meshgrid(
            self.s.geomx, self.s.geomy, self.s.geomz, indexing="ij"
        )
        select_mask = np.logical_and.reduce(
            (
                XX >= -self.boxsize,
                XX <= self.boxsize,
                YY >= -self.boxsize,
                YY <= self.boxsize,
                ZZ >= -self.boxsize,
                ZZ <= self.boxsize,
            )
        )
        if self.remove_bound_core:
            bm = self.s.get_bound_material()
            select_mask = np.logical_and(select_mask, ~bm)

        input_points = np.array(
            [XX[select_mask].ravel(), YY[select_mask].ravel(), ZZ[select_mask].ravel()]
        ).T
        interp = NearestNDInterpolator(
            input_points,
            self.s.density[select_mask].ravel(),
        )

        return interp(xx, yy, zz)

    def _get_rho_ejecta(self, vacuum_threshold=1e-4, nneighbours=32, res=200):
        try:
            import calcGrid
        except ImportError:
            raise ImportError("The calcGrid module is required.")

        unbound_mask = np.logical_not(
            self.s.get_bound_material(vacuum_threshold=vacuum_threshold)
        )

        rhoa = np.array(self.s.data["density"][unbound_mask], dtype=np.float64)
        vxa = np.array(self.s.data["velx"][unbound_mask], dtype=np.float64)
        vya = np.array(self.s.data["vely"][unbound_mask], dtype=np.float64)
        vza = np.array(self.s.data["velz"][unbound_mask], dtype=np.float64)

        fpos = (
            np.array(
                [self.s.time * vxa, self.s.time * vya, self.s.time * vza],
                dtype=np.float64,
            )
        ).T
        mass = rhoa * self.s.data["vol"][unbound_mask]

        sphgrid_aux = calcGrid.gatherAbundGrid(
            fpos,
            mass,
            rhoa,
            np.zeros([rhoa.size, 1]),
            nneighbours,
            res,
            res,
            res,
            self.boxsize,
            self.boxsize,
            self.boxsize,
            0,
            0,
            0,
            densitycut=vacuum_threshold,
            densityfield=self.rhointp,
            forceneighbourcount=0,
            single_precision=False,
        )

        return sphgrid_aux[:, :, :, 1]

    def _get_expansion_center(self, vacuum_threshold=1e-4):
        if not self.quiet:
            print("Centering explosion...")

        def dist(center, xpos, ypos, zpos, velx, vely, velz, time):
            return np.sqrt(
                (xpos - velx * time - center[0]) ** 2
                + (ypos - vely * time - center[1]) ** 2
                + (zpos - velz * time - center[2]) ** 2
            )

        bm = self.s.get_bound_material(vacuum_threshold=vacuum_threshold)

        tofit = (
            ~bm
            & (self.s.vel_abs > 1e3 * 1e5)
            & (self.s.vel_abs < 2.5e4 * 1e5)
            & (self.s.density > vacuum_threshold)
        )
        XX, YY, ZZ = np.meshgrid(
            self.s.geomx, self.s.geomy, self.s.geomz, indexing="ij"
        )
        xpos, ypos, zpos = XX[tofit], YY[tofit], ZZ[tofit]
        velx, vely, velz = self.s.velx[tofit], self.s.vely[tofit], self.s.velz[tofit]
        bound = self.s.time * 1e4 * 1e5
        opt = least_squares(
            dist,
            [0, 0, 0],
            args=(xpos, ypos, zpos, velx, vely, velz, self.s.time),
            bounds=[np.ones(3) * (-bound), np.ones(3) * bound],
            verbose=True,
            jac="3-point",
        )
        if not self.quiet:
            print("Center of expansion: {0}".format(opt.x))
        return opt.x

    def _write_3D_grid(
        self,
        abundgrid,
        resx,
        resy,
        resz,
        boxx,
        boxy,
        boxz,
        cx,
        cy,
        cz,
        species,
        radioactives,
        vacuum_threshold,
        nneighbours,
        res,
        overwrite,
    ):
        grid = np.zeros((resx, resy, resz, self.max_element))
        gridradioactives = np.zeros((resx, resy, resz, len(radioactives)))
        grid_ige = np.zeros((resx, resy, resz))
        grid_abar = np.zeros((resx, resy, resz))
        grid_stable = np.zeros((resx, resy, resz))

        nspecies = species["count"]
        for i in range(nspecies):
            if species["nz"][i] > 0 and species["nz"][i] <= self.max_element:
                grid[:, :, :, int(species["nz"][i] - 1)] += abundgrid[:, :, :, i]
                # Check for iron group elements
                if species["nz"][i] >= 21:
                    grid_ige += abundgrid[:, :, :, i]
                # Compute aber, invert after loop
                grid_abar += abundgrid[:, :, :, i] / species["na"][i]
            # Check for radioactive isotopes
            for j in range(len(radioactives)):
                if species["names"][i] == radioactives[j]:
                    gridradioactives[:, :, :, j] += abundgrid[:, :, :, i]

        # Compute abar
        ind1 = grid_abar > 0
        grid_abar[ind1] = 1.0 / grid_abar[ind1]
        grid_abar[~ind1] = 0.0

        # Compute stable isotopes
        grid_stable = grid_ige - gridradioactives.sum(axis=3)

        gg = grid.sum(axis=3)
        if not self.quiet:
            print(f"gg.min={gg.min()}, gg.max={gg.max()}")
            print("Writing 3D grid files...")

        if not overwrite:
            if os.path.exists("model.txt") or os.path.exists("abundances.txt"):
                raise ValueError(
                    "Files already exist. Use overwrite=True to overwrite."
                )

        frho = open("model.txt", "w")
        fabund = open("abundances.txt", "w")

        frho.write("%d\n" % (resx * resy * resz))
        frho.write("%g\n" % (self.s.time / (24.0 * 3600)))

        if not self.quiet:
            print(f"Velocity: {boxy / self.s.time / 1e5:.2f} km/s")

        frho.write("%g\n" % (boxy / self.s.time))

        cellcount = 0
        xcellsize = 2 * boxx / resx
        ycellsize = 2 * boxy / resy
        zcellsize = 2 * boxz / resz

        vol = np.zeros((resx, resy, resz))

        rhogrid = self.rhointp

        rhogrid[rhogrid <= vacuum_threshold] = 0.0

        for k in range(resz):
            cellz = (k + 0.5) * zcellsize - boxz + cz
            for j in range(resy):
                celly = (j + 0.5) * ycellsize - boxy + cy
                for i in range(resx):
                    cellx = (i + 0.5) * xcellsize - boxx + cx

                    vol[i, j, k] = xcellsize * ycellsize * zcellsize

                    cellcount += 1

                    frho.write(
                        "%d %g %g %g %g\n"
                        % (cellcount, cellx, celly, cellz, rhogrid[i, j, k])
                    )
                    text = format("%g" % grid_ige[i, j, k])
                    for l in range(len(radioactives)):
                        text += format(" %g" % gridradioactives[i, j, k, l])
                    frho.write(text + "\n")

                    abtext = format("%d" % cellcount)
                    for l in range(self.max_element):
                        abtext += format(" %g" % grid[i, j, k, l])
                    fabund.write(abtext + "\n")

        if not self.quiet:
            print("Total mass: %g" % ((rhogrid * vol).sum() / const.M_SOL))
            print(
                "Total IGE mass: %g" % ((grid_ige * rhogrid * vol).sum() / const.M_SOL)
            )

            for i in range(len(radioactives)):
                print(
                    "Total mass of %s: %g"
                    % (
                        radioactives[i],
                        (gridradioactives[:, :, :, i] * rhogrid * vol).sum()
                        / const.M_SOL,
                    )
                )

        frho.close()
        fabund.close()

        if not self.quiet:
            print("Done writing files.")

        return

    def _write_1D_grid(
        self,
        shell_rho,
        shell_abund,
        shell_iso,
        shell_ige,
        shell_rad,
        shell_vel,
        res,
        boxsize,
        vacuum_threshold,
        radioactives,
        overwrite,
    ):
        if not overwrite:
            if os.path.exists("model_1D.txt") or os.path.exists("abundances_1D.txt"):
                raise ValueError(
                    "Files already exist. Use overwrite=True to overwrite."
                )
        frho = open("model_1D.txt", "w")
        fabund = open("abundances_1D.txt", "w")

        frho.write("%d\n" % res)
        frho.write("%g\n" % (self.s.time / (24.0 * 3600)))

        for i in range(res):
            if shell_rho[i] > 0:
                frho.write("%d %g %g " % (i + 1, shell_vel[i], np.log10(shell_rho[i])))
            else:
                frho.write("%d %g %g " % (i + 1, shell_vel[i], 0.0))
            text = format("%g" % shell_ige[i])
            for j in range(len(radioactives)):
                text += format(" %g" % shell_rad[i, j])
            frho.write(text + "\n")
            abtext = format("%d" % (i + 1))
            for j in range(self.max_element):
                abtext += format(" %g" % shell_abund[i, j + 1])
            fabund.write(abtext + "\n")

        frho.close()
        fabund.close()

        return

    def map1D(
        self,
        res=100,
        vacuum_threshold=1e-4,
        max_vel=0.0,
        radioactives=["ni56", "co56", "fe52", "cr48"],
        decay_time=0.0,
        write_files=True,
        overwrite=False,
    ):
        """
        Map the TPPNP abundances to the LEAFS model in 1D.

        Parameters
        ----------
        res : int, optional
            The number of shells to consider. Default: 100.
        vacuum_threshold : float, optional
            The threshold for vacuum cells. Default: 1e-4.
        max_vel : float, optional
            The maximum velocity to consider. Default: 0.
        radioactives : list, optional
            The list of radioactive isotopes to consider. Default: ["ni56", "co56", "fe52", "cr48"].
        decay_time : float, optional
            If decay_time > 0, all radioactive isotopes will be decayed by decay_time.
            Time has to be in days. Default: 0.0.
        write_files : bool, optional
            Write the ARTIS files. Default: True.
        overwrite : bool, optional
            Overwrite existing files. Default: False.
        """

        self.boxsize = self._guess_boxsize(max_vel=max_vel)

        # Remove the bound core
        bound_mask = self.s.get_bound_material(vacuum_threshold=vacuum_threshold)
        xx, yy, zz = np.meshgrid(self.s.geomx, self.s.geomy, self.s.geomz)
        rad = np.sqrt(xx**2 + yy**2 + zz**2)
        rad_bound = np.max(rad[bound_mask])

        if not self.quiet:
            print("Calculating 1D mapping...")
            print(f"Max radius: {0.5 * self.boxsize} cm.")
            print(f"Min radius: {rad_bound} cm.")
            print(f"Number of shells: {res}.")
            print(f"dr of shells: {0.5 * self.boxsize / res} cm.")

        shell_n, _ = np.histogram(
            self.tracer.frad, bins=res, range=[rad_bound, self.boxsize / 2]
        )

        shell_radius, shell_rho, shell_edges = self.s.get_rad_profile(
            "density",
            res=res,
            min_radius=rad_bound,
            max_radius=0.5 * self.boxsize,
            return_edges=True,
        )
        shell_mass = shell_rho * (
            4.0 / 3.0 * np.pi * (shell_edges[1:] ** 3 - shell_edges[:-1] ** 3)
        )
        shell_iso = np.zeros([res, len(self.tracer.isos)])
        for iso in range(len(self.tracer.isos)):
            shell_iso[:, iso], _, _ = util.binned_statistic_weighted(
                self.tracer.frad,
                self.tracer.xiso[:, iso],
                self.tracer.mass,
                bins=res,
                range=[rad_bound, self.boxsize / 2],
                statistic="sum",
            )

        for i in range(res):
            if shell_rho[i] <= vacuum_threshold:
                shell_iso[i, :] = 0.0

        shell_mass, _, _ = util.binned_statistic_weighted(
            self.tracer.frad,
            self.tracer.mass,
            np.ones(self.tracer.npart),
            bins=res,
            range=[rad_bound, self.boxsize / 2],
            statistic="sum",
        )

        if decay_time > 0:
            print("Doing radioactive decay...")
            rd = lrd.RadioactiveDecay(
                shell_iso,
                shell_mass,
                self.tracer.isos,
                exclude=[x.capitalize() for x in radioactives],
            )
            shell_iso = rd.decay(decay_time)

        if np.any(shell_iso > 1.0):
            raise ValueError("This shouldn't happen")

        zm = int(max(self.tracer.z) + 1)
        shell_abund = np.zeros([res, zm])
        for iz in range(zm):
            if iz > self.max_element - 1:
                # Bin all elements above the maximum element into iron to
                # ensure mass conservation. Implicitly assumes max_element > 26.
                shell_abund[:, 26] += shell_iso[:, self.tracer.z == iz].sum(axis=1)
            else:
                shell_abund[:, iz] = shell_iso[:, self.tracer.z == iz].sum(axis=1)

        shell_ige = shell_iso[:, 21:].sum(axis=1)

        shell_rad = np.zeros([res, len(radioactives)])
        for i in range(len(radioactives)):
            ispecies = self.tracer.isos.index(radioactives[i])
            shell_rad[:, i] = shell_iso[:, ispecies]

        if not self.quiet:
            print(f"Lost mass: {self.tracer.mass.sum() - shell_rho.sum()}")
            print(f"Lost tracers: {self.tracer.npart - shell_n.sum()}")

        for i in range(res):
            if shell_rho[i] <= vacuum_threshold:
                shell_abund[i, :] = 0.0
                shell_iso[i, :] = 0.0
                shell_ige[i] = 0.0
                shell_rad[i, :] = 0.0
                shell_rho[i] = 0.0

        mtot = self.tracer.mass.sum()
        msum = 0.0
        cutoff = res - 1
        for i in range(res):
            msum += shell_rho[i] * (
                4.0 / 3.0 * np.pi * (shell_edges[i + 1] ** 3 - shell_edges[i] ** 3)
            )
            if shell_n[i] == 0 and msum >= 0.95 * mtot:
                cutoff = i
                break

        if not self.quiet:
            print(f"Cutting profiles off at cell {cutoff}.")

        shell_rho[cutoff:] = 0.0
        shell_abund[cutoff:, :] = 0.0
        shell_iso[cutoff:, :] = 0.0
        shell_ige[cutoff:] = 0.0

        shell_vel = shell_radius / self.s.time / 1e5

        self.shell_rho = shell_rho
        self.shell_abund = shell_abund
        self.shell_iso = shell_iso
        self.shell_ige = shell_ige
        self.shell_vel = shell_vel

        # TODO: Implement concistency checks

        if write_files:
            self._write_1D_grid(
                shell_rho,
                shell_abund,
                shell_iso,
                shell_ige,
                shell_rad,
                shell_vel,
                res,
                self.boxsize,
                vacuum_threshold,
                radioactives,
                overwrite,
            )

        return

    def map3D(
        self,
        res=200,
        vacuum_threshold=1e-4,
        max_vel=0.0,
        center_expansion=False,
        radioactives=["ni56", "co56", "fe52", "cr48"],
        nneighbours=32,
        write_files=True,
        overwrite=False,
    ):
        """
        Map the TPPNP abundances to the LEAFS model.

        Parameters
        ----------
        res : int, optional
            The resolution of the mapping. Default: 200.
        vacuum_threshold : float, optional
            The threshold for vacuum cells. Default: 1e-4.
        max_vel : float, optional
            The maximum velocity to consider. Default: 0.
        center_expansion : bool, optional
            Center the box on the expansion center. Default: False.
        radioactives : list, optional
            The list of radioactive isotopes to consider. Default: ["ni56", "co56", "fe52", "cr48"].
        nneighbours : int, optional
            The number of neighbours to consider. Default: 32.
        write_files : bool, optional
            Write the ARTIS files. Default: True.
        overwrite : bool, optional
            Overwrite existing files. Default: False.

        Returns
        -------
        None
        """

        try:
            import calcGrid
        except ImportError:
            raise ImportError("The calcGrid module is required.")

        self.boxsize = self._guess_boxsize(max_vel=max_vel)
        if center_expansion:
            self.center = self._get_expansion_center(vacuum_threshold=vacuum_threshold)
        else:
            self.center = np.array([0.0, 0.0, 0.0])

        # Center data inplace. Overwrites data, probably not ideal,
        # but it is what it is.
        self.tracer.fpos[:, 0] -= self.center[0]
        self.tracer.fpos[:, 1] -= self.center[1]
        self.tracer.fpos[:, 2] -= self.center[2]

        self.s.geomx -= self.center[0]
        self.s.geomy -= self.center[1]
        self.s.geomz -= self.center[2]

        self.rhointp = self._get_rhointp(res=res)

        forceneighbourcount = 0

        if not self.quiet:
            print("Calculating abundance grid...")
            print(f"Resolution: {res}x{res}x{res}.")
            print(f"Box: {self.boxsize}x{self.boxsize}x{self.boxsize}.")
            print(f"Densitycut: {vacuum_threshold}.")
            print(f"NNeighbours: {nneighbours}.")

        (ttt,) = np.where(
            (np.abs(self.tracer.fpos[:, 0]) < self.boxsize)
            & (np.abs(self.tracer.fpos[:, 1]) < self.boxsize)
            & (np.abs(self.tracer.fpos[:, 2]) < self.boxsize)
        )

        if not self.quiet:
            print(f"Using only {len(ttt)} of {self.tracer.npart} tracers.")

        abundgrid = calcGrid.gatherAbundGrid(
            self.tracer.fpos[ttt],
            self.tracer.mass[ttt] * const.M_SOL,
            self.tracer.rho[ttt],
            self.tracer.xiso[ttt],
            nneighbours,
            res,
            res,
            res,
            self.boxsize,
            self.boxsize,
            self.boxsize,
            0,
            0,
            0,
            densitycut=vacuum_threshold,
            densityfield=self.rhointp,
            forceneighbourcount=forceneighbourcount,
            single_precision=False,
        )

        species = {}
        species["na"] = self.tracer.a
        species["nz"] = self.tracer.z
        species["names"] = self.tracer.isos
        species["count"] = len(self.tracer.isos)

        print(abundgrid.shape)
        self.abundgrid = abundgrid

        if write_files:
            self._write_3D_grid(
                abundgrid,
                res,
                res,
                res,
                self.boxsize,
                self.boxsize,
                self.boxsize,
                0,
                0,
                0,
                species,
                radioactives,
                vacuum_threshold,
                nneighbours,
                res,
                overwrite,
            )

        if not self.quiet:
            print("Done.")

        return
