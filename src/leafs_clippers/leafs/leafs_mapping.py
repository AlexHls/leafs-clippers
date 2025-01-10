import os
import numpy as np

from leafs_clippers.leafs import leafs as lc
from leafs_clippers.leafs import leafs_tracer as lt
from leafs_clippers.leafs import const as const


class MappingTracer:
    def __init__(
        self,
        snapshot,
        trajectory,
        snappath,
        model="one_def",
        remove_bound_core=True,
        remnant_threshold=1e-4,
    ):
        self._model = model
        self._snapshot = snapshot
        self._trajectory = trajectory
        self._snappath = snappath
        self._remove_bound_core = remove_bound_core
        self._remnant_threshold = remnant_threshold

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

        # Convert tppnp ids to array indices
        unbound = unbound - 1

        self.fpos = self.fpos[unbound]
        self.mass = self.mass[unbound]
        self.rho = self.rho[unbound]
        self.xisos = self.xiso[unbound]
        self.npart = len(unbound)

        return


class LeafsMapping:
    def __init__(
        self,
        snappath,
        tppnppath,
        model="one_def",
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
        self.traj.load_final_abundances(tppnppath)

        snaps = lc.get_snaplist(snappath=snappath, model=model)
        self.s = lc.readsnap(
            snaps[-1],
            model=model,
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
        )

        self.rhointp = None

    def _guess_boxsize(self, max_vel=0):
        if max_vel > 0:
            boxsize = max_vel * self.s.time
        else:
            boxsize = self.s.edgex[-1] - self.s.edgex[0]

        if not self.quiet:
            print(
                f"Boxsize is {boxsize:e}cm, equivalent to {boxsize/self.s.time / 1e5 / 2:.2f} km/s"
            )

        return boxsize

    def _get_rhointp(self, res=200):
        """
        Get the density field for the LEAFS model.
        """

        if not self.quiet:
            print("Calculating density field from last hydro snapshot...")

        dims = (res, res, res)
        ndims = 3
        offset = (0.5 * self.boxsize, 0.5 * self.boxsize, 0.5 * self.boxsize)

        cellsize = self.boxsize / res
        rhointp = np.zeros(dims)

        if not self.quiet:
            print(f"Hydro time: {self.s.time:.2f}s")
            print(
                f"Min, max density: {self.s.density.min():.2e}, {self.s.density.max():.2e}"
            )

        indices = np.zeros((np.max(dims), ndims), dtype=int)
        for dim in range(ndims):
            hygridpos = [self.s.geomx, self.s.geomy, self.s.geomz][dim]
            hygridres = [self.s.gnx, self.s.gny, self.s.gnz][dim]
            hyidx = 0
            idx = 0

            gridpos = (
                np.linspace(0, dims[dim] - 1, dims[dim]) + 0.5
            ) * cellsize - offset[dim]
            if not self.quiet:
                print(
                    f"Dimension {dim}, gridpos: {gridpos.min():.2e}, {gridpos.max():.2e}"
                )

            while hygridpos[0] > gridpos[idx]:
                indices[idx, dim] = 1
                idx += 1

            while idx < dims[dim]:
                while hygridpos[hyidx] < gridpos[idx] and hyidx < hygridres - 1:
                    hyidx += 1

                while idx < dims[dim] and (
                    (hygridpos[hyidx] >= gridpos[idx]) or (hyidx == hygridres - 1)
                ):
                    indices[idx, dim] = hyidx
                    idx += 1

        rho = self.s.data["density"]
        idxleftx = np.minimum(np.maximum(indices[: dims[0], 0] - 1, 0), self.s.gnx - 1)[
            :, None, None
        ] * np.ones(dims, dtype=int)
        idxrightx = np.minimum(np.maximum(indices[: dims[0], 0], 0), self.s.gnx - 1)[
            :, None, None
        ] * np.ones(dims, dtype=int)
        gridposx = (np.linspace(0, dims[0] - 1, dims[0]) + 0.5) * cellsize - offset[0]
        facx = (gridposx[:, None, None] - self.s.geomx[idxleftx]) / (
            self.s.geomx[idxrightx] - self.s.geomx[idxleftx]
        )

        idxlefty = np.minimum(np.maximum(indices[: dims[1], 1] - 1, 0), self.s.gny - 1)[
            None, :, None
        ] * np.ones(dims, dtype=int)
        idxrighty = np.minimum(np.maximum(indices[: dims[1], 1], 0), self.s.gny - 1)[
            None, :, None
        ] * np.ones(dims, dtype=int)
        gridposy = (np.linspace(0, dims[1] - 1, dims[1]) + 0.5) * cellsize - offset[1]
        facy = (gridposy[None, :, None] - self.s.geomy[idxlefty]) / (
            self.s.geomy[idxrighty] - self.s.geomy[idxlefty]
        )

        idxleftz = np.minimum(np.maximum(indices[:, 2] - 1, 0), self.s.gnz - 1)[
            None, None, :
        ] * np.ones(dims, dtype=int)
        idxrightz = np.minimum(np.maximum(indices[:, 2], 0), self.s.gnz - 1)[
            None, None, :
        ] * np.ones(dims, dtype=int)
        gridposz = (np.linspace(0, dims[2] - 1, dims[2]) + 0.5) * cellsize - offset[2]
        facz = (gridposz[None, None, :] - self.s.geomz[idxleftz]) / (
            self.s.geomz[idxrightz] - self.s.geomz[idxleftz]
        )

        y1 = rho[idxleftx, idxlefty, idxleftz] + facx * (
            rho[idxrightx, idxlefty, idxleftz] - rho[idxleftx, idxlefty, idxleftz]
        )
        y2 = rho[idxleftx, idxrighty, idxleftz] + facx * (
            rho[idxrightx, idxrighty, idxleftz] - rho[idxleftx, idxrighty, idxleftz]
        )

        y3 = rho[idxleftx, idxlefty, idxrightz] + facx * (
            rho[idxrightx, idxlefty, idxrightz] - rho[idxleftx, idxlefty, idxrightz]
        )
        y4 = rho[idxleftx, idxrighty, idxrightz] + facx * (
            rho[idxrightx, idxrighty, idxrightz] - rho[idxleftx, idxrighty, idxrightz]
        )

        z1 = y1 + facy * (y2 - y1)
        z2 = y3 + facy * (y4 - y3)

        rhointp = z1 + facz * (z2 - z1)

        if not self.quiet:
            print("Done.")
            print(
                f"Rhointp, min={rhointp.min()}, max={rhointp.max()}, av={rhointp.sum()/res**3}, totmass={rhointp.sum()*self.boxsize**3/res**3/const.M_SOL}"
            )

        return rhointp

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

        frho.write("%g\n" % (resx * resy * resz))
        frho.write("%g\n" % (self.s.time / (24.0 * 3600)))

        if not self.quiet:
            print(f"Velocity: {0.5 * boxy / self.s.time / 1e5:.2f} km/s")

        frho.write("%g\n" % (0.5 * boxy / self.s.time))

        cellcount = 0
        xcellsize = boxx / resx
        ycellsize = boxy / resy
        zcellsize = boxz / resz

        vol = np.zeros((resx, resy, resz))

        if self.remove_bound_core:
            rho_ej = self._get_rho_ejecta(
                vacuum_threshold=vacuum_threshold, nneighbours=nneighbours, res=res
            )
            rhogrid = rho_ej
        else:
            rhogrid = self.rhointp

        rhogrid[rhogrid <= vacuum_threshold] = 0.0

        for k in range(resz):
            cellz = (k + 0.5) * zcellsize - 0.5 * boxz + cz
            for j in range(resy):
                celly = (j + 0.5) * ycellsize - 0.5 * boxy + cy
                for i in range(resx):
                    cellx = (i + 0.5) * xcellsize - 0.5 * boxx + cx

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
        vel,
        nshells,
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

        frho.write("$d\n" % nshells)
        frho.write("%g\n" % (self.s.time / (24.0 * 3600)))

        for i in range(nshells):
            if shell_rho[i] > 0:
                frho.write("%d %g %g " % (i + 1, vel[i], np.log10(shell_rho[i])))
            else:
                frho.write("%d %g %g " % (i + 1, vel[i], 0.0))
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
        nshells=100,
        vacuum_threshold=1e-4,
        max_vel=0,
        radioactives=["ni56", "co56", "fe52", "cr48"],
        write_files=True,
        overwrite=False,
    ):
        """
        Map the TPPNP abundances to the LEAFS model in 1D.

        Parameters
        ----------
        nshells : int, optional
            The number of shells to consider. Default: 100.
        vacuum_threshold : float, optional
            The threshold for vacuum cells. Default: 1e-4.
        max_vel : float, optional
            The maximum velocity to consider. Default: 0.
        radioactives : list, optional
            The list of radioactive isotopes to consider. Default: ["ni56", "co56", "fe52", "cr48"].
        write_files : bool, optional
            Write the ARTIS files. Default: True.
        overwrite : bool, optional
            Overwrite existing files. Default: False.
        """

        self.boxsize = self._guess_boxsize(max_vel=max_vel)
        if not self.quiet:
            print("Calculating 1D mapping...")
            print(f"Max radius: {0.5 * self.boxsize} cm.")
            print(f"Number of shells: {nshells}.")
            print(f"dr of shells: {self.boxsize / nshells} cm.")

        dr = self.boxsize / nshells

        shell_rho, _ = np.histogram(
            self.tracers.frad,
            bins=nshells,
            range=[0, self.boxsize / 2],
            weights=self.tracers.mass,
        )
        shell_rhoav, _ = np.histogram(
            self.tracers.frad,
            bins=nshells,
            range=[0, self.boxsize / 2],
            weights=self.tracers.rho,
        )
        shell_n, _ = np.histogram(
            self.tracers.frad, bins=nshells, range=[0, self.boxsize / 2]
        )

        zm = int(max(self.tracer.z) + 1)
        el = np.zeros([self.tracer.npart, zm])
        shell_abund = np.zeros([nshells, zm])
        for iz in range(zm):
            el[:, iz] = self.tracer.xiso[:, self.tracer.z == iz].sum(axis=1)

        for iz in range(zm):
            shell_abund[:, iz], _ = np.histogram(
                self.tracer.frad,
                bins=nshells,
                range=[0, self.boxsize / 2],
                weights=el[:, iz] * self.tracer.mass,
            )

        shell_iso = np.zeros([nshells, len(self.tracer.isos)])
        for iso in range(len(self.tracer.isos)):
            shell_iso[:, iso], _ = np.histogram(
                self.tracer.frad,
                bins=nshells,
                range=[0, self.boxsize / 2],
                weights=self.tracer.xiso[:, iso] * self.tracer.mass,
            )

        shell_ige, _ = np.histogram(
            self.tracer.frad,
            bins=nshells,
            range=[0, self.boxsize / 2],
            weights=el[:, 21:].sum(axis=1) * self.tracer.mass,
        )

        shell_rad = np.zeros([nshells, len(radioactives)])
        for i in range(len(radioactives)):
            ispecies = self.tracer.isos.index(radioactives[i])
            shell_rad[:, i], _ = np.histogram(
                self.tracer.frad,
                bins=nshells,
                range=[0, self.boxsize / 2],
                weights=self.tracer.xiso[:, ispecies] * self.tracer.mass,
            )

        if not self.quiet:
            print(f"Lost mass: {self.tracer.mass.sum() - shell_rho.sum()}")
            print(f"Lost tracers: {self.tracer.npart - shell_n.sum()}")

        # Normalize
        for i in range(self.nshells1D):
            if shell_rho[i] > vacuum_threshold:
                shell_abund[i, :] /= shell_rho[i]
                shell_rad[i, :] /= shell_rho[i]
                shell_iso[i, :] /= shell_rho[i]
                shell_ige[i] /= shell_rho[i]
                shell_rho[i] *= const.M_SOL / (
                    4.0 / 3.0 * np.pi * dr**3 * ((i + 1) ** 3 - i**3)
                )
            else:
                shell_abund[i, :] = 0.0
                shell_iso[i, :] = 0.0
                shell_ige[i] = 0.0
                shell_rho[i] = 0.0

        # Remove the bound core
        bound_mask = self.s.get_bound_material(vacuum_threshold=vacuum_threshold)
        xx, yy, zz = np.meshgrid(self.s.geomx, self.s.geomy, self.s.geomz)
        rad = np.sqrt(xx**2 + yy**2 + zz**2)
        rad_bound = np.max(rad[bound_mask])

        _, shell_rhohydro = self.s.get_rad_profile(
            "density", res=nshells, min_radius=rad_bound, max_radius=0.5 * self.boxsize
        )
        for i in range(nshells):
            shell_rho[i] = shell_rhohydro[i]
            if shell_rho[i] < vacuum_threshold:
                shell_rho[i] = 0.0
                shell_abund[i, :] = 0.0
                shell_iso[i, :] = 0.0
                shell_ige[i] = 0.0

        # TODO Remove the bound core before this, otherwise this isn't going to work
        mtot = self.tracer.mass.sum()
        msum = 0.0
        cutoff = nshells - 1
        for i in range(nshells):
            msum += shell_rho[i] * (4.0 / 3.0 * np.pi * dr**3 * ((i + 1) ** 3 - i**3))
            if shell_n[i] == 0 and msum >= 0.95 * mtot:
                cutoff = i
                break

        if not self.quiet:
            print(f"Cutting profiles off at cell {cutoff}.")

        shell_rho[cutoff:] = 0.0
        shell_abund[cutoff:, :] = 0.0
        shell_iso[cutoff:, :] = 0.0
        shell_ige[cutoff:] = 0.0

        vel = (np.arange(nshells) + 1) * dr / self.s.time / 1e5

        # TODO: Implement concistency checks

        if write_files:
            self._write_1D_grid(
                shell_rho,
                shell_abund,
                shell_iso,
                shell_ige,
                shell_rad,
                vel,
                nshells,
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
        max_vel=0,
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
        self.rhointp = self._get_rhointp(res=res)

        forceneighbourcount = 0

        if not self.quiet:
            print("Calculating abundance grid...")
            print(f"Resolution: {res}x{res}x{res}.")
            print(f"Box: {self.boxsize}x{self.boxsize}x{self.boxsize}.")
            print(f"Densitycut: {vacuum_threshold}.")
            print(f"NNeighbours: {nneighbours}.")

        (ttt,) = np.where(
            (np.abs(self.tracer.fpos[:, 0]) < 0.5 * self.boxsize)
            & (np.abs(self.tracer.fpos[:, 1]) < 0.5 * self.boxsize)
            & (np.abs(self.tracer.fpos[:, 2]) < 0.5 * self.boxsize)
        )

        if not self.quiet:
            print(f"Using only {len(ttt)} of {self.tracer.npart} tracers.")

        abundgrid = calcGrid.gatherAbundGrid(
            self.tracer.fpos[ttt],
            self.tracer.mass[ttt] * const.M_SOL,
            self.tracer.rho[ttt],
            self.tracer.xisos[ttt],
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
