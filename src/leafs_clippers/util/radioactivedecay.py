import numpy as np
from tqdm import tqdm

import radioactivedecay as rd
from joblib import Parallel, delayed


class RadioactiveDecay:
    def __init__(self, xiso, mass, isos, exclude=["Ni56", "Co56", "Fe52", "Cr48"]):
        """
        Initialize the RadioactiveDecay class.

        Parameters:
        ----------
        xiso : numpy.ndarray
            The initial isotopic composition of the particles for.
            Shape needs to be (len(mass), len(isos)).
        mass : numpy.ndarray
            The mass of the particles.
        isos : list
            List with isotope names in xiso.
        exclude : list
            List of isotopes to exclude from the decay calculation.
        """

        self.xiso = xiso
        self.mass = mass
        self.isos = isos
        self.exclude = exclude

    @classmethod
    def convert_radstr_to_iso(cls, s):
        """
        Convert radioactive decay string to iso string
        """
        if len(s) < 2:
            return s
        if "-" not in s:
            return s
        s = s.split("-")

        return f"{s[0].lower()}{s[1]}"

    def _decay_single_particle(self, tp, t_days):
        """
        Perform the radioactive decay for a single particle index tp.
        Returns the resulting isotopic composition for that particle.
        """
        inv = rd.Inventory({}, "g")
        isos_to_decay = []
        if all(self.xiso[tp, :] == 0.0):
            print(f"Empty article {tp}, skipping...")
            return np.zeros(len(self.isos))

        for i in range(len(self.isos)):
            iso_upper = self.isos[i].capitalize()
            try:
                inv.add({iso_upper: self.xiso[tp, i] * self.mass[tp]}, "g")
            except ValueError:
                if self.isos[i] == "ni56":
                    inv.add({"Ni-56": self.xiso[tp, i] * self.mass[tp]}, "g")
                    print(iso_upper)
                # print("Missing: ", iso_upper)
                continue
            isos_to_decay.append(self.isos[i])
        inv.remove(self.exclude)

        inv_dec = inv.decay(t_days, "d")

        dec_masses = inv_dec.masses("g")
        masses_before = self.xiso[tp, :] * self.mass[tp]
        masses_after = masses_before.copy()
        dropped_mass = 0
        for iso in dec_masses:
            iso_conv = self.convert_radstr_to_iso(iso)
            try:
                ind = self.isos.index(iso_conv)
            except ValueError:
                dropped_mass += dec_masses[iso]
                continue
            if iso_conv in [x.lower() for x in self.exclude]:
                masses_after[ind] += dec_masses[iso]
            else:
                masses_after[ind] = dec_masses[iso]

        drop_rel = dropped_mass / masses_before.sum()
        if drop_rel > 1e-3:
            print(f"Particle {tp} dropped {drop_rel:.2%} of its mass")

        # This is likely no longer normalized to 1, but ARTIS should be able to handle this
        return masses_after / self.mass[tp]

    def decay(self, t_days, n_jobs=-1):
        """
        Calculate the decay of isotopes over time.

        Parameters:
        ----------
        t_days : float
            Time in days over which to calculate decay.
        n_jobs : int, optional
            Number of parallel jobs to use. Default is -1 (use all available cores).

        Returns:
        --------
        xiso_after : numpy.ndarray
            The isotopic composition after decay.
        """

        n_particles = len(self.mass)
        print(f"Starting parallel decay for {n_particles} particles...")

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._decay_single_particle)(tp, t_days)
            for tp in tqdm(range(n_particles))
        )

        xiso_after = np.vstack(results)
        return xiso_after
