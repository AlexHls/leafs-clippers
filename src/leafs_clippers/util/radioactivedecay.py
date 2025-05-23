import numpy as np
from tqdm import tqdm


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

        try:
            import radioactivedecay as rd
        except ImportError:
            raise ImportError(
                "Please install the radioactivedecay package to use this class."
            )
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

    def decay(self, t_days):
        """
        Calculate the decay of isotopes over time.

        Parameters:
        ----------
        t_days : float
            Time in days over which to calculate decay.

        Returns:
        --------
        xiso_after : numpy.ndarray
            The isotopic composition after decay.
        """

        xiso_after = np.zeros((len(self.mass), len(self.isos)))

        inv = rd.Inventory({}, "g")
        isos_to_decay = []
        for tp in tqdm(range(len(self.mass))):
            for i in range(len(self.isos)):
                iso_upper = self.isos[i].upper()
                try:
                    inv.add({iso_upper: self.xiso[tp, i] * self.mass[tp]}, "g")
                except ValueError:
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
            xiso_after[tp, :] = masses_after / self.mass[tp]

        return xiso_after
