import numpy as np
from tqdm import tqdm

import radioactivedecay as rd
from joblib import Parallel, delayed


def decay_single_particle(
    xiso_row, mass_val, isos, exclude, t_days, particle_index=None
):
    """
    Perform radioactive decay for a single particle.
    Receives the 1-D xiso_row (len(isos)) and scalar mass_val.

    Returns the normalized xiso row after decay (1-D array len(isos)).
    """
    if np.all(xiso_row == 0.0):
        return np.zeros_like(xiso_row)

    inv = rd.Inventory({}, "g")

    for i, iso_name in enumerate(isos):
        iso_upper = iso_name.capitalize()
        mass_to_add = xiso_row[i] * mass_val
        try:
            inv.add({iso_upper: mass_to_add}, "g")
        except ValueError:
            if iso_name.lower() == "ni56":
                inv.add({"Ni-56": mass_to_add}, "g")
            else:
                continue

    inv.remove(exclude)

    inv_dec = inv.decay(t_days, "d")
    dec_masses = inv_dec.masses("g")

    def convert_radstr_to_iso(s):
        """
        Convert radioactive decay string to iso string
        """
        if len(s) < 2:
            return s
        if "-" not in s:
            return s
        s = s.split("-")

        return f"{s[0].lower()}{s[1]}"

    masses_before = xiso_row * mass_val
    masses_after = masses_before.copy()
    dropped_mass = 0.0

    isos_lower = [x.lower() for x in isos]
    exclude_lower = [x.lower() for x in exclude]
    for iso_str, mass_val_after in dec_masses.items():
        iso_conv = convert_radstr_to_iso(iso_str)
        try:
            ind = isos_lower.index(iso_conv)
        except ValueError:
            dropped_mass += mass_val_after
            continue

        if iso_conv in exclude_lower:
            masses_after[ind] += mass_val_after
        else:
            masses_after[ind] = mass_val_after

    total_mass_before = masses_before.sum()
    if total_mass_before == 0:
        return np.zeros_like(xiso_row)

    drop_rel = dropped_mass / masses_before.sum()
    if drop_rel > 1e-3 and particle_index is not None:
        print(f"Particle {particle_index} dropped {drop_rel:.2%} of its mass")

    # This is likely no longer normalized to 1, but ARTIS should be able to handle this
    return masses_after / mass_val


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

    def decay(self, t_days, n_jobs=-1, show_progress=True):
        """
        Calculate the decay of isotopes over time.

        Parameters:
        ----------
        t_days : float
            Time in days over which to calculate decay.
        n_jobs : int, optional
            Number of parallel jobs to use. Default is -1 (use all available cores).
        show_progress : bool, optional
            Whether to show a progress bar. Default is True.

        Returns:
        --------
        xiso_after : numpy.ndarray
            The isotopic composition after decay.
        """

        n_particles = len(self.mass)
        print(f"Starting parallel decay for {n_particles} particles...")
        if n_particles == 0:
            return np.empty((0, len(self.isos)))

        task = [
            (
                self.xiso[tp, :],
                float(self.mass[tp]),
                self.isos,
                self.exclude,
                float(t_days),
                tp,
            )
            for tp in range(n_particles)
        ]

        iterator = tqdm(task, desc="submitting decay tasks") if show_progress else task

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(decay_single_particle)(*task) for task in iterator
        )

        xiso_after = np.vstack(results)
        return xiso_after
