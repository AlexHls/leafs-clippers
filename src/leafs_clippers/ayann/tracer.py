import struct
import numpy as np
import pandas as pd

from leafs_clippers.util import const, utilities


class Tracer:
    def __init__(self, tracer_path: str, species_path: str = "species.txt") -> None:
        self.tracer_path = tracer_path
        self.masses, self.xnuc = self.read_tracer(tracer_path)
        self.species = utilities.load_species(species_path)

    def read_tracer(self, tracer_path: str) -> np.ndarray:
        """Read tracer file and return masses and xnuc.

        Args:
            tracer_path (str): tracer file path

        Returns:
            np.ndarray: masses and xnuc
        """
        with open(tracer_path, "rb") as f:
            ntracer, nspecies = struct.unpack("ii", f.read(8))

            masses = np.fromfile(f, dtype="f8", count=ntracer)
            xnuc = np.fromfile(f, dtype="f8", count=nspecies * ntracer).reshape(
                ntracer, nspecies
            )

        return masses, xnuc

    @property
    def abundances(self) -> pd.DataFrame:
        """Return abundances.

        Returns:
            np.ndarray: abundances
        """
        abundances = self.species.copy()
        x_all = (self.xnuc.T * self.masses).sum(axis=1) / const.M_SOL
        abundances["Xnuc"] = x_all
        return abundances
