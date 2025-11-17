"""
Burn table calibration utilities for LEAFS simulations.

This module provides functions for writing burn table data to FORTRAN binary files.
"""
import numpy as np
from scipy.io import FortranFile


def write_burn_table(filename, rho, abunds):
    """
    Write a burn table to a FORTRAN binary file.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    rho : array_like
        Density array of the material.
    abunds : dict
        Dictionary of mass fractions for each element.
        These will be written in the order provided.
        
    Notes
    -----
    The density and abundance data are sorted in descending order by density
    before writing to the file.
    """
    dtablnr = len(rho)

    # Sort density in descending order and apply same sorting to abundances
    sorted_indices = np.argsort(rho)[::-1]
    rho_bins = rho[sorted_indices]
    
    # Sort abundances using the same indices
    sorted_abunds = {
        elem: np.array(abunds[elem])[sorted_indices] 
        for elem in abunds
    }

    data = {"rho": rho_bins}
    data.update(sorted_abunds)

    with FortranFile(filename, "w") as f:
        f.write_record(np.array([dtablnr]))
        f.write_record(
            np.vstack([data["rho"], *[data[elem] for elem in abunds]]).astype(
                np.float64
            )
        )
