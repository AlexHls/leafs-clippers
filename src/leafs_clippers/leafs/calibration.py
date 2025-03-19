import numpy as np
from scipy.io import FortranFile


def write_burn_table(filename, rho, abunds):
    """
    Write a burn table to a file.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    rho : float
        Density array of the material.
    abunds : float
        Dictionary of mass fractions for each element.
        These will be written in the order provided.
    """

    dtablnr = len(rho)

    # Sort density in descending order
    rho_bins = np.sort(rho)[::-1]
    with FortranFile(filename, "w") as f:
        f.write_record(np.array([dtablnr]))
        f.write_record(rho_bins)
        for elem in abunds:
            # Sort abundances in descending order based on Density
            xfrac = [x for _, x in sorted(zip(rho, abunds[elem]), reverse=True)]
            f.write_record(np.array(xfrac, dtype=np.float64))

    return
