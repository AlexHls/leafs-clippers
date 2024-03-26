import numpy as np
from scipy.io import FortranFile


def read_inipos(filename):
    data = {}
    f = FortranFile(filename, "r")

    data["n_bub"] = f.read_ints(np.int32)[0]
    data["r_bub"] = f.read_reals()[0]
    data["x"] = f.read_reals()
    data["y"] = f.read_reals()
    data["z"] = f.read_reals()

    f.close()

    return data
