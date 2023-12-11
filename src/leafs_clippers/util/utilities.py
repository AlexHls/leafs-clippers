import numpy as np
import pandas as pd


class anyobject:
    def __init__(self):
        return


def dict2obj(dict):
    o = anyobject()
    for key in dict:
        o.__dict__[key] = dict[key]

    return o


def obj2dict(obj):
    d = {}
    for key in obj.__dict__:
        d[key] = obj.__dict__[key]

    return d


def load_species(species_path: str) -> pd.DataFrame:
    """Load species from species file.

    Args:
        species_path (str): species file path

    Returns:
        pd.DataFrame: species dataframe
    """
    species, a, z = np.genfromtxt(
        species_path,
        dtype=str,
        unpack=True,
        skip_header=1,
    )
    a = a.astype(int)
    z = z.astype(int)

    arrays = [a, z]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=["A", "Z"])
    s = pd.DataFrame({"Name": species, "A": a, "Z": z}, index=index)
    return s
