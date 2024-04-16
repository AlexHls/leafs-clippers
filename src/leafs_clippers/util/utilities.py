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


def get_abar_zbar(xnuc, species):
    """
    Calculate abar and zbar from xnuc and species.

    Parameters
    ----------
    xnuc : dict
        dictionary of xnuc
    species : pd.DataFrame
        species dataframe
    """
    abar = 0
    zbar = 0
    xsum = 0
    spec_in_xnuc = list(xnuc.keys())

    for i in range(len(xnuc)):
        name = spec_in_xnuc[i]
        ymass = xnuc[name] / species[species["Name"] == name]["A"].values[0]
        abar += ymass
        zbar += species[species["Name"] == name]["Z"].values[0] * ymass
        xsum += xnuc[name]

    abar = xsum / abar
    zbar = zbar / xsum * abar

    return abar, zbar


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


class LazyDict(dict):
    """
    A dictionary that loads values on demand from a HDF5 file.
    """

    def __init__(self, filename, keys, *args, **kwargs):
        self._loaded_keys = []
        self.filename = filename
        self.set_keys(keys)
        super().__init__(*args, **kwargs)

    def set_keys(self, keys):
        self._keys = keys
        for key in keys:
            self[key] = None

    def __getitem__(self, key):
        if key in self._keys and key not in self._loaded_keys:
            self._load_key(key)
        return super().__getitem__(key)

    def _load_key(self, key):
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for LazyDict")
        with h5py.File(self.filename, "r") as f:
            aux = f.get(key)
            self[key] = np.array(aux[:] if aux is not None else None)
            self._loaded_keys.append(key)
            del aux


def extrapolate_nkk_aux(array):
    new_array = np.zeros(143)
    for i in range(11):
        for j in range(13):
            if j == 12:
                new_array[i * 13 + j] = array[i * 12 + j - 1]
            else:
                new_array[i * 13 + j] = array[i * 12 + j]

    return new_array


def extrapolate_oda_aux(array):
    new_array = np.zeros(143)
    for i in range(11):
        for j in range(13):
            if j == 12:
                new_array[i * 13 + j] = array[i * 12 + j - 1]
            else:
                new_array[i * 13 + j] = array[i * 12 + j]

    return new_array
