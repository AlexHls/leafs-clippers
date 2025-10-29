"""Tests for util/utilities.py module."""
import pytest
import numpy as np
import pandas as pd
from leafs_clippers.util.utilities import (
    anyobject,
    dict2obj,
    obj2dict,
    binned_statistic_weighted,
    get_abar_zbar,
    load_species,
    LazyDict,
)


class TestAnyObject:
    """Test anyobject class."""

    def test_anyobject_creation(self):
        """Test creating an anyobject."""
        obj = anyobject()
        assert obj is not None
        assert isinstance(obj, anyobject)

    def test_anyobject_attributes(self):
        """Test setting attributes on anyobject."""
        obj = anyobject()
        obj.x = 10
        obj.y = "test"
        assert obj.x == 10
        assert obj.y == "test"


class TestDict2Obj:
    """Test dict2obj function."""

    def test_dict2obj_simple(self):
        """Test converting simple dict to object."""
        d = {"a": 1, "b": 2, "c": 3}
        obj = dict2obj(d)
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == 3

    def test_dict2obj_mixed_types(self):
        """Test converting dict with mixed types."""
        d = {"int_val": 42, "str_val": "hello", "float_val": 3.14, "list_val": [1, 2, 3]}
        obj = dict2obj(d)
        assert obj.int_val == 42
        assert obj.str_val == "hello"
        assert obj.float_val == 3.14
        assert obj.list_val == [1, 2, 3]

    def test_dict2obj_empty(self):
        """Test converting empty dict."""
        d = {}
        obj = dict2obj(d)
        assert isinstance(obj, anyobject)


class TestObj2Dict:
    """Test obj2dict function."""

    def test_obj2dict_simple(self):
        """Test converting simple object to dict."""
        obj = anyobject()
        obj.a = 1
        obj.b = 2
        obj.c = 3
        d = obj2dict(obj)
        assert d == {"a": 1, "b": 2, "c": 3}

    def test_obj2dict_mixed_types(self):
        """Test converting object with mixed types to dict."""
        obj = anyobject()
        obj.int_val = 42
        obj.str_val = "hello"
        obj.float_val = 3.14
        d = obj2dict(obj)
        assert d["int_val"] == 42
        assert d["str_val"] == "hello"
        assert d["float_val"] == 3.14

    def test_dict2obj_obj2dict_roundtrip(self):
        """Test roundtrip conversion dict -> obj -> dict."""
        original = {"x": 10, "y": 20, "z": 30}
        obj = dict2obj(original)
        result = obj2dict(obj)
        assert result == original


class TestBinnedStatisticWeighted:
    """Test binned_statistic_weighted function."""

    def test_weighted_mean_simple(self):
        """Test weighted mean with simple data."""
        x = np.array([1, 2, 3, 4, 5])
        values = np.array([10, 20, 30, 40, 50])
        weights = np.array([1, 1, 1, 1, 1])
        
        result, bin_edges, binnumber = binned_statistic_weighted(
            x, values, weights, statistic='sum', bins=2, range=(0, 6)
        )
        
        assert len(result) == 2
        assert len(bin_edges) == 3
        assert isinstance(result, np.ndarray)

    def test_weighted_mean_uniform_weights(self):
        """Test with uniform weights."""
        x = np.array([1, 2, 3, 4])
        values = np.array([1, 2, 3, 4])
        weights = np.array([1, 1, 1, 1])
        
        result, _, _ = binned_statistic_weighted(
            x, values, weights, statistic='sum', bins=2, range=(0, 5)
        )
        
        # With uniform weights, weighted mean should equal regular mean
        assert len(result) == 2

    def test_weighted_different_weights(self):
        """Test with different weights."""
        x = np.array([1, 2])
        values = np.array([10, 20])
        weights = np.array([2, 1])
        
        result, _, _ = binned_statistic_weighted(
            x, values, weights, statistic='sum', bins=1, range=(0, 3)
        )
        
        # weighted_values = [20, 20], weights = [2, 1]
        # sum(weighted_values) = 40, sum(weights) = 3
        # result = 40 / 3 ≈ 13.33
        assert len(result) == 1
        assert np.isclose(result[0], 40.0 / 3.0)

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise an error."""
        x = np.array([1, 2, 3])
        values = np.array([10, 20, 30])
        weights = np.array([1, 1])  # Wrong shape
        
        with pytest.raises(AssertionError):
            binned_statistic_weighted(x, values, weights)


class TestGetAbarZbar:
    """Test get_abar_zbar function."""

    def test_get_abar_zbar_single_species(self):
        """Test with a single species."""
        # Create a simple species dataframe
        species_data = {
            'Name': ['h1'],
            'A': [1],
            'Z': [1]
        }
        species = pd.DataFrame(species_data)
        species.index = pd.MultiIndex.from_tuples([(1, 1)], names=['A', 'Z'])
        
        xnuc = {'h1': 1.0}
        
        abar, zbar = get_abar_zbar(xnuc, species)
        
        assert abar == 1.0
        assert zbar == 1.0

    def test_get_abar_zbar_multiple_species(self):
        """Test with multiple species."""
        # Create species dataframe with He4 and C12
        species_data = {
            'Name': ['he4', 'c12'],
            'A': [4, 12],
            'Z': [2, 6]
        }
        species = pd.DataFrame(species_data)
        species.index = pd.MultiIndex.from_tuples([(4, 2), (12, 6)], names=['A', 'Z'])
        
        xnuc = {'he4': 0.5, 'c12': 0.5}
        
        abar, zbar = get_abar_zbar(xnuc, species)
        
        # he4: X/A = 0.5/4 = 0.125
        # c12: X/A = 0.5/12 = 0.04167
        # sum(X/A) = 0.16667
        # abar = sum(X) / sum(X/A) = 1.0 / 0.16667 = 6
        # zbar = (2*0.125 + 6*0.04167) / 0.16667 * 6 = 3
        
        assert np.isclose(abar, 6.0)
        assert np.isclose(zbar, 3.0)


class TestLazyDict:
    """Test LazyDict class."""

    def test_lazydict_initialization(self):
        """Test LazyDict initialization."""
        import tempfile
        import h5py
        
        # Create a temporary HDF5 file with proper data
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Create a proper HDF5 file with test data
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('key1', data=np.array([1, 2, 3]))
            f.create_dataset('key2', data=np.array([4, 5, 6]))
        
        keys = ['key1', 'key2']
        lazy_dict = LazyDict(tmp_path, keys)
        
        assert lazy_dict.filename == tmp_path
        assert 'key1' in lazy_dict
        assert 'key2' in lazy_dict
        
        # Test that keys are initially None (not loaded yet)
        assert lazy_dict._keys == keys
        assert 'key1' not in lazy_dict._loaded_keys
        assert 'key2' not in lazy_dict._loaded_keys
        
        # Access a key to trigger lazy loading
        data1 = lazy_dict['key1']
        assert np.array_equal(data1, np.array([1, 2, 3]))
        assert 'key1' in lazy_dict._loaded_keys
        
        # Clean up
        import os
        os.unlink(tmp_path)

    def test_lazydict_set_keys(self):
        """Test setting keys in LazyDict."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_path = tmp.name
        
        lazy_dict = LazyDict(tmp_path, [])
        lazy_dict.set_keys(['a', 'b', 'c'])
        
        assert 'a' in lazy_dict
        assert 'b' in lazy_dict
        assert 'c' in lazy_dict
        
        # Clean up
        import os
        os.unlink(tmp_path)
