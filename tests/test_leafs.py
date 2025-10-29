"""Tests for leafs/leafs.py module."""
import pytest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from leafs_clippers.leafs.leafs import (
    get_snaplist,
    readsnap,
    readprotocol,
    readflameprotocol,
)


class TestGetSnaplist:
    """Test get_snaplist function."""

    def test_get_snaplist_hdf5_files(self):
        """Test get_snaplist with HDF5 files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test snapshot files
            model = "test"
            for i in [1, 5, 10, 15]:
                filepath = os.path.join(tmpdir, f"{model}o{i:03d}.hdf5")
                open(filepath, 'a').close()
            
            snaplist = get_snaplist(model, snappath=tmpdir, legacy=False)
            
            assert snaplist == [1, 5, 10, 15]

    def test_get_snaplist_reduced_output(self):
        """Test get_snaplist with reduced output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test snapshot files with 'redo' prefix
            model = "test"
            for i in [2, 4, 6]:
                filepath = os.path.join(tmpdir, f"{model}redo{i:03d}.hdf5")
                open(filepath, 'a').close()
            
            snaplist = get_snaplist(model, snappath=tmpdir, legacy=False, reduced_output=True)
            
            assert snaplist == [2, 4, 6]

    def test_get_snaplist_legacy_parallel(self):
        """Test get_snaplist with legacy parallel files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test snapshot files with .000 extension
            model = "test"
            for i in [1, 3, 7]:
                filepath = os.path.join(tmpdir, f"{model}o{i:03d}.000")
                open(filepath, 'a').close()
            
            snaplist = get_snaplist(model, snappath=tmpdir, legacy=True)
            
            assert snaplist == [1, 3, 7]

    def test_get_snaplist_legacy_serial(self):
        """Test get_snaplist with legacy serial files (fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test snapshot files without .000 extension
            model = "test"
            for i in [2, 8]:
                filepath = os.path.join(tmpdir, f"{model}o{i:03d}")
                open(filepath, 'a').close()
            
            snaplist = get_snaplist(model, snappath=tmpdir, legacy=True)
            
            assert snaplist == [2, 8]

    def test_get_snaplist_empty_directory(self):
        """Test get_snaplist with no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snaplist = get_snaplist("test", snappath=tmpdir, legacy=False)
            
            assert snaplist == []

    def test_get_snaplist_sorted_output(self):
        """Test that get_snaplist returns sorted list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in non-sequential order
            model = "test"
            for i in [10, 1, 5, 3, 7]:
                filepath = os.path.join(tmpdir, f"{model}o{i:03d}.hdf5")
                open(filepath, 'a').close()
            
            snaplist = get_snaplist(model, snappath=tmpdir)
            
            assert snaplist == [1, 3, 5, 7, 10]

    def test_get_snaplist_mixed_files(self):
        """Test get_snaplist ignores non-matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = "test"
            # Create matching files
            for i in [1, 2]:
                filepath = os.path.join(tmpdir, f"{model}o{i:03d}.hdf5")
                open(filepath, 'a').close()
            # Create non-matching files
            open(os.path.join(tmpdir, "other_file.hdf5"), 'a').close()
            open(os.path.join(tmpdir, f"{model}o999.txt"), 'a').close()
            
            snaplist = get_snaplist(model, snappath=tmpdir)
            
            assert snaplist == [1, 2]


class TestReadsnap:
    """Test readsnap function."""

    def test_readsnap_creates_legacy_snapshot(self):
        """Test that readsnap creates LeafsLegacySnapshot for legacy files."""
        with patch('leafs_clippers.leafs.leafs.LeafsLegacySnapshot') as mock_legacy:
            mock_instance = Mock()
            mock_legacy.return_value = mock_instance
            
            result = readsnap(
                1, "test", snappath="/tmp", 
                legacy=True, simulation_type="ONeDef"
            )
            
            mock_legacy.assert_called_once()
            assert result == mock_instance
            # Verify the filename construction
            call_args = mock_legacy.call_args
            assert "testo001" in call_args[0][0]

    def test_readsnap_creates_hdf5_snapshot(self):
        """Test that readsnap creates LeafsSnapshot for HDF5 files."""
        with patch('leafs_clippers.leafs.leafs.LeafsSnapshot') as mock_snap:
            mock_instance = Mock()
            mock_snap.return_value = mock_instance
            
            result = readsnap(
                5, "model", snappath="/tmp",
                legacy=False, quiet=True
            )
            
            mock_snap.assert_called_once()
            assert result == mock_instance
            # Verify the filename construction
            call_args = mock_snap.call_args
            assert "modelo005.hdf5" in call_args[0][0]

    def test_readsnap_reduced_output_filename(self):
        """Test readsnap with reduced_output parameter."""
        with patch('leafs_clippers.leafs.leafs.LeafsSnapshot') as mock_snap:
            mock_instance = Mock()
            mock_snap.return_value = mock_instance
            
            result = readsnap(
                3, "test", snappath="/tmp",
                legacy=False, reduced_output=True
            )
            
            # Verify the filename uses 'redo' instead of 'o'
            call_args = mock_snap.call_args
            assert "testredo003.hdf5" in call_args[0][0]

    def test_readsnap_passes_parameters(self):
        """Test that readsnap passes all parameters correctly."""
        with patch('leafs_clippers.leafs.leafs.LeafsSnapshot') as mock_snap:
            mock_instance = Mock()
            mock_snap.return_value = mock_instance
            
            result = readsnap(
                10, "test", snappath="/data",
                quiet=True, write_derived=False,
                ignore_cache=True, remnant_threshold=5000
            )
            
            call_args = mock_snap.call_args
            assert call_args[1]['quiet'] == True
            assert call_args[1]['write_derived'] == False
            assert call_args[1]['ignore_cache'] == True
            assert call_args[1]['remnant_threshold'] == 5000


class TestReadprotocol:
    """Test readprotocol function."""

    def test_readprotocol_creates_protocol_object(self):
        """Test that readprotocol creates LeafsProtocol."""
        with patch('leafs_clippers.leafs.leafs.LeafsProtocol') as mock_proto:
            mock_instance = Mock()
            mock_proto.return_value = mock_instance
            
            result = readprotocol("test", snappath="/tmp", simulation_type="ONeDef")
            
            mock_proto.assert_called_once_with(
                model="test",
                snappath="/tmp",
                simulation_type="ONeDef",
                quiet=False
            )
            assert result == mock_instance

    def test_readprotocol_default_parameters(self):
        """Test readprotocol with default parameters."""
        with patch('leafs_clippers.leafs.leafs.LeafsProtocol') as mock_proto:
            mock_instance = Mock()
            mock_proto.return_value = mock_instance
            
            result = readprotocol("mymodel")
            
            call_args = mock_proto.call_args
            assert call_args[1]['model'] == "mymodel"
            assert call_args[1]['snappath'] == "./"
            assert call_args[1]['simulation_type'] == "ONeDef"
            assert call_args[1]['quiet'] == False

    def test_readprotocol_quiet_mode(self):
        """Test readprotocol with quiet parameter."""
        with patch('leafs_clippers.leafs.leafs.LeafsProtocol') as mock_proto:
            mock_instance = Mock()
            mock_proto.return_value = mock_instance
            
            result = readprotocol("test", quiet=True)
            
            call_args = mock_proto.call_args
            assert call_args[1]['quiet'] == True


class TestReadflameprotocol:
    """Test readflameprotocol function."""

    def test_readflameprotocol_creates_protocol_object(self):
        """Test that readflameprotocol creates FlameProtocol."""
        with patch('leafs_clippers.leafs.leafs.FlameProtocol') as mock_flame:
            mock_instance = Mock()
            mock_flame.return_value = mock_instance
            
            result = readflameprotocol("test", snappath="/data")
            
            mock_flame.assert_called_once_with(
                model="test",
                snappath="/data"
            )
            assert result == mock_instance

    def test_readflameprotocol_default_snappath(self):
        """Test readflameprotocol with default snappath."""
        with patch('leafs_clippers.leafs.leafs.FlameProtocol') as mock_flame:
            mock_instance = Mock()
            mock_flame.return_value = mock_instance
            
            result = readflameprotocol("mymodel")
            
            call_args = mock_flame.call_args
            assert call_args[1]['model'] == "mymodel"
            assert call_args[1]['snappath'] == "./"
