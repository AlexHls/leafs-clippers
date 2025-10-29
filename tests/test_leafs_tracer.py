"""Tests for leafs/leafs_tracer.py module."""
import pytest
import os
import tempfile
import struct
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open
from leafs_clippers.leafs.leafs_tracer import (
    read_tracer,
    get_bound_unbound_ids,
    LeafsTracer,
)


class TestReadTracer:
    """Test read_tracer function."""

    def test_read_tracer_creates_leafs_tracer(self):
        """Test that read_tracer creates LeafsTracer object."""
        with patch('leafs_clippers.leafs.leafs_tracer.LeafsTracer') as mock_tracer:
            mock_instance = Mock()
            mock_tracer.return_value = mock_instance
            
            result = read_tracer("test", snappath="/tmp", npart=100)
            
            mock_tracer.assert_called_once_with("test", "/tmp", 100, "", True)
            assert result == mock_instance

    def test_read_tracer_with_file(self):
        """Test read_tracer with file parameter."""
        with patch('leafs_clippers.leafs.leafs_tracer.LeafsTracer') as mock_tracer:
            mock_instance = Mock()
            mock_tracer.return_value = mock_instance
            
            result = read_tracer("test", file="tracer.dat", vartracer=False)
            
            call_args = mock_tracer.call_args
            assert call_args[0][3] == "tracer.dat"
            assert call_args[0][4] == False

    def test_read_tracer_default_parameters(self):
        """Test read_tracer with default parameters."""
        with patch('leafs_clippers.leafs.leafs_tracer.LeafsTracer') as mock_tracer:
            mock_instance = Mock()
            mock_tracer.return_value = mock_instance
            
            result = read_tracer("mymodel")
            
            call_args = mock_tracer.call_args
            assert call_args[0][0] == "mymodel"
            assert call_args[0][1] == "./"
            assert call_args[0][2] == 0
            assert call_args[0][3] == ""
            assert call_args[0][4] == True


class TestGetBoundUnboundIds:
    """Test get_bound_unbound_ids function."""

    def test_get_bound_unbound_from_cache(self):
        """Test loading bound/unbound particles from cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache files
            bound_ids = np.array([1, 2, 3, 4, 5])
            unbound_ids = np.array([6, 7, 8, 9, 10])
            np.savetxt(os.path.join(tmpdir, "bound.txt"), bound_ids, fmt="%d")
            np.savetxt(os.path.join(tmpdir, "unbound.txt"), unbound_ids, fmt="%d")
            
            bound, unbound = get_bound_unbound_ids(
                model="test",
                snappath=tmpdir,
                ignore_cache=False
            )
            
            assert np.array_equal(bound, bound_ids)
            assert np.array_equal(unbound, unbound_ids)

    def test_get_bound_unbound_ignore_cache(self):
        """Test calculating bound/unbound particles when ignoring cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache files that should be ignored
            np.savetxt(os.path.join(tmpdir, "bound.txt"), [1, 2], fmt="%d")
            np.savetxt(os.path.join(tmpdir, "unbound.txt"), [3, 4], fmt="%d")
            
            # Mock the tracer reading
            with patch('leafs_clippers.leafs.leafs_tracer.read_tracer') as mock_read:
                mock_tracer = Mock()
                mock_last = Mock()
                
                # Create mock data: 3 particles
                # offset = 1 + 6 + qn(6) + 2*nlset(1) = 15
                # Need at least 19 rows (15 + 4 for velocities)
                mock_data = np.zeros((20, 3))
                mock_data[5, :] = [-1, 2, -3]  # eint
                mock_data[15, :] = [-2, -1, -4]  # egrav (off=15)
                mock_data[16, :] = [0, 0, 1]  # vx (off+1)
                mock_data[17, :] = [0, 0, 0]  # vy (off+2)
                mock_data[18, :] = [0, 0, 0]  # vz (off+3)
                
                mock_last.data = mock_data
                mock_tracer.last.return_value = mock_last
                mock_tracer.npart = 3
                mock_read.return_value = mock_tracer
                
                bound, unbound = get_bound_unbound_ids(
                    model="test",
                    snappath=tmpdir,
                    ignore_cache=True,
                    remnant_velocity=[0, 0, 0],
                    writeout=False
                )
                
                # Particle 1: eint=(-1), egrav=(-2), ekin=0, etot=-3 < 0 -> bound
                # Particle 2: eint=(2), egrav=(-1), ekin=0, etot=1 >= 0 -> unbound  
                # Particle 3: eint=(-3), egrav=(-4), ekin=0.5, etot=-6.5 < 0 -> bound
                assert 1 in bound or 3 in bound
                assert 2 in unbound

    def test_get_bound_unbound_writeout(self):
        """Test that bound/unbound files are written when writeout=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('leafs_clippers.leafs.leafs_tracer.read_tracer') as mock_read:
                mock_tracer = Mock()
                mock_last = Mock()
                
                # Simple data: 2 particles, offset = 15
                mock_data = np.zeros((20, 2))
                mock_data[5, :] = [-1, 2]  # eint
                mock_data[15, :] = [-2, -1]  # egrav (off=15)
                mock_data[16, :] = [0, 0]  # vx
                mock_data[17, :] = [0, 0]  # vy
                mock_data[18, :] = [0, 0]  # vz
                
                mock_last.data = mock_data
                mock_tracer.last.return_value = mock_last
                mock_tracer.npart = 2
                mock_read.return_value = mock_tracer
                
                bound, unbound = get_bound_unbound_ids(
                    model="test",
                    snappath=tmpdir,
                    ignore_cache=True,
                    writeout=True
                )
                
                # Check that files were created
                assert os.path.exists(os.path.join(tmpdir, "bound.txt"))
                assert os.path.exists(os.path.join(tmpdir, "unbound.txt"))


class TestLeafsTracer:
    """Test LeafsTracer class."""

    def create_test_tracer_file(self, filepath, npart=10, has_mass_header=True, vartracer=False):
        """Helper to create a test tracer file."""
        with open(filepath, 'wb') as f:
            # Write header
            if has_mass_header:
                f.write(struct.pack('<i', 12))  # header length
                f.write(struct.pack('<i', npart))
                f.write(struct.pack('<d', 1.0))  # mass
            else:
                f.write(struct.pack('<i', 4))  # header length
                f.write(struct.pack('<i', npart))
            
            f.write(struct.pack('<i', 0))  # end of header
            
            # Write time record
            f.write(struct.pack('<i', 8))
            f.write(struct.pack('<d', 0.0))  # time
            f.write(struct.pack('<i', 8))

    def test_leafs_tracer_initialization_with_file(self):
        """Test LeafsTracer initialization with a specific file."""
        with tempfile.NamedTemporaryFile(suffix='.trace', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            self.create_test_tracer_file(filepath, npart=5)
            
            tracer = LeafsTracer("test", snappath="/tmp", file=filepath)
            
            assert tracer.name == "test"
            assert tracer.snappath == "/tmp"
            assert tracer.npart == 5
            assert tracer.nfiles == 1
            assert len(tracer.files) == 1
            assert len(tracer.starttimes) == 1
        finally:
            os.unlink(filepath)

    def test_leafs_tracer_initialization_default(self):
        """Test LeafsTracer initialization with default file search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test tracer files
            for i in range(3):
                filepath = os.path.join(tmpdir, f"test{i:03d}.trace")
                self.create_test_tracer_file(filepath, npart=10)
            
            tracer = LeafsTracer("test", snappath=tmpdir)
            
            assert tracer.name == "test"
            assert tracer.snappath == tmpdir
            assert tracer.npart == 10
            assert tracer.nfiles == 3
            assert len(tracer.files) == 3
            assert len(tracer.starttimes) == 3

    def test_leafs_tracer_vartracer_parameter(self):
        """Test LeafsTracer with vartracer parameter."""
        with tempfile.NamedTemporaryFile(suffix='.trace', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            self.create_test_tracer_file(filepath)
            
            tracer = LeafsTracer("test", file=filepath, vartracer=False)
            
            assert tracer.vartracer == False
        finally:
            os.unlink(filepath)

    def test_leafs_tracer_readheader_with_mass(self):
        """Test readheader method with mass header."""
        with tempfile.NamedTemporaryFile(suffix='.trace', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            self.create_test_tracer_file(filepath, npart=100, has_mass_header=True)
            
            tracer = LeafsTracer("test", file=filepath)
            
            assert tracer.npart == 100
            assert tracer.mass == 1.0
        finally:
            os.unlink(filepath)

    def test_leafs_tracer_readheader_without_mass(self):
        """Test readheader method without mass header."""
        with tempfile.NamedTemporaryFile(suffix='.trace', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            self.create_test_tracer_file(filepath, npart=50, has_mass_header=False)
            
            tracer = LeafsTracer("test", file=filepath)
            
            assert tracer.npart == 50
            assert tracer.mass == 0.0
        finally:
            os.unlink(filepath)

    def test_leafs_tracer_no_files_found(self):
        """Test LeafsTracer when no tracer files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AssertionError, match="No tracer files found"):
                LeafsTracer("test", snappath=tmpdir)

    def test_leafs_tracer_multiple_files_sorted(self):
        """Test that multiple tracer files are found in order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in non-sequential order
            for i in [0, 2, 1]:
                filepath = os.path.join(tmpdir, f"model{i:03d}.trace")
                self.create_test_tracer_file(filepath)
            
            tracer = LeafsTracer("model", snappath=tmpdir)
            
            assert tracer.nfiles == 3
            # Files should be found in sequential order 000, 001, 002
            expected_files = [
                os.path.join(tmpdir, f"model{i:03d}.trace") for i in range(3)
            ]
            assert tracer.files == expected_files
