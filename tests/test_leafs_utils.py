"""Tests for leafs/utils.py module."""
import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, MagicMock
from leafs_clippers.leafs.utils import LeafsXdmf3Writer


class TestLeafsXdmf3Writer:
    """Test LeafsXdmf3Writer class."""

    def create_mock_snapshot(self, gnx=10, gny=10, gnz=10, time=1.0):
        """Create a mock LeafsSnapshot for testing."""
        snapshot = Mock()
        snapshot.filename = "/path/to/snapshot.hdf5"
        snapshot.basename = "/path/to/snapshot"
        snapshot.gnx = gnx
        snapshot.gny = gny
        snapshot.gnz = gnz
        snapshot.time = time
        snapshot.keys = ['density', 'temp', 'pressure', 'velx', 'time', 'gnx']
        return snapshot

    def test_initialization_3d(self):
        """Test LeafsXdmf3Writer initialization with 3D snapshot."""
        snapshot = self.create_mock_snapshot(gnx=10, gny=10, gnz=10)
        writer = LeafsXdmf3Writer(snapshot)
        
        assert writer.snapshot == snapshot
        assert writer.filename == "snapshot.hdf5"
        assert writer.grid_shape == (11, 11, 11)
        assert writer.attribute_shape == (10, 10, 10)
        assert writer.twod is False
        assert writer.outname == "/path/to/snapshot.xdmf"

    def test_initialization_2d(self):
        """Test LeafsXdmf3Writer initialization with 2D snapshot."""
        snapshot = self.create_mock_snapshot(gnx=10, gny=10, gnz=1)
        writer = LeafsXdmf3Writer(snapshot)
        
        assert writer.grid_shape == (11, 11, 2)
        assert writer.attribute_shape == (10, 10, 1)
        assert writer.twod is True

    def test_ignore_keys_property(self):
        """Test _ignore_keys property."""
        snapshot = self.create_mock_snapshot()
        writer = LeafsXdmf3Writer(snapshot)
        
        ignore_keys = writer._ignore_keys
        expected_ignore = [
            "time", "gnx", "gny", "gnz", "geomx", "geomy", "geomz",
            "edgez", "edgey", "edgex", "ncells", "rad_wd", "rad_fl",
            "idx_wd", "idx_fl", "simulation_type"
        ]
        
        assert set(ignore_keys) == set(expected_ignore)

    def test_write_header(self):
        """Test _write_header method."""
        snapshot = self.create_mock_snapshot()
        writer = LeafsXdmf3Writer(snapshot)
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            writer._write_header(f)
            f.seek(0)
            content = f.read()
        
        assert '<?xml version="1.0" ?>' in content
        assert '<Xdmf Version="3.0">' in content
        assert '<Domain>' in content
        assert '<Grid Name="3DStructuredGrid" GridType="Uniform">' in content
        
        # Clean up
        os.unlink(f.name)

    def test_write_footer(self):
        """Test _write_footer method."""
        snapshot = self.create_mock_snapshot()
        writer = LeafsXdmf3Writer(snapshot)
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            writer._write_footer(f)
            f.seek(0)
            content = f.read()
        
        assert '</Grid>' in content
        assert '</Domain>' in content
        assert '</Xdmf>' in content
        
        # Clean up
        os.unlink(f.name)

    def test_write_time(self):
        """Test _write_time method."""
        snapshot = self.create_mock_snapshot(time=123.45)
        writer = LeafsXdmf3Writer(snapshot)
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            writer._write_time(f, snapshot.time)
            f.seek(0)
            content = f.read()
        
        assert '<Time Value="123.45" />' in content
        
        # Clean up
        os.unlink(f.name)

    def test_write_topology_3d(self):
        """Test _write_topology method for 3D grid."""
        snapshot = self.create_mock_snapshot(gnx=10, gny=20, gnz=30)
        writer = LeafsXdmf3Writer(snapshot)
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            writer._write_topology(f, writer.grid_shape)
            f.seek(0)
            content = f.read()
        
        assert '<Topology TopologyType="3DRectMesh"' in content
        assert 'Dimensions="11 21 31"' in content
        assert '</Topology>' in content
        
        # Clean up
        os.unlink(f.name)

    def test_write_attribute(self):
        """Test _write_attribute method."""
        snapshot = self.create_mock_snapshot(gnx=10, gny=10, gnz=10)
        writer = LeafsXdmf3Writer(snapshot)
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            writer._write_attribute(f, "density", writer.attribute_shape, "test.hdf5")
            f.seek(0)
            content = f.read()
        
        assert '<Attribute Name="density"' in content
        assert 'AttributeType="Scalar"' in content
        assert 'Center="Cell"' in content
        assert 'Dimensions="10 10 10"' in content
        assert 'test.hdf5:/density' in content
        assert '</Attribute>' in content
        
        # Clean up
        os.unlink(f.name)

    def test_write_geometry_3d(self):
        """Test _write_geometry method for 3D."""
        snapshot = self.create_mock_snapshot(gnx=10, gny=10, gnz=10)
        writer = LeafsXdmf3Writer(snapshot)
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            writer._write_geometry(f, writer.grid_shape, "test.hdf5")
            f.seek(0)
            content = f.read()
        
        assert '<Geometry GeometryType="VXVYVZ">' in content
        assert 'test.hdf5:/edgex' in content
        assert 'test.hdf5:/edgey' in content
        assert 'test.hdf5:/edgez' in content
        assert '</Geometry>' in content
        
        # Clean up
        os.unlink(f.name)

    def test_write_geometry_2d(self):
        """Test _write_geometry method for 2D."""
        snapshot = self.create_mock_snapshot(gnx=10, gny=10, gnz=1)
        writer = LeafsXdmf3Writer(snapshot)
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            writer._write_geometry(f, writer.grid_shape, "test.hdf5")
            f.seek(0)
            content = f.read()
        
        assert '<Geometry GeometryType="VXVYVZ">' in content
        # In 2D mode, the writer outputs edgey first, then edgex
        assert 'test.hdf5:/edgey' in content
        assert 'test.hdf5:/edgex' in content
        assert 'test.hdf5:/edgez' in content
        
        # Clean up
        os.unlink(f.name)

    def test_write_full_file(self):
        """Test complete write method."""
        snapshot = self.create_mock_snapshot()
        writer = LeafsXdmf3Writer(snapshot)
        
        # Create a temporary directory for the output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override the basename to write to temp directory
            writer.snapshot.basename = os.path.join(tmpdir, "test_snapshot")
            outname = writer.write()
            
            # Check that file was created
            expected_path = os.path.join(tmpdir, "test_snapshot.xdmf")
            assert os.path.exists(expected_path)
            
            # Read and validate content
            with open(expected_path, 'r') as f:
                content = f.read()
            
            assert '<?xml version="1.0" ?>' in content
            assert '<Xdmf Version="3.0">' in content
            assert '<Time Value="1.0" />' in content
            assert '<Topology' in content
            assert '<Geometry' in content
            # Attributes that should be written (not in ignore list)
            assert 'density' in content
            assert 'temp' in content
            assert 'pressure' in content
            # Attributes that should be ignored
            assert 'Name="time"' not in content
            assert 'Name="gnx"' not in content
