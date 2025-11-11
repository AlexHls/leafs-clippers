"""Tests for leafs/leafs_mapping.py module."""
import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from leafs_clippers.leafs.leafs_mapping import (
    read_1d_artis_model,
    MappingTracer,
)


class TestRead1dArtisModel:
    """Test read_1d_artis_model function."""

    def create_test_artis_files(self, tmpdir, res=5, nradioactives=4, max_element=30):
        """Helper to create test ARTIS model files."""
        # Create model_1D.txt
        model_lines = [f"{res}\n", "1.0\n"]
        for i in range(res):
            vel = 1000.0 * (i + 1)
            logrho = 3.0 - i * 0.5
            ige = 0.1 * i
            radioactives = " ".join([f"{0.01 * j}" for j in range(nradioactives)])
            model_lines.append(f"{i} {vel} {logrho} {ige} {radioactives}\n")
        
        with open(os.path.join(tmpdir, "model_1D.txt"), 'w') as f:
            f.writelines(model_lines)
        
        # Create abundances_1D.txt
        abundance_lines = []
        for i in range(res):
            abundances = " ".join([f"{i}" if j == 0 else f"{0.01 * j}" for j in range(max_element)])
            abundance_lines.append(f"{i} {abundances}\n")
        
        with open(os.path.join(tmpdir, "abundances_1D.txt"), 'w') as f:
            f.writelines(abundance_lines)

    def test_read_1d_artis_model_basic(self):
        """Test basic reading of 1D ARTIS model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.create_test_artis_files(tmpdir, res=3)
            
            model = read_1d_artis_model(root_dir=tmpdir)
            
            assert model['data']['res'] == 3
            assert model['data']['time'] == 1.0
            assert len(model['data']['vel']) == 3
            assert len(model['data']['rho']) == 3
            assert len(model['data']['ige']) == 3

    def test_read_1d_artis_model_velocity_data(self):
        """Test that velocity data is read correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.create_test_artis_files(tmpdir, res=3)
            
            model = read_1d_artis_model(root_dir=tmpdir)
            
            # Velocity should be 1000, 2000, 3000
            assert np.isclose(model['data']['vel'][0], 1000.0)
            assert np.isclose(model['data']['vel'][1], 2000.0)
            assert np.isclose(model['data']['vel'][2], 3000.0)

    def test_read_1d_artis_model_density_conversion(self):
        """Test that log density is converted to linear density."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.create_test_artis_files(tmpdir, res=3)
            
            model = read_1d_artis_model(root_dir=tmpdir)
            
            # logrho values: 3.0, 2.5, 2.0
            # rho should be: 10^3.0, 10^2.5, 10^2.0
            assert np.isclose(model['data']['rho'][0], 1000.0)
            assert np.isclose(model['data']['rho'][1], 10**2.5)
            assert np.isclose(model['data']['rho'][2], 100.0)

    def test_read_1d_artis_model_zero_density(self):
        """Test handling of zero log density."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a model with zero log density
            model_lines = ["2\n", "1.0\n"]
            model_lines.append("0 1000.0 0.0 0.1 0.01 0.02 0.03 0.04\n")
            model_lines.append("1 2000.0 2.0 0.2 0.01 0.02 0.03 0.04\n")
            
            with open(os.path.join(tmpdir, "model_1D.txt"), 'w') as f:
                f.writelines(model_lines)
            
            # Create abundances
            abundance_lines = []
            for i in range(2):
                abundances = " ".join([f"{0.01 * j}" for j in range(30)])
                abundance_lines.append(f"{i} {abundances}\n")
            
            with open(os.path.join(tmpdir, "abundances_1D.txt"), 'w') as f:
                f.writelines(abundance_lines)
            
            model = read_1d_artis_model(root_dir=tmpdir)
            
            # Zero log density should result in zero density
            assert model['data']['rho'][0] == 0.0
            assert np.isclose(model['data']['rho'][1], 100.0)

    def test_read_1d_artis_model_radioactives(self):
        """Test that radioactive isotope data is read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.create_test_artis_files(tmpdir, res=2, nradioactives=3)
            
            model = read_1d_artis_model(root_dir=tmpdir, nradioactives=3)
            
            assert model['data']['radioactives'].shape == (2, 3)

    def test_read_1d_artis_model_abundances(self):
        """Test that abundance data is read correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.create_test_artis_files(tmpdir, res=3, max_element=20)
            
            model = read_1d_artis_model(root_dir=tmpdir, max_element=20)
            
            assert model['abundances'].shape == (3, 20)

    def test_read_1d_artis_model_custom_parameters(self):
        """Test reading with custom nradioactives and max_element."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.create_test_artis_files(tmpdir, res=4, nradioactives=5, max_element=25)
            
            model = read_1d_artis_model(
                root_dir=tmpdir,
                nradioactives=5,
                max_element=25
            )
            
            assert model['data']['res'] == 4
            assert model['data']['radioactives'].shape == (4, 5)
            assert model['abundances'].shape == (4, 25)

    def test_read_1d_artis_model_ige_values(self):
        """Test that IGE mass fraction is read correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.create_test_artis_files(tmpdir, res=3)
            
            model = read_1d_artis_model(root_dir=tmpdir)
            
            # IGE values should be 0.0, 0.1, 0.2
            assert np.isclose(model['data']['ige'][0], 0.0)
            assert np.isclose(model['data']['ige'][1], 0.1)
            assert np.isclose(model['data']['ige'][2], 0.2)


class TestMappingTracer:
    """Test MappingTracer class."""

    def test_mapping_tracer_convert_iso_string_neut(self):
        """Test isotope string conversion for neutron."""
        result = MappingTracer.convert_iso_string(b"neut")
        assert result == "n"

    def test_mapping_tracer_convert_iso_string_prot(self):
        """Test isotope string conversion for proton."""
        result = MappingTracer.convert_iso_string(b"prot")
        assert result == "p"

    def test_mapping_tracer_convert_iso_string_general(self):
        """Test general isotope string conversion."""
        result = MappingTracer.convert_iso_string(b"He 4")
        assert result == "he4"

    def test_mapping_tracer_convert_iso_string_lowercase(self):
        """Test that conversion makes strings lowercase."""
        result = MappingTracer.convert_iso_string(b"FE56")
        assert result == "fe56"

    def test_mapping_tracer_convert_iso_string_removes_spaces(self):
        """Test that conversion removes spaces."""
        result = MappingTracer.convert_iso_string(b"C 12")
        assert result == "c12"

    def test_mapping_tracer_initialization(self):
        """Test MappingTracer initialization."""
        # Create mocks for dependencies
        mock_snapshot = Mock()
        mock_snapshot._get_remnant_velocity.return_value = [0, 0, 0]
        
        mock_trajectory = Mock()
        mock_trajectory.posfin.return_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_trajectory.xiso.return_value = np.array([[0.5, 0.5], [0.3, 0.7]])
        mock_trajectory.isos = [b"he4", b"c12"]
        mock_trajectory.a = np.array([4, 12])
        mock_trajectory.z = np.array([2, 6])
        
        # Mock tracer reading
        with patch('leafs_clippers.leafs.leafs_mapping.lt.read_tracer') as mock_read:
            mock_tracer = Mock()
            mock_last = Mock()
            mock_last.data = np.array([[0, 0], [0, 0], [0, 0], [1.0, 2.0]])
            mock_tracer.last.return_value = mock_last
            mock_tracer.tmass = np.array([0.1, 0.2])
            mock_read.return_value = mock_tracer
            
            with patch('leafs_clippers.leafs.leafs_mapping.lt.get_bound_unbound_ids') as mock_bound:
                mock_bound.return_value = (np.array([]), np.array([1, 2]))
                
                mapping = MappingTracer(
                    snapshot=mock_snapshot,
                    trajectory=mock_trajectory,
                    snappath="/tmp",
                    model="test",
                    remove_bound_core=True
                )
                
                assert mapping._model == "test"
                assert mapping._snappath == "/tmp"
                assert mapping.npart == 2
                assert len(mapping.isos) == 2
                assert mapping.isos[0] == "he4"
                assert mapping.isos[1] == "c12"

    def test_mapping_tracer_initialization_no_bound_removal(self):
        """Test MappingTracer initialization without removing bound core."""
        mock_snapshot = Mock()
        mock_trajectory = Mock()
        mock_trajectory.posfin.return_value = np.array([[1.0, 2.0, 3.0]])
        mock_trajectory.xiso.return_value = np.array([[0.5, 0.5]])
        mock_trajectory.isos = [b"he4"]
        mock_trajectory.a = np.array([4])
        mock_trajectory.z = np.array([2])
        
        with patch('leafs_clippers.leafs.leafs_mapping.lt.read_tracer') as mock_read:
            mock_tracer = Mock()
            mock_last = Mock()
            mock_last.data = np.array([[0], [0], [0], [1.0]])
            mock_tracer.last.return_value = mock_last
            mock_tracer.tmass = np.array([0.1])
            mock_read.return_value = mock_tracer
            
            mapping = MappingTracer(
                snapshot=mock_snapshot,
                trajectory=mock_trajectory,
                snappath="/tmp",
                remove_bound_core=False
            )
            
            assert mapping.npart == 1
            assert mapping._remove_bound_core == False

    def test_mapping_tracer_remnant_threshold(self):
        """Test MappingTracer with custom remnant threshold."""
        mock_snapshot = Mock()
        mock_trajectory = Mock()
        mock_trajectory.posfin.return_value = np.array([[1.0, 2.0, 3.0]])
        mock_trajectory.xiso.return_value = np.array([[0.5, 0.5]])
        mock_trajectory.isos = [b"he4"]
        mock_trajectory.a = np.array([4])
        mock_trajectory.z = np.array([2])
        
        with patch('leafs_clippers.leafs.leafs_mapping.lt.read_tracer') as mock_read:
            mock_tracer = Mock()
            mock_last = Mock()
            mock_last.data = np.array([[0], [0], [0], [1.0]])
            mock_tracer.last.return_value = mock_last
            mock_tracer.tmass = np.array([0.1])
            mock_read.return_value = mock_tracer
            
            mapping = MappingTracer(
                snapshot=mock_snapshot,
                trajectory=mock_trajectory,
                snappath="/tmp",
                remove_bound_core=False,
                remnant_threshold=5e-4
            )
            
            assert mapping._remnant_threshold == 5e-4
