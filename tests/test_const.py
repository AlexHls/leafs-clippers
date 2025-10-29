"""Tests for util/const.py module."""
import pytest
from leafs_clippers.util.const import (
    C_LIGHT,
    H_ERG,
    N_A,
    M_E,
    M_U,
    K_B,
    C_AA,
    M_SOL,
    R_SOL,
    G_GRAV,
    FLOAT_FMT,
    INT_FMT,
    KEY_TO_LABEL_DICT,
    KEY_TO_CMAP_DICT,
    KEY_TO_FMT_DICT,
)


class TestPhysicalConstants:
    """Test physical constants are defined correctly."""

    def test_speed_of_light(self):
        """Test speed of light constant."""
        assert C_LIGHT == 299792.458  # km/s
        assert isinstance(C_LIGHT, float)

    def test_planck_constant(self):
        """Test Planck's constant."""
        assert H_ERG == 6.62607015e-27  # erg.s
        assert isinstance(H_ERG, float)

    def test_avogadro_number(self):
        """Test Avogadro's number."""
        assert N_A == 6.02214076e23  # mol^-1
        assert isinstance(N_A, float)

    def test_electron_mass(self):
        """Test electron mass."""
        assert M_E == 9.10938188e-28  # g
        assert isinstance(M_E, float)

    def test_atomic_mass_unit(self):
        """Test atomic mass unit."""
        assert M_U == 1.660538782e-24  # g
        assert isinstance(M_U, float)

    def test_boltzmann_constant(self):
        """Test Boltzmann constant."""
        assert K_B == 1.3806504e-16
        assert isinstance(K_B, float)

    def test_speed_of_light_angstrom(self):
        """Test speed of light in angstroms per second."""
        assert C_AA == 299792458 * 1.0e10  # AA/s
        assert isinstance(C_AA, float)

    def test_solar_mass(self):
        """Test solar mass."""
        assert M_SOL == 1.989e33  # g
        assert isinstance(M_SOL, float)

    def test_solar_radius(self):
        """Test solar radius."""
        assert R_SOL == 6.955e10  # cm
        assert isinstance(R_SOL, float)

    def test_gravitational_constant(self):
        """Test gravitational constant."""
        assert G_GRAV == 6.67259e-8  # cm^3 g^-1 s^-2
        assert isinstance(G_GRAV, float)


class TestFormatterFunctions:
    """Test formatter functions."""

    def test_float_fmt_positive(self):
        """Test FLOAT_FMT with positive number."""
        result = FLOAT_FMT(3.14159, None)
        assert result == "3.14"

    def test_float_fmt_negative(self):
        """Test FLOAT_FMT with negative number."""
        result = FLOAT_FMT(-2.71828, None)
        assert result == "-2.72"

    def test_float_fmt_zero(self):
        """Test FLOAT_FMT with zero."""
        result = FLOAT_FMT(0.0, None)
        assert result == "0.00"

    def test_float_fmt_large(self):
        """Test FLOAT_FMT with large number."""
        result = FLOAT_FMT(123456.789, None)
        assert result == "123456.79"

    def test_int_fmt_positive(self):
        """Test INT_FMT with positive integer."""
        result = INT_FMT(42, None)
        assert result == "42"

    def test_int_fmt_zero(self):
        """Test INT_FMT with zero."""
        result = INT_FMT(0, None)
        assert result == "0"

    def test_int_fmt_negative(self):
        """Test INT_FMT with negative integer."""
        result = INT_FMT(-17, None)
        assert result == "-17"

    def test_int_fmt_float_conversion(self):
        """Test INT_FMT converts float to int."""
        result = INT_FMT(42.9, None)
        assert result == "42"


class TestDictionaries:
    """Test configuration dictionaries."""

    def test_key_to_label_dict_keys(self):
        """Test KEY_TO_LABEL_DICT has expected keys."""
        expected_keys = [
            "density", "temp", "pressure", "velx", "vely", "velz",
            "ye", "xnuc01", "xnuc02", "xnuc03", "xnuc04", "xnuc05",
            "xnuc06", "energy", "Amean", "q_sgs"
        ]
        assert set(KEY_TO_LABEL_DICT.keys()) == set(expected_keys)

    def test_key_to_label_dict_values(self):
        """Test KEY_TO_LABEL_DICT values are strings."""
        for value in KEY_TO_LABEL_DICT.values():
            assert isinstance(value, str)

    def test_key_to_cmap_dict_keys(self):
        """Test KEY_TO_CMAP_DICT has expected keys."""
        expected_keys = [
            "density", "temp", "pressure", "velx", "vely", "velz",
            "ye", "xnuc01", "xnuc02", "xnuc03", "xnuc04", "xnuc05",
            "xnuc06", "energy", "mach"
        ]
        assert set(KEY_TO_CMAP_DICT.keys()) == set(expected_keys)

    def test_key_to_cmap_dict_values(self):
        """Test KEY_TO_CMAP_DICT values are valid colormap names."""
        valid_cmaps = [
            "inferno", "afmhot", "viridis", "coolwarm", "cividis",
            "ocean", "magma", "BuPu_r"
        ]
        for value in KEY_TO_CMAP_DICT.values():
            assert isinstance(value, str)
            assert value in valid_cmaps

    def test_key_to_fmt_dict_keys(self):
        """Test KEY_TO_FMT_DICT has expected keys."""
        expected_keys = [
            "density", "temp", "pressure", "velx", "vely", "velz",
            "ye", "xnuc01", "xnuc02", "xnuc03", "xnuc04", "xnuc05",
            "xnuc06", "Amean"
        ]
        assert set(KEY_TO_FMT_DICT.keys()) == set(expected_keys)

    def test_key_to_fmt_dict_values(self):
        """Test KEY_TO_FMT_DICT values are callable."""
        for value in KEY_TO_FMT_DICT.values():
            assert callable(value)

    def test_key_to_fmt_dict_amean_uses_int_fmt(self):
        """Test that Amean uses INT_FMT."""
        assert KEY_TO_FMT_DICT["Amean"] == INT_FMT

    def test_key_to_fmt_dict_others_use_float_fmt(self):
        """Test that other keys use FLOAT_FMT."""
        for key in ["density", "temp", "pressure", "velx"]:
            assert KEY_TO_FMT_DICT[key] == FLOAT_FMT
