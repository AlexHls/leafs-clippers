# Tests for leafs-clippers

This directory contains unit tests for the core functionality of leafs-clippers.

## Running Tests

To run the tests, you need to have pytest installed. You can install it via:

```bash
# Using apt (Ubuntu/Debian)
sudo apt-get install python3-pytest

# Or using pip
pip install pytest
```

You also need to install the required dependencies:

```bash
# Using apt (Ubuntu/Debian)
sudo apt-get install python3-numpy python3-pandas python3-scipy python3-h5py python3-tqdm

# Or using pip
pip install numpy pandas scipy h5py tqdm
```

### Run all tests

```bash
# From the repository root
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/ -v
```

### Run specific test files

```bash
# Test constants module
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/test_const.py -v

# Test utilities module
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/test_utilities.py -v

# Test radioactive decay module
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/test_radioactivedecay.py -v

# Test LEAFS utilities module
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/test_leafs_utils.py -v

# Test LEAFS core module
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/test_leafs.py -v

# Test LEAFS tracer module
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/test_leafs_tracer.py -v

# Test LEAFS mapping module
PYTHONPATH=src:$PYTHONPATH python3 -m pytest tests/test_leafs_mapping.py -v
```

## Test Coverage

The tests cover the following modules:

- **util/const.py**: Physical constants and formatter functions
- **util/utilities.py**: Utility functions including dict/obj conversion, weighted statistics, and LazyDict
- **util/radioactivedecay.py**: RadioactiveDecay class for isotope decay calculations
- **leafs/utils.py**: LeafsXdmf3Writer for XDMF file generation
- **leafs/leafs.py**: Core LEAFS functions including snapshot list reading and protocol management
- **leafs/leafs_tracer.py**: Tracer file reading and bound/unbound particle calculation
- **leafs/leafs_mapping.py**: ARTIS model mapping and tracer mapping functionality

## Test Organization

Tests are organized by module:

- `test_const.py`: Tests for constants and formatter functions
- `test_utilities.py`: Tests for utility functions
- `test_radioactivedecay.py`: Tests for RadioactiveDecay class
- `test_leafs_utils.py`: Tests for LEAFS utilities (XDMF writer)
- `test_leafs.py`: Tests for LEAFS core module (snapshot management, protocols)
- `test_leafs_tracer.py`: Tests for LEAFS tracer module (file reading, bound/unbound particles)
- `test_leafs_mapping.py`: Tests for LEAFS mapping module (ARTIS model, isotope conversion)

Each test file contains multiple test classes, each focusing on a specific aspect of the module.
