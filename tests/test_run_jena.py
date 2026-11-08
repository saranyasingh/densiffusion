"""Path extraction for the Jena solar-radiation benchmark."""

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

_scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_scripts))
_spec = importlib.util.spec_from_file_location("run_jena", _scripts / "run_jena.py")
jena = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jena)


def test_every_spacing_puts_the_fine_grid_on_the_measurements():
    for spacing in jena.TASKS.values():
        assert spacing % (jena.FINE * 10) == 0


def test_extract_paths_skips_missing_values_and_stays_in_its_years():
    start = datetime(2004, 1, 1)
    values = np.arange(3 * 52_560, dtype=float)  # about three years of 10 min
    values[100:105] = np.nan
    paths, times, offsets = jena.extract_paths(
        values, start, 10, (2004, 2005), spacing=80, count=5
    )
    assert paths.shape == (5, 257)
    assert not np.isnan(paths).any()
    np.testing.assert_allclose(np.diff(times), 10 / 60)
    # Each fine step is one 10-minute reading; the first path moved past the gap.
    np.testing.assert_array_equal(paths[:, 1] - paths[:, 0], 1)
    assert offsets[0] == 105
    assert offsets[-1] + 256 < (datetime(2006, 1, 1) - start).days * 144
