"""Frequency response function as used in structural dynamics.

pyFRF is a port of sdypy-FRF (https://github.com/sdypy/sdypy-FRF), where the
code is maintained. This module re-exports sdypy.FRF under the pyFRF name.
"""
__version__ = '1.6.0'

from sdypy.FRF import FRF, assert_sep005, direction_dict
from sdypy.FRF.frf import (
    _EXC_TYPES,
    _RESP_TYPES,
    _FRF_TYPES,
    _FRF_FORM,
    _WGH_TYPES,
    _WINDOWS,
    _DIRECTIONS,
    _DIRECTIONS_NR,
)
