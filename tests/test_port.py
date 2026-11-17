"""Check that pyFRF exposes exactly the sdypy-FRF code."""
import pytest

import sdypy.FRF
import sdypy.FRF.frf
import sdypy.FRF.fft_tools
import pyFRF


@pytest.mark.parametrize("name", sdypy.FRF.__all__)
def test_top_level_names(name):
    assert getattr(pyFRF, name) is getattr(sdypy.FRF, name)


@pytest.mark.parametrize(
    "name",
    ["_EXC_TYPES", "_RESP_TYPES", "_FRF_TYPES", "_FRF_FORM", "_WGH_TYPES",
     "_WINDOWS", "_DIRECTIONS", "_DIRECTIONS_NR"],
)
def test_private_constants(name):
    assert getattr(pyFRF, name) is getattr(sdypy.FRF.frf, name)


def test_version():
    assert isinstance(pyFRF.__version__, str)


def test_fft_tools():
    import fft_tools
    from fft_tools import frequency_derivation, convert_frf

    assert frequency_derivation is sdypy.FRF.fft_tools.frequency_derivation
    assert convert_frf is sdypy.FRF.fft_tools.convert_frf
    assert fft_tools._FRF_TYPES is sdypy.FRF.fft_tools._FRF_TYPES
