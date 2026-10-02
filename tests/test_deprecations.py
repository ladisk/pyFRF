"""
Tests for the frf_type -> frf_estimator / type -> frf_estimator / form -> frf_form
deprecation shims (frf_estimator/frf_form since 1.5.0).
"""

import pickle
import warnings

import numpy as np
import pytest

import pyFRF


def _make_frf(**kwargs):
    rng = np.random.default_rng(0)
    N = 256
    exc = rng.standard_normal((1, 1, N))
    resp = rng.standard_normal((1, 1, N))
    return pyFRF.FRF(sampling_freq=1000, exc=exc, resp=resp, exc_type='f',
                      resp_type='a', window='none', fft_len=N, **kwargs)


def test_new_ctor_keyword_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        frf = _make_frf(frf_estimator='H2')
    assert frf.frf_estimator == 'H2'


def test_old_ctor_keyword_warns_and_equals_new():
    with pytest.warns(DeprecationWarning):
        frf_old = _make_frf(frf_type='H2')
    frf_new = _make_frf(frf_estimator='H2')
    assert frf_old.frf_estimator == frf_new.frf_estimator == 'H2'


def test_frf_type_property_get_warns():
    frf = _make_frf(frf_estimator='Hv')
    with pytest.warns(DeprecationWarning):
        value = frf.frf_type
    assert value == 'Hv'


def test_frf_type_property_set_warns():
    frf = _make_frf()
    with pytest.warns(DeprecationWarning):
        frf.frf_type = 'H2'
    assert frf.frf_estimator == 'H2'


def test_get_FRF_positional_no_warning_and_equals_keyword_forms():
    frf = _make_frf()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        positional = frf.get_FRF('H1', 'accelerance')
        keyword_new = frf.get_FRF(frf_estimator='H1', frf_form='accelerance')
    np.testing.assert_array_equal(positional, keyword_new)


def test_get_FRF_old_keywords_warn_and_equal():
    frf = _make_frf()
    with pytest.warns(DeprecationWarning) as record:
        deprecated = frf.get_FRF(type='H1', form='accelerance')
    assert len(record) == 2
    new = frf.get_FRF(frf_estimator='H1', frf_form='accelerance')
    np.testing.assert_array_equal(deprecated, new)


def test_unpickle_frf_type_in_dict_moves_to_frf_estimator():
    frf = _make_frf(frf_estimator='H2')
    state = frf.__dict__.copy()
    state['frf_type'] = state.pop('frf_estimator')
    data = pickle.dumps(state)
    restored_state = pickle.loads(data)

    restored = pyFRF.FRF.__new__(pyFRF.FRF)
    restored.__setstate__(restored_state)

    assert restored.__dict__['frf_estimator'] == 'H2'
    assert 'frf_type' not in restored.__dict__
