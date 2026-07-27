"""
Unit tests for pyFRF.py
"""

import matplotlib.pyplot as plt # dg

import numpy as np
import scipy
from scipy.signal import detrend
import pyExSi
import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import model
import pyFRF

import pytest

# function used to acquire true systems FRF:
def get_true_FRF(T=10, fs=300, ndof=3):

    time = np.arange(0, T, 1/fs)
    freq = np.fft.rfftfreq(len(time), 1/fs)
    omega = 2*np.pi*freq

    m_ = np.arange(1, 1+ndof, 1)
    c_ = np.ones(len(m_)+1, dtype=np.float64) * 60 # N/m/s
    k_ = np.ones(len(m_)+1, dtype=np.float64) * 150000 # N/m

    # MDOF model:
    fr_, xi_, eig = model.modal_model(m_, k_, c_)
    FRF = model.FRF_matrix(omega, *eig)

    return FRF, freq, time


# function used to acquire response via true FRF matrix and known excitation:
def get_response(exc, FRF_matrix, exc_dofs, resp_dofs):
    """
    exc: excitation array
    FRF_matrix: true system FRF matrix
    exc_dofs: list of known excitation dof
    resp_dofs: list of wanted response dof
    """
    
    x = model.compute_response(FRF_matrix, exc, exc_dofs)
    return x[:,resp_dofs]





# TEST FUNCTIONS:
def test_FRF_SISO():
    # get true FRF matrix, freq and time:
    FRF_matrix, freq, t = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0] = 1.
    x = np.zeros_like(f)
    x[0,0] = np.fft.irfft(FRF_matrix[0, 0])

    # get relevant FRFs from FRF matrix based on excitation dofs and response dofs:
    true_frf = np.zeros((len(resp_dofs), len(exc_dofs), FRF_matrix.shape[2]), dtype="complex128")
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            true_frf[i,j] = FRF_matrix[resp_dofs[i],exc_dofs[j]]
    
    # get FRF from pyFRF:
    pyFRF_obj = pyFRF.FRF(sampling_freq=int(1/t[1]), exc=f, resp=x, 
                                  window="none", exc_type='f', resp_type='d', 
                                  nperseg=None, noverlap=None, fft_len=None)
    H1 = pyFRF_obj.get_H1()
    H2 = pyFRF_obj.get_H2()
    Hv = pyFRF_obj.get_Hv()
    
    for H in [H1, H2, Hv]:
        # test frf amplitudes:
        np.testing.assert_allclose(np.abs(H[:, :, 1:-1]), np.abs(true_frf[:,:,1:-1]), 
                                rtol=1e-04, atol=1e-06)
                
        # test frf phase:
        np.testing.assert_allclose(np.angle(H[:, :, 1:-1]), np.angle(true_frf[:,:,1:-1]), 
                        rtol=1e-04, atol=1e-06)


def test_FRF_SIMO():
    # get true FRF matrix, freq and time:
    FRF_matrix, freq, t = get_true_FRF()

    # define excitation and get response (SIMO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0,1,2]  # multiple outputs
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0] = 1.
    x = np.zeros((n_measurements, len(resp_dofs), t.shape[0]))
    for i in range(len(resp_dofs)):
        x[0, i, :] = np.fft.irfft(FRF_matrix[resp_dofs[i], exc_dofs[0]])


    # get relevant FRFs from FRF matrix based on excitation dofs and response dofs:
    true_frf = np.zeros((len(resp_dofs), len(exc_dofs), FRF_matrix.shape[2]), dtype="complex128")
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            true_frf[i,j] = FRF_matrix[resp_dofs[i],exc_dofs[j]]
    
    # get FRF from pyFRF:
    pyFRF_obj = pyFRF.FRF(sampling_freq=int(1/t[1]), exc=f, resp=x, 
                                  window="none", exc_type='f', resp_type='d', 
                                  nperseg=None, noverlap=None, fft_len=None)
    H1 = pyFRF_obj.get_H1()
    H2 = pyFRF_obj.get_H2()
    Hv = pyFRF_obj.get_Hv()

    for H in [H1, H2, Hv]:
        # test frf amplitudes:
        np.testing.assert_allclose(np.abs(H[:, :, 1:-1]), np.abs(true_frf[:,:,1:-1]), 
                                rtol=1e-03, atol=1e-06)
                
        # test frf phase:
        np.testing.assert_allclose(np.angle(H[:, :, 1:-1]), np.angle(true_frf[:,:,1:-1]), 
                        rtol=1e-03, atol=1e-06)
            

def test_FRF_MIMO():
    # get true FRF matrix, freq and time:
    T = 10  
    fs = 300

    T_welch = 1
    N_welch = int(T_welch*fs)

    FRF_matrix, freq, t = get_true_FRF(T=T, fs=fs)
    print(FRF_matrix.shape)
    FRF_matrix_w, freq_w, t_w = get_true_FRF(T=T_welch, fs=fs)

    # define excitation and get response (MISO):
    n_measurements = 10
    exc_dofs = [0, 1]  # multiple inputs
    resp_dofs = [0, 1, 2]  # single output
    freq_lower = 0 # PSD lower frequency limit  [Hz]
    freq_upper = 300 # PSD upper frequency limit [Hz]
    PSD = pyExSi.get_psd(freq, freq_lower, freq_upper) # one-sided flat-shaped PSD
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            f[i][j] = pyExSi.random_gaussian(f.shape[-1], PSD, fs)
    x = get_response(f, FRF_matrix, exc_dofs, resp_dofs)

    # get relevant FRFs from FRF matrix based on excitation dofs and response dofs:
    true_frf = np.zeros((len(resp_dofs), len(exc_dofs), FRF_matrix_w.shape[2]), dtype="complex128")
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            true_frf[i, j] = FRF_matrix_w[i, j]
    
    # get FRF from pyFRF:
    print(f.shape, x.shape, N_welch)
    pyFRF_obj = pyFRF.FRF(sampling_freq=fs, exc=f, resp=x, 
                                window="hann", exc_type='f', resp_type='d', 
                                nperseg=N_welch, noverlap=N_welch//2, fft_len=N_welch,
                                anyltical_inverse=False)
    H1 = pyFRF_obj.get_H1()
    H2 = pyFRF_obj.get_H2()

    for H in [H1, H2]:
        # test frf amplitudes:
        np.testing.assert_allclose(np.abs(H[:, :, 1:-1]), np.abs(true_frf[:,:,1:-1]), 
                                rtol=5e-01, atol=1e-06)
                
        # test frf phase:
        np.testing.assert_allclose(np.angle(H[:, :, 1:-1]),
                                   np.angle(true_frf[:,:,1:-1]),
                                   rtol=5e-1, atol=3e-01)


def test_FRF_SIMO_add_all_equals_per_channel():
    """Regression test for issue #20.

    For a single-input (SIMO) measurement the FRF must be identical whether all
    response channels are added in one call or channel-by-channel. The performance
    fix in ``_get_frf_av`` (diagonal-only ``S_XX`` for single input and ``S_FX``
    mirrored from ``S_XF``) must not change the numerical result. It also checks
    the two structural invariants the fix relies on.
    """
    T = 10
    fs = 300
    T_welch = 1
    N_welch = int(T_welch * fs)

    FRF_matrix, freq, t = get_true_FRF(T=T, fs=fs)

    n_measurements = 5
    exc_dofs = [0]             # single input -> SIMO
    resp_dofs = [0, 1, 2]      # multiple outputs
    PSD = pyExSi.get_psd(freq, 0, fs / 2)  # flat one-sided PSD
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            f[i][j] = pyExSi.random_gaussian(f.shape[-1], PSD, fs)
    x = get_response(f, FRF_matrix, exc_dofs, resp_dofs)

    common = dict(sampling_freq=fs, window="hann", exc_type='f', resp_type='d',
                  nperseg=N_welch, noverlap=N_welch // 2, fft_len=N_welch)

    # (A) all response channels added at once
    frf_all = pyFRF.FRF(**common)
    frf_all.add_data(f[:, 0, :], x)

    # (B) each response channel added on its own
    per_channel = []
    for ch in range(len(resp_dofs)):
        obj = pyFRF.FRF(**common)
        obj.add_data(f[:, 0, :], x[:, ch, :])
        per_channel.append(obj)

    for est in ['H1', 'H2', 'Hv']:
        H_all = frf_all.get_FRF(est)
        H_ref = np.concatenate([o.get_FRF(est) for o in per_channel], axis=0)
        np.testing.assert_allclose(H_all, H_ref, rtol=1e-12, atol=1e-12)

    coh_all = frf_all.get_coherence()
    coh_ref = np.concatenate([o.get_coherence() for o in per_channel], axis=0)
    np.testing.assert_allclose(coh_all, coh_ref, rtol=1e-12, atol=1e-12)

    # invariant 1: S_FX is the conjugate transpose of S_XF
    np.testing.assert_allclose(frf_all.S_FX, np.conj(frf_all.S_XF).transpose(1, 0, 2),
                               rtol=0, atol=1e-20)
    # invariant 2: single-input S_XX is stored compactly as the response auto-spectra
    # (n_resp, 1, freq), but the public S_XX property stays backward compatible: it
    # returns the full (n_resp, n_resp, freq) matrix with zeros off-diagonal.
    freq_len = np.fft.rfftfreq(N_welch, 1. / fs).shape[0]
    n_resp = len(resp_dofs)
    assert frf_all._S_XX.shape == (n_resp, 1, freq_len)          # compact internal store
    assert frf_all.S_XX.shape == (n_resp, n_resp, freq_len)      # full public matrix
    # diagonal holds the auto-spectra, off-diagonal is zero
    for i in range(n_resp):
        np.testing.assert_allclose(frf_all.S_XX[i, i, :], frf_all._S_XX[i, 0, :])
    off_diagonal = frf_all.S_XX.copy()
    for i in range(n_resp):
        off_diagonal[i, i, :] = 0
    assert np.all(off_diagonal == 0)


def test_MIMO_cross_spectra_hermitian():
    """Regression test for the Hermitian upper-triangle optimization (issue #20).

    For multiple inputs ``S_XX`` and ``S_FF`` are built from their upper triangle
    only and mirrored with a conjugate. Verify the stored matrices are Hermitian
    (``M[i,j] == conj(M[j,i])``, which also forces a real diagonal) and that the
    off-diagonal response cross-spectra are actually populated by the mirror.
    """
    T = 4
    fs = 300
    N_welch = fs  # 1 s Welch segments

    FRF_matrix, freq, t = get_true_FRF(T=T, fs=fs)

    n_measurements = 4
    exc_dofs = [0, 1]          # multiple inputs -> MIMO
    resp_dofs = [0, 1, 2]
    PSD = pyExSi.get_psd(freq, 0, fs / 2)
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            f[i][j] = pyExSi.random_gaussian(f.shape[-1], PSD, fs)
    x = get_response(f, FRF_matrix, exc_dofs, resp_dofs)

    obj = pyFRF.FRF(sampling_freq=fs, exc=f, resp=x, window="hann",
                    exc_type='f', resp_type='d',
                    nperseg=N_welch, noverlap=N_welch // 2, fft_len=N_welch)

    for M in (obj.S_XX, obj.S_FF):
        np.testing.assert_allclose(M, np.conj(M).transpose(1, 0, 2), rtol=1e-9, atol=1e-12)

    # the mirror actually filled the lower/upper off-diagonal response cross-spectra
    assert np.any(np.abs(obj.S_XX[0, 1, :]) > 0)
    assert np.any(np.abs(obj.S_XX[1, 0, :]) > 0)


def test_freq():
    # get true FRF matrix, freq and time:
    FRF_matrix, freq, t = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:50] = 50 * np.sin(2*np.pi*np.arange(0,50,1)/100)
    x = get_response(f, FRF_matrix, exc_dofs, resp_dofs)

    # define sampling frequency and length of fft:
    sampling_freq = int(1/t[1])
    fft_len=8000

    # create a test object:
    test_object = pyFRF.FRF(sampling_freq=sampling_freq, exc=f, resp=x, 
                             window="none", exc_type='f', resp_type='d', 
                             nperseg=None, noverlap=None, fft_len=fft_len)

    # test:
    np.testing.assert_equal(np.arange(0, sampling_freq/2+sampling_freq/fft_len, sampling_freq/fft_len), 
                            test_object.get_f_axis())
    

def test_w():
    # get true FRF matrix, freq and time:
    FRF_matrix, freq, t = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:50] = 50 * np.sin(2*np.pi*np.arange(0,50,1)/100)
    x = get_response(f, FRF_matrix, exc_dofs, resp_dofs)

    # define sampling frequency and length of fft:
    sampling_freq = int(1/t[1])
    fft_len=8000

    # create a test object:
    test_object = pyFRF.FRF(sampling_freq=sampling_freq, exc=f, resp=x, 
                                    window="none", exc_type='f', resp_type='d', 
                                    nperseg=None, noverlap=None, fft_len=fft_len)

    # test:
    np.testing.assert_equal(2*np.pi*np.arange(0, sampling_freq/2+sampling_freq/fft_len, sampling_freq/fft_len), 
                            test_object.get_w_axis())
    


def test_t():
    # get true FRF matrix, freq and time:
    FRF_matrix, freq, t = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:0] = 1.
    x = get_response(f, FRF_matrix, exc_dofs, resp_dofs)

    # define sampling frequency and length of fft:
    sampling_freq = int(1/t[1])

    # create a test object:
    test_object = pyFRF.FRF(sampling_freq=sampling_freq, exc=f, resp=x, 
                             window="none", exc_type='f', resp_type='d', 
                             nperseg=None, noverlap=None, fft_len=None)
    
    # test:
    np.testing.assert_allclose(np.arange(len(t)) / sampling_freq, 
                                   test_object.get_t_axis())
    

def test_df():
    # get true FRF matrix, freq and time:
    FRF_matrix, freq, t = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:50] = 50 * np.sin(2*np.pi*np.arange(0,50,1)/100)
    x = get_response(f, FRF_matrix, exc_dofs, resp_dofs)

    # define sampling frequency and length of fft:
    sampling_freq = int(1/t[1])
    fft_len=8000

    # create a test object:
    test_object = pyFRF.FRF(sampling_freq=sampling_freq, exc=f, resp=x, 
                             window="none", exc_type='f', resp_type='d', 
                             nperseg=None, noverlap=None, fft_len=fft_len)

    #test:
    np.testing.assert_equal(sampling_freq/fft_len, 
                            test_object.get_df())
    

def test_double_impact():
    # create a test object:
    test_object = pyFRF.FRF(sampling_freq=1000, fft_len=500)

    # define excitation
    f = np.zeros(1000)

    # single impact:
    f[0] = 1
    # test single impact
    np.testing.assert_equal(test_object._is_double_impact(f), False)

    # add another impact:
    f[100] = 0.5
    # test double impact:
    np.testing.assert_equal(test_object._is_double_impact(f), True)


def test_overflow():
    # create a test object:
    test_object = pyFRF.FRF(sampling_freq=1000, fft_len=500)

    # define exponentially falling signal:
    x = np.exp(np.log(0.2) * (np.arange(1000)) / (1000 - 1))
    # test no overflow:
    np.testing.assert_equal(test_object._is_overflow(x), False)

    # add overflow (3x same max number):
    x [50] = 1
    x [100] = 1
    # test overflow:
    np.testing.assert_equal(test_object._is_overflow(x), True)


def test_is_data_ok():
    # create a test object:
    test_object = pyFRF.FRF(sampling_freq=1000, fft_len=500)
    
    # no overflow, no double impact:
    x = np.exp(np.log(0.2) * (np.arange(1000)) / (1000 - 1))
    f = np.zeros(1000)
    f[0] = 1
    # test no overflow, no double impact - data ok:
    np.testing.assert_equal(test_object.is_data_ok(f, x), True)

    # add only overflow:
    x = np.exp(np.log(0.2) * (np.arange(1000)) / (1000 - 1))
    x[50] = 1
    x[100] = 1
    f = np.zeros(1000)
    f[0] = 1
    # test only overflow, no double impact - data not ok:
    np.testing.assert_equal(test_object.is_data_ok(f, x), False)

    # add only double impact:
    x = np.exp(np.log(0.2) * (np.arange(1000)) / (1000 - 1))
    f = np.zeros(1000)
    f[0] = 1
    f[100] = 0.5
    # test no overflow, only double impact - data not ok:
    np.testing.assert_equal(test_object.is_data_ok(f, x), False)

    # both overflow and double impact:
    x = np.exp(np.log(0.2) * (np.arange(1000)) / (1000 - 1))
    x[50] = 1
    x[100] = 1
    f = np.zeros(1000)
    f[0] = 1
    f[100] = 0.5
    # test both overflow and double impact - data not ok:
    np.testing.assert_equal(test_object.is_data_ok(f, x), False)


def test_analytical_matrix_inverse():
    # create a test object:
    test_object = pyFRF.FRF(sampling_freq=1000, fft_len=500, analytical_inverse=True)

    # 2x2 matrix inverse:
    A = np.array([[1, 2], 
                  [3, 4]])
    # test
    np.testing.assert_allclose(test_object._matrix_inverse(A), 
                                   np.linalg.inv(A))
    
    # 3x3 matrix inverse:
    A = np.array([[1, 2, 3], 
                  [2, 1, 3], 
                  [3, 2, 1]])
    # test:
    np.testing.assert_allclose(test_object._matrix_inverse(A), 
                                   np.linalg.inv(A))
    

# SEP-005 compatibility tests:
def generate_sep005_ts(fs=1000, n_samples=100, n_channels=1, quantity='f'):
    """
    Helper: Generate sep-005 dict with synthetic data.
    """
    time = np.linspace(0, 1, n_samples)
    if n_channels == 1:
        data = np.sin(2 * np.pi * 5 * time)
    else:
        data = np.array([np.sin(2 * np.pi * (5 + i) * time) for i in range(n_channels)])
    return {
        'data': data,
        'fs': fs,
        'quantity': quantity,
        'name': 'test_signal',
        'unit_str': 'm/s2' if quantity == 'a' else 'N',
    }


def test_sep005_single_channel():
    """
    Basic test: Single-channel input
    """
    exc_ts = generate_sep005_ts(quantity='f')
    resp_ts = generate_sep005_ts(quantity='a')

    frf = pyFRF.FRF(sampling_freq=None, exc=exc_ts, resp=resp_ts)

    assert frf.exc.shape == (1, 1, 100)
    assert frf.resp.shape == (1, 1, 100)
    assert frf.exc_type == 'f'
    assert frf.resp_type == 'a'
    assert frf.exc_sampling_freq == 1000
    assert frf.resp_sampling_freq == 1000


def test_sep005_list_of_single_channel():
    """
    Test: Multiple timeseries, single-channel each
    """
    exc_ts_list = [generate_sep005_ts(quantity='f') for _ in range(3)]
    resp_ts_list = [generate_sep005_ts(quantity='a') for _ in range(3)]

    frf = pyFRF.FRF(sampling_freq=None, exc=exc_ts_list, resp=resp_ts_list)

    assert frf.exc.shape == (3, 1, 100)
    assert frf.resp.shape == (3, 1, 100)


def test_sep005_multi_channel_input():
    """
    Test: Single timeseries with multi-channel data
    """
    exc_ts = generate_sep005_ts(n_channels=2, quantity='f')
    resp_ts = generate_sep005_ts(n_channels=2, quantity='a')

    frf = pyFRF.FRF(sampling_freq=None, exc=exc_ts, resp=resp_ts)

    assert frf.exc.shape == (1, 2, 100)
    assert frf.resp.shape == (1, 2, 100)


def test_sep005_list_of_multi_channel():
    """
    Test: List of multi-channel inputs (full MIMO case)
    """
    exc_ts_list = [generate_sep005_ts(n_channels=2, quantity='f') for _ in range(3)]
    resp_ts_list = [generate_sep005_ts(n_channels=4, quantity='a') for _ in range(3)]

    frf = pyFRF.FRF(sampling_freq=None, exc=exc_ts_list, resp=resp_ts_list)

    assert frf.exc.shape == (3, 2, 100)
    assert frf.resp.shape == (3, 4, 100)


def test_sep005_mismatched_quantity():
    """
    Test: Mismatched quantities in excitation and response
    """
    exc_ts = generate_sep005_ts(quantity='f')
    resp_ts_1 = generate_sep005_ts(quantity='a')
    resp_ts_2 = generate_sep005_ts(quantity='v')
    with pytest.raises(ValueError):
        pyFRF.FRF(sampling_freq=None, exc=exc_ts, resp=[resp_ts_1, resp_ts_2])

def test_sep005_unsupported_quantity():
    """
    Test: Unsupported quantity type
    """
    ts = generate_sep005_ts(quantity='x')  # Invalid quantity
    with pytest.raises(ValueError):
        pyFRF.FRF(sampling_freq=None, exc=ts, resp=ts)

def test_add_data_with_sep005():
    """
    Test: Adding new data to an existing pyFRF object
    """
    # Initial signals: 2 measurements, single channel each
    exc_ts_list = [generate_sep005_ts(quantity='f') for _ in range(2)]
    resp_ts_list = [generate_sep005_ts(quantity='a') for _ in range(2)]
    
    frf = pyFRF.FRF(sampling_freq=None, exc=exc_ts_list, resp=resp_ts_list)
    assert frf.total_meas == 2  # two measurements

    # New signals: 1 measurement, single channel
    new_exc_ts = generate_sep005_ts(quantity='f')
    new_resp_ts = generate_sep005_ts(quantity='a')

    frf.add_data(exc=new_exc_ts, resp=new_resp_ts)

    # Now 3 measurements should exist
    assert frf.total_meas == 3

    # Confirm time dimension consistency
    assert frf.exc.shape[-1] == new_exc_ts['data'].shape[-1]
    assert frf.resp.shape[-1] == new_resp_ts['data'].shape[-1]


def _assert_frf_close_where_finite(H, ref, rtol=1e-6, atol=1e-6, max_nonfinite_frac=0.01):
    """Assert estimated FRF ``H`` matches ``ref`` on all finite bins.

    The H2 and Hv estimators are mathematically undefined at response nodes
    (where the response, and hence the response-excitation cross-spectrum, is
    zero), producing isolated NaN/inf bins; ``ref`` is finite everywhere. Compare
    on the finite bins and require that almost all bins are finite so a degenerate
    (all-NaN) estimate cannot pass.
    """
    finite = np.isfinite(H)
    assert finite.mean() > 1 - max_nonfinite_frac
    np.testing.assert_allclose(H[finite], ref[finite], rtol=rtol, atol=atol)


def test_FRF_SIMO_pylump():
    """SIMO FRF cross-validated against an independent pyLump MDOF model.

    A mass-spring-damper model (pyLump) synthesises the exact response of a
    single-input / multiple-output system to a broadband excitation; pyFRF must
    recover the frequency response function that generated it. The reference is
    the FRF returned by ``Model.get_response(return_matrix=True)`` -- i.e. the
    exact transfer function the response was built from (pyLump's ``get_response``
    and ``get_FRF_matrix`` differ by one frequency bin, so the used matrix is the
    self-consistent ground truth for the generated data).
    """
    pyLump = pytest.importorskip("pyLump")
    rng = np.random.default_rng(1)
    n_dof, fs, N = 5, 2000, 8000
    model = pyLump.Model(n_dof, mass=1.0, stiffness=2e4, damping=2.0, boundaries="both")

    exc_dof, resp_dof = [0], [0, 1, 2, 3, 4]
    u = rng.standard_normal(N)
    resp, H_used = model.get_response(exc_dof, u, fs, resp_dof=resp_dof,
                                      domain='f', return_matrix=True)
    ref = H_used[np.ix_(resp_dof, exc_dof)]          # (n_resp, n_exc, freq), receptance

    frf = pyFRF.FRF(sampling_freq=fs, exc=u[None, None, :], resp=resp[None, :, :],
                    exc_type='f', resp_type='d', window='none',
                    nperseg=N, noverlap=0, fft_len=N)

    sl = slice(1, -1)                                # skip DC and Nyquist bins
    ref_s = ref[:, :, sl]
    for est in ('H1', 'H2', 'Hv'):
        H = frf.get_FRF(est)
        assert H.shape == ref.shape
        _assert_frf_close_where_finite(H[:, :, sl], ref_s)
    # H1 is well-defined at every frequency line for a single input
    assert np.all(np.isfinite(frf.get_FRF('H1')[:, :, sl]))

    # noise-free single measurement -> coherence is unity where defined
    coh = np.abs(frf.get_coherence()[:, sl])
    finite = np.isfinite(coh)
    assert finite.mean() > 0.99
    np.testing.assert_allclose(coh[finite], 1.0, atol=1e-6)


def test_FRF_MIMO_pylump():
    """MIMO FRF cross-validated against an independent pyLump MDOF model.

    Two inputs, four outputs, several linearly independent excitation records so
    the input cross-spectral matrix is invertible. pyFRF's H1/H2 estimators must
    recover the FRF used to synthesise the responses.
    """
    pyLump = pytest.importorskip("pyLump")
    rng = np.random.default_rng(7)
    n_dof, fs, N = 4, 2000, 6000
    model = pyLump.Model(n_dof, mass=[1, 1.5, 1.2, 0.8], stiffness=1.5e4,
                         damping=3.0, boundaries="both")

    exc_dof, resp_dof = [0, 2], [0, 1, 2, 3]
    n_meas = 6                                       # >= number of inputs
    exc = np.empty((n_meas, len(exc_dof), N))
    resp = np.empty((n_meas, len(resp_dof), N))
    H_used = None
    for m in range(n_meas):
        u = rng.standard_normal((len(exc_dof), N))
        r, H_used = model.get_response(exc_dof, u, fs, resp_dof=resp_dof,
                                       domain='f', return_matrix=True)
        exc[m], resp[m] = u, r
    ref = H_used[np.ix_(resp_dof, exc_dof)]          # (n_resp, n_exc, freq)

    frf = pyFRF.FRF(sampling_freq=fs, exc=exc, resp=resp, exc_type='f', resp_type='d',
                    window='none', nperseg=N, noverlap=0, fft_len=N)

    sl = slice(1, -1)
    ref_s = ref[:, :, sl]
    for est in ('H1', 'H2'):                         # Hv is single-input only
        H = frf.get_FRF(est)
        assert H.shape == ref.shape
        _assert_frf_close_where_finite(H[:, :, sl], ref_s)

    coh = np.abs(frf.get_coherence()[:, sl])
    finite = np.isfinite(coh)
    assert finite.mean() > 0.99
    np.testing.assert_allclose(coh[finite], 1.0, atol=1e-6)


def test_FRF_SIMO_pylump_output_noise():
    """H1 is insensitive to output (response) noise.

    With noise added only to the responses, the (Welch-averaged) H1 estimator must
    still recover the noise-free H1, and the coherence must stay within [0, 1]
    while reaching high values near the resonances.
    """
    pyLump = pytest.importorskip("pyLump")
    rng = np.random.default_rng(11)
    n_dof, fs, N = 4, 1500, 12000
    model = pyLump.Model(n_dof, mass=1.0, stiffness=1e4, damping=2.0, boundaries="both")

    exc_dof, resp_dof = [0], [0, 1, 2, 3]
    u = rng.standard_normal(N)
    resp, _ = model.get_response(exc_dof, u, fs, resp_dof=resp_dof,
                                 domain='f', return_matrix=True)
    noise = 0.02 * np.std(resp) * rng.standard_normal(resp.shape)

    nps = 2048
    common = dict(sampling_freq=fs, exc_type='f', resp_type='d', window='hann',
                  nperseg=nps, noverlap=nps // 2, fft_len=nps)
    H1_clean = pyFRF.FRF(exc=u[None, None, :], resp=resp[None, :, :], **common).get_FRF('H1')
    frf_noisy = pyFRF.FRF(exc=u[None, None, :], resp=(resp + noise)[None, :, :], **common)
    H1_noisy = frf_noisy.get_FRF('H1')

    sl = slice(1, -1)
    # 2 % output noise perturbs H1 by well under 5 %
    assert np.nanmax(np.abs(H1_noisy[:, :, sl] - H1_clean[:, :, sl])) \
        <= 0.05 * np.nanmax(np.abs(H1_clean[:, :, sl]))

    coh = np.abs(frf_noisy.get_coherence())
    assert np.nanmax(coh[:, sl]) > 0.9               # high coherence near resonances
    assert np.nanmax(coh[:, sl]) <= 1.0 + 1e-9       # bounded above by 1


# Run the tests
if __name__ == '__main__':
    np.testing.run_module_suite()