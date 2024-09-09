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

if __name__ == '__main__':
    np.testing.run_module_suite()