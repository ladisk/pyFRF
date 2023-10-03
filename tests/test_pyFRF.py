"""
Unit tests for pyFRF.py
"""

import numpy as np
import scipy
import pyExSi
import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pyFRF




# function used to acquire true systems FRF:
def get_true_FRF():
    ndof = 3

    # mass matrix:
    m = 2
    M = np.zeros((ndof, ndof))
    np.fill_diagonal(M, m)

    # stiffness matrix:
    k = 4000000
    K = np.zeros((ndof, ndof))

    for i in range(K.shape[0] - 1):
        K[i,i] = 2*k
        K[i+1,i] = -k
        K[i,i+1] = -k
    K[ndof-1,ndof-1] = k

    # damping matrix:
    c = 20
    C = np.zeros((ndof, ndof))

    for i in range(C.shape[0] - 1):
        C[i,i] = 2*c
        C[i+1,i] = -c
        C[i,i+1] = -c
    C[ndof-1,ndof-1] = c

    # eig_freq:
    eig_val, eig_vec = scipy.linalg.eigh(K, M)
    eig_val.sort()
    eig_omega = np.sqrt(np.abs(np.real(eig_val)))
    eig_freq = eig_omega / (2 * np.pi)

    # freq:
    freq = np.linspace(0.0, 1000, 4001)
    omega = 2 * np.pi * freq

    # true FRF:
    FRF = np.zeros([M.shape[0], M.shape[0], len(freq)], dtype="complex128")
    for i, omega_i in enumerate(omega):
        FRF[:,:,i] = scipy.linalg.inv(K - omega_i**2 * M + 1j*omega_i*C)

    # peak indexes
    peak_ind = scipy.signal.find_peaks(np.abs(FRF[0,0]))[0]

    # time:
    T = 1 / (freq[1] - freq[0])
    dt = 1 / (2*freq[-1])
    t = np.arange(0, T+dt, dt)

    return FRF, freq, t, peak_ind


# function used to acquire response via true FRF matrix and known excitation:
def get_response(exc, FRF_matrix, freq, exc_dofs, resp_dofs):
    """
    exc: excitation array
    FRF_matrix: true system FRF matrix
    exc_dofs: list of known excitation dof
    resp_dofs: list of wanted response dof
    """
    
    F = np.fft.rfft(exc)
    X = np.zeros((F.shape[0], FRF_matrix.shape[0], FRF_matrix.shape[2]), dtype="complex128")

    for i in range(F.shape[0]):
        for j in range(F.shape[2]):
            X[i,:,j] = FRF_matrix[:,exc_dofs,j] @ F[i,:,j]

    x = np.fft.irfft(X, n=2*(len(freq)-1)+1)
    x = x[:,resp_dofs]

    return x





# TEST FUNCTIONS:
def test_FRF_SISO():
    # get true FRF matrix, freq and time:
    (FRF_matrix, freq, t, true_peaks) = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:50] = 50 * np.sin(2*np.pi*np.arange(0,50,1)/100)
    x = get_response(f, FRF_matrix, freq, exc_dofs, resp_dofs)

    # get relevant FRFs from FRF matrix based on excitation dofs and response dofs:
    true_frf = np.zeros((len(resp_dofs), len(exc_dofs), FRF_matrix.shape[2]), dtype="complex128")
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            true_frf[i,j] = np.abs(FRF_matrix[resp_dofs[i],exc_dofs[j]])
    
    # get FRF from pyFRF:
    frf_pyFRF = np.abs(pyFRF.FRF(sampling_freq=int(1/t[1]), exc=f, resp=x, 
                                  window="none", exc_type='f', resp_type='d', 
                                  nperseg=None, noverlap=None, fft_len=None).get_H1())
    
    # test peaks:
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            frf_peaks = scipy.signal.find_peaks(frf_pyFRF[i,j], width=1.5, distance=200, height=6e-7)[0]
            np.testing.assert_allclose(frf_peaks, true_peaks, atol=20, rtol=0)

    # test frf values:
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            np.testing.assert_allclose(frf_pyFRF[i,j,1:], true_frf[i,j,1:], 
                                       rtol=1e-07, atol=1e-08)
            

def test_FRF_SIMO():
    # get true FRF matrix, freq and time:
    (FRF_matrix, freq, t, true_peaks) = get_true_FRF()

    # define excitation and get response (SIMO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0,1,2]  # multiple outputs
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:50] = 50 * np.sin(2*np.pi*np.arange(0,50,1)/100)
    x = get_response(f, FRF_matrix, freq, exc_dofs, resp_dofs)

    # get relevant FRFs from FRF matrix based on excitation dofs and response dofs:
    true_frf = np.zeros((len(resp_dofs), len(exc_dofs), FRF_matrix.shape[2]), dtype="complex128")
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            true_frf[i,j] = np.abs(FRF_matrix[resp_dofs[i],exc_dofs[j]])
    
    # get FRF from pyFRF:
    frf_pyFRF = np.abs(pyFRF.FRF(sampling_freq=int(1/t[1]), exc=f, resp=x, 
                                  window="none", exc_type='f', resp_type='d', 
                                  nperseg=None, noverlap=None, fft_len=None).get_H1())
    
    # test peaks:
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            frf_peaks = scipy.signal.find_peaks(frf_pyFRF[i,j], width=1.5, distance=200, height=6e-7)[0]
            np.testing.assert_allclose(frf_peaks, true_peaks, atol=20, rtol=0)
            
    # test frf values:
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            np.testing.assert_allclose(frf_pyFRF[i,j,1:], true_frf[i,j,1:], 
                                       rtol=1e-07, atol=1e-08)
            

def test_FRF_MISO():
    # get true FRF matrix, freq and time:
    (FRF_matrix, freq, t, true_peaks) = get_true_FRF()

    # define excitation and get response (MISO):
    n_measurements = 5
    exc_dofs = [0,1,2]  # multiple inputs
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            f[i][j] = 100 * pyExSi.normal_random(N=t.shape[0])
    x = get_response(f, FRF_matrix, freq, exc_dofs, resp_dofs)

    # get relevant FRFs from FRF matrix based on excitation dofs and response dofs:
    true_frf = np.zeros((len(resp_dofs), len(exc_dofs), FRF_matrix.shape[2]), dtype="complex128")
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            true_frf[i,j] = np.abs(FRF_matrix[resp_dofs[i],exc_dofs[j]])
    
    # get FRF from pyFRF:
    frf_pyFRF = np.abs(pyFRF.FRF(sampling_freq=int(1/t[1]), exc=f, resp=x, 
                                  window="hann", exc_type='f', resp_type='d', 
                                  nperseg=4000, noverlap=None, fft_len=None).get_H1())

    # test peaks:
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            frf_peaks = scipy.signal.find_peaks(frf_pyFRF[i,j], width=1.5, distance=200, height=6e-7)[0]
            np.testing.assert_allclose(frf_peaks, true_peaks, atol=20, rtol=0)

    # test frf values:
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            np.testing.assert_allclose(frf_pyFRF[i,j,1:], true_frf[i,j,1:], 
                                       rtol=3, atol=1e-08)
            

def test_FRF_MIMO():
    # get true FRF matrix, freq and time:
    (FRF_matrix, freq, t, true_peaks) = get_true_FRF()

    # define excitation and get response (MIMO):
    n_measurements = 5
    exc_dofs = [0,1,2]  # multiple inputs
    resp_dofs = [0,1,2]  # multiple outputs
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            f[i][j] = 100 * pyExSi.normal_random(N=t.shape[0])
    x = get_response(f, FRF_matrix, freq, exc_dofs, resp_dofs)

    # get relevant FRFs from FRF matrix based on excitation dofs and response dofs:
    true_frf = np.zeros((len(resp_dofs), len(exc_dofs), FRF_matrix.shape[2]), dtype="complex128")
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            true_frf[i,j] = np.abs(FRF_matrix[resp_dofs[i],exc_dofs[j]])
    
    # get FRF from pyFRF:
    frf_pyFRF = np.abs(pyFRF.FRF(sampling_freq=int(1/t[1]), exc=f, resp=x, 
                                  window="hann", exc_type='f', resp_type='d', 
                                  nperseg=4000, noverlap=None, fft_len=None).get_H1())

    # test peaks:
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            frf_peaks = scipy.signal.find_peaks(frf_pyFRF[i,j,20:], width=2, distance=200, height=6e-7)[0]
            np.testing.assert_allclose(frf_peaks+20, true_peaks, atol=20, rtol=0)

    # test frf values:
    for i in range(len(resp_dofs)):
        for j in range(len(exc_dofs)):
            np.testing.assert_allclose(frf_pyFRF[i,j,1:], true_frf[i,j,1:], 
                                       rtol=3, atol=1e-08)
            

def test_freq():
    # get true FRF matrix, freq and time:
    (FRF_matrix, freq, t, peaks) = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:50] = 50 * np.sin(2*np.pi*np.arange(0,50,1)/100)
    x = get_response(f, FRF_matrix, freq, exc_dofs, resp_dofs)

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
    (FRF_matrix, freq, t, peaks) = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:50] = 50 * np.sin(2*np.pi*np.arange(0,50,1)/100)
    x = get_response(f, FRF_matrix, freq, exc_dofs, resp_dofs)

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
    (FRF_matrix, freq, t, peaks) = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:50] = 50 * np.sin(2*np.pi*np.arange(0,50,1)/100)
    x = get_response(f, FRF_matrix, freq, exc_dofs, resp_dofs)

    # define sampling frequency and length of fft:
    sampling_freq = int(1/t[1])
    fft_len=8000

    # create a test object:
    test_object = pyFRF.FRF(sampling_freq=sampling_freq, exc=f, resp=x, 
                             window="none", exc_type='f', resp_type='d', 
                             nperseg=None, noverlap=None, fft_len=fft_len)
    
    # test:
    np.testing.assert_allclose(np.arange(0, t[-1], 1/sampling_freq), 
                                   test_object.get_t_axis())
    

def test_df():
    # get true FRF matrix, freq and time:
    (FRF_matrix, freq, t, peaks) = get_true_FRF()

    # define excitation and get response (SISO):
    n_measurements = 1
    exc_dofs = [0]  # single input
    resp_dofs = [0]  # single output
    f = np.zeros((n_measurements, len(exc_dofs), t.shape[0]))
    f[0,0,0:50] = 50 * np.sin(2*np.pi*np.arange(0,50,1)/100)
    x = get_response(f, FRF_matrix, freq, exc_dofs, resp_dofs)

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
    test_object = pyFRF.FRF(sampling_freq=1000, fft_len=500)

    # 2x2 matrix inverse:
    A = np.array([[1, 2], 
                  [3, 4]])
    # test
    np.testing.assert_allclose(test_object._analytical_matrix_inverse(A), 
                                   np.linalg.inv(A))
    
    # 3x3 matrix inverse:
    A = np.array([[1, 2, 3], 
                  [2, 1, 3], 
                  [3, 2, 1]])
    # test:
    np.testing.assert_allclose(test_object._analytical_matrix_inverse(A), 
                                   np.linalg.inv(A))





if __name__ == '__mains__':
    np.testing.run_module_suite()