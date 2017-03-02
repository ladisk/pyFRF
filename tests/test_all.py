"""
Unit test for pyFRF.py
"""
import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from pyFRF import FRF

def test_synthetic():
    C = 0.5 + 0.1j  # modal constant
    eta = 5e-3  # damping loss factor
    f0 = 320  # natural frequency
    df = 1  # freq resolution
    D = 1e-8 * (1 - .1j)  # residual

    f = 1 * np.arange(0, 1400, step=df)  # / frequency range

    w0 = f0 * 2 * np.pi  # to rad/s
    w = f * 2 * np.pi
    H1_syn = C / (w0 ** 2 - w ** 2 + 1.j * eta * w0 ** 2) + \
             +0.5 * np.conj(C) / ((w0 * 2) ** 2 - w ** 2 + 1.j * eta * (w0 * 2) ** 2) \
             + 0.25 * C / ((w0 * 3) ** 2 - w ** 2 + 1.j * eta * (w0 * 3) ** 2) \
             + D

    h = np.fft.irfft(H1_syn)
    l = len(H1_syn) * 2 - 2
    t = np.arange(l) / (l - 1)
    exc = np.zeros_like(t)
    exc[0] = 1

    frf = FRF(sampling_freq=1 / t[1], exc=exc, resp=h, exc_window='None', resp_type='d', resp_window='None')
    a = np.abs(frf.get_FRF())
    b = np.abs(H1_syn)

    np.testing.assert_allclose(a[1:],b[1:], atol=1e-8)

if __name__ == '__mains__':
    np.testing.run_module_suite()