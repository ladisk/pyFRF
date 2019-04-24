"""
Unit test for pyFRF.py
"""
import pytest
import numpy as np
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import fft_tools

@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_derivation_integration():
    # create time data
    t, dt = np.linspace(0,1,1000, endpoint=False, retstep=True)
    fr = 100
    omega = 2*np.pi*fr
    displacement = np.sin(omega*t)
    velocity = omega*np.cos(omega*t)
    acceleration = -omega**2*np.sin(omega*t)

    # freq domain
    n = len(displacement)
    D = np.fft.rfft(displacement)*2/n
    V = np.fft.rfft(velocity)*2/n
    A = np.fft.rfft(acceleration)*2/n    
    w = 2*np.pi*np.fft.rfftfreq(n, d=dt)
    VfromD_freq = fft_tools.frequency_derivation(D, w)
    AfromD_freq = fft_tools.frequency_derivation(D, w, order=2)
    DfromV_freq = fft_tools.frequency_integration(V, w)
    DfromA_freq = fft_tools.frequency_integration(A, w, order=2)

    if False:
        print(V[100], VfromD_freq[100])
        print(A[100], AfromD_freq[100])
        print(D[100], DfromV_freq[100])
        print(D[100], DfromA_freq[100])
    np.testing.assert_allclose(V, VfromD_freq, atol=1e-8)
    np.testing.assert_allclose(A, AfromD_freq, atol=1e-7)
    np.testing.assert_allclose(V, VfromD_freq, atol=1e-8)
    np.testing.assert_allclose(A, AfromD_freq, atol=1e-7)

if __name__ == '__main__':
    test_derivation_integration()
    
if __name__ == '__mains__':
    np.testing.run_module_suite()