import numpy as np

def mdof_K_C(p):
    """
    Construct the stiffness or damping matrix of a N-DOF lumped parameter
    mass-spring-damper system from given parameter values `p`.
    """

    N = len(p) - 1
    mat = np.zeros((N, N))
    for i in range(N):
        if i == 0:
            mat[i, :2] = np.array([ p[0] + p[1], -p[1] ]) 
        elif i == N-1:
            mat[i, -2:] = np.array([ -p[-2], p[-2] + p[-1] ])
        else:
            mat[i, i-1:i+2] = np.array([ -p[i], p[i]+p[i+1], -p[i+1] ])
    return mat.astype(np.float64)


def modal_model(m, k, c, get_matrices=False):
    """
    Compute the modal model (natural frequencies, damping ratios, system 
    roots and eigenvectors) of a N-DOF system with provided properties.
    """

    # System matrices
    N = len(m)
    M = np.diag(m).astype(np.float64)
    C = mdof_K_C(c)
    K = mdof_K_C(k)
    
    # State-space model
    A = np.zeros(2*np.array(M.shape))
    A[:N, :N] = C
    A[:N, -N:] = M
    A[-N:, :N] = M 
    B = np.zeros_like(A)
    B[:N, :N] = K
    B[-N:, -N:] = -M

    # Save system matrices
    matrices = (M, K, C, A, B)

    # Modal analysis
    AB_eig = np.linalg.inv(A) @ B
    w, v = np.linalg.eig(AB_eig)

    roots = -w[1::2][::-1]
    roots_c = -w[::2][::-1]
    _vectors = v[:N, ::-2]     # non-normalized
    _vectors_c = v[:N, -2::-2] # non-normalized

    # eigenvector matrix for normalization
    PHI = np.zeros_like(v)
    PHI[:N, :N] = _vectors
    PHI[-N:, :N] = roots*_vectors
    PHI[:N, -N:] = _vectors_c
    PHI[-N:, -N:] = roots_c*_vectors_c
    a_r = np.diagonal(PHI.T @ A @ PHI)
    _a_r = a_r[:N]
    _a_r_c = a_r[N:]

    # A-normalization
    vectors_a = _vectors / np.sqrt(_a_r) # A-normalized
    vectors_a_c = _vectors_c / np.sqrt(_a_r_c) # A-normalized

    # Order returned data by system roots amplitude
    order = np.argsort(np.abs(roots))
    eig = (roots[order], roots_c[order], vectors_a[:, order], vectors_a_c[:, order]) # A-normalization

    # System natural frequencies and viscous damping ratios
    w_r = np.abs(roots[order])
    d = -np.real(roots[order]) / w_r
    f = w_r / 2 / np.pi # [Hz]

    if get_matrices:
        return f, d, eig, matrices
    else:
        return f, d, eig


def FRF_matrix(omega, roots, roots_c, vectors, vectors_c):
    """
    Compute the receptance matrix of a system with provided roots and 
    eigenvectors at selected frequencies `omega`.
    """
    N = len(roots)
    n = len(omega)
    FRF = np.zeros((N, N, n), dtype=np.complex128)

    for j in range(N): # response location
        for k in range(N): # excitation location 
            FRF_jk = (vectors[j]*vectors[k])[:, None] / (1j*omega[None, :] - roots[:, None])
            FRF_jk += (vectors_c[j]*vectors_c[k])[:, None] / (1j*omega[None, :] - roots_c[:, None])
            FRF[j, k] = np.sum(FRF_jk, axis=0)

    return FRF


def compute_response(FRF, excitation, excitation_locations=None):
    """
    Compute the time-reposnse of the N-DOF lumped-mass system with FRF matrix
    `FRF` for the input `excitation` force vectors.
    
    :param FRF: array of shape (N, N, n), the receptance matrix of the
        N-DOF lumped parameter system.
    :param excitaion: array of shape (M, P, N_samples), excitation (force) time
        signals for the `M` measurement repetitions `P` input locations.
    :param excitation_locations: array of shape (P,) indices of the locations 
        (DOFs) that will be excited. If None, it is assumed that all N DOFs 
        are to be excited, and the number of excitation vectors `P` must match 
        the number of DOFs `N`. Defaults to None.
    :returns: array of shape (N, N_samples), the time-response of the system
        at the `N` DOFs.
    """
    if excitation_locations is None and excitation.shape[1] == FRF.shape[1]:
        FRF = FRF
    elif len(excitation_locations) == excitation.shape[1] <= FRF.shape[1]:
        FRF = FRF[:, excitation_locations, :]
    else:
        raise ValueError('FRF degrees-of-freedom and excitation_locations mismatch.')
        
    N_samples = excitation.shape[2]
    N = FRF.shape[0]
    M = excitation.shape[0]
    
    # Frequency-domain superposition
    F_f = np.fft.rfft(excitation) * 2 / excitation.shape[-1] # excitation spectra
    
    # Compute response spectra at each frequency for each meausrement
    X_f = np.zeros((M, N, FRF.shape[-1]), dtype=np.complex128)
    for m in range(M):
        for i in range(FRF.shape[-1]):
            X_f[m, :, i] = np.dot(FRF[:, :, i], F_f[m, :, i])
    
    # Time-domain response with added random noise
    X_t = (np.fft.irfft(X_f) * (X_f.shape[-1]-1))[:, :, :N_samples]
    
    return X_t