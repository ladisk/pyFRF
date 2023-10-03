"""

Module for FRF signal processing.

Classes:
    class FRF:      Frequency response function.
"""

import numpy as np
import scipy
import warnings

__version__ = '1.0'
_EXC_TYPES = ['f', 'a', 'v', 'd', 'e']  # force for EMA and kinematics for OMA
_RESP_TYPES = ['a', 'v', 'd', 'e']  # acceleration, velocity, displacement, strain
_FRF_TYPES = ['H1', 'H2', 'Hv', 'ODS']
_FRF_FORM = ['accelerance', 'mobility', 'receptance']
_WGH_TYPES = ['linear', 'exponential']
_WINDOWS = ['none', 'force', 'exponential', 'hann', 'hamming', 'bartlett', 'blackman', 'kaiser']

_DIRECTIONS = ['scalar', '+x', '+y', '+z', '-x', '-y', '-z']
_DIRECTIONS_NR = [0, 1, 2, 3, -1, -2 - 3]



def direction_dict():
    dir_dict = {a: b for a, b in zip(_DIRECTIONS, _DIRECTIONS_NR)}
    return dir_dict

class FRF:
    """
    Perform Dual Channel Spectral Analysis

    """
    
    def __init__(self, sampling_freq,
                 exc=None,
                 resp=None,
                 exc_type='f', resp_type='a',
                 window='none',
                 resp_delay=0.,
                 weighting='linear',
                 fft_len=None,
                 nperseg=None,
                 noverlap=None,
                 archive_time_data=False,
                 frf_type='H1',
                 copy=True,
                 **kwargs):
        """
        Initiates the Data class:

        :param sampling_freq: Sampling frequency of excitation and response signals.
            int - same sampling frequency for both excitation and response signals.
            tuple, list - different sampling frequencies for excitation and reponse signals. 
            First element represents excitation, second element represents response sampling frequency.
        :type sampling_freq: int, tuple(int), list[int]
        :param exc: Excitation array.
            A 3D, 2D or 1D ndarray:
            3D (general shape, allows multiple inputs (MIMO, MISO)): `(number_of_measurements, excitation_DOFs, time_series)`.
            2D (multiple measurements, single input): `(number_of_measurements, time_series)`.
            1D (single measurement, single input): `(time_series)`.
        :type exc: ndarray
        :param resp: Response array.
            A 3D, 2D or 1D ndarray:
            3D (general shape, allows, multiple outputs): `(number_of_measurements, response_DOFs, time_series)`.
            2D (multiple measurements, single output): `(number_of_measurements, time_series)`.
            1D (single measurement, single output): `(time_series)`.
        :type resp: ndarray
        :param exc_type: Excitation type, see _EXC_TYPES.
        :type exc_type: str
        :param resp_type: Response type, see _RESP_TYPES.
        :type resp_type: str
        :param window: Scipy window used for cross power spectral desnity computation or 
            excitation and reponse signals. see _WINDOWS.
            str - used for csd. 
            tuple, list - first element is excitation signal window, second element is response signal window
        :type window: str, tuple(str), list[str]
        :param resp_delay: Response time delay (in seconds) with regards to the excitation.
        :type resp_delay: float
        :param weighting: weighting type used for averaging with continous measurements - see _WGH_TYPES.
            If exponential weighting is used, specify the number of averages - example: 'exponential:5'
        :type weighting: str
        :param fft_len: The length of the FFT (zero-padding if longer than length of data),
            If None, freq length matches time length.
        :type fft_len: int
        :param nperseg: Optional segment length, by default one segment (data length) is analyzed.
        :type nperseg: int
        :param noverlap: Optional segment overlap, by default ``noverlap = nperseg // 2``.
        :type noverlap: int
        :param archive_time_data: Archive the time data (this can consume a lot of memory).
        :type archive_time_data: bool
        :param frf_type: Default frf type returned at self.get_frf(), see _FRF_TYPES.
        :type frf_type: str
        :param copy: If true the excitation and response arrays are copied 
            (if data is not copied the applied window affects the source arrays).
        :type copy: bool
        """
        # previous pyFRF kwargs:
        if ("exc_window" in kwargs) or ("resp_window" in kwargs):
            raise ValueError("The exc_window argument is no longer supported, use window argumet instead an see documentation.")
        if ("n_averages" in kwargs):
            raise ValueError("The n_averages argument is no longer supported. You can pass different array shapes of excitation "\
                             "and response data or provide N for exponential averaging through weighting argument.")


        # sampling_freq info:
        if isinstance(sampling_freq, int):
            self.sampling_freq = sampling_freq
            self.exc_sampling_freq = sampling_freq
            self.resp_sampling_freq = sampling_freq
        elif isinstance(sampling_freq, tuple) or isinstance(sampling_freq, list):
            if len(sampling_freq) != 2:
                raise Exception('''If parameter sampling_freq is passed as tuple or list, the length of it must be 2: 
                                first element is excitation sampling frequency, second element is response sampling frequency''')
            if isinstance(sampling_freq[0], int) and isinstance(sampling_freq[1], int):
                self.exc_sampling_freq = sampling_freq[0]
                self.resp_sampling_freq = sampling_freq[1]
                # take the lower sampling rate, the signal with higher sampling rate will be downsampled to lower
                if sampling_freq[0] >= sampling_freq[1]:
                    self.sampling_freq = sampling_freq[1]
                else:
                    self.sampling_freq = sampling_freq[0]
            else:
                raise Exception('Both elements of tuple or list must be int type!')
        else:
            raise Exception('Invalid argument type for parameter sampling_freq! int, tuple and list supported')

        # data info:  
        self._data_available = False
        self.exc_type = exc_type
        self.resp_type = resp_type
        self.resp_delay = resp_delay
        
        if isinstance(window, str):
            self.csd_window = window.lower()
            self.exc_window = 'none'
            self.resp_window = 'none'
        elif isinstance(window, tuple) or isinstance(window, list):
            if len(window) != 2:
                raise Exception('''If parameter window is passed as tuple or list, the length of it must be 2: 
                                first element is excitation window, second element is response window''')
            if isinstance(window[0], str) and isinstance(window[1], str):
                self.csd_window = 'none'
                self.exc_window = window[0].lower()
                self.resp_window = window[1].lower()
            else:
                raise Exception('Both elements of tuple or list must be string type!')
        else:
            raise Exception('Invalid argument type for parameter window! string, tuple and list supported')


        if self.resp_window.split(':')[0] == 'force':
            #print("force window used on response signal")
            warnings.warn("force window used on response signal")
        if self.csd_window.split(':')[0] == 'force':
            #print("force window used with csd")
            warnings.warn("force window used with csd")
        
        self.frf_type = frf_type
        self.copy = copy

        # ini
        self.exc = np.array([])
        self.resp = np.array([])
        self.exc_archive = []
        self.resp_archive = []
        self.samples = None

        # set averaging and weighting
        self.weighting = weighting
        self.frf_norm = 1.

        if exc is not None:
            if len(exc.shape) == 1:
                exc = np.expand_dims(exc, (0,1))
            if len(exc.shape) == 2:
                exc = np.expand_dims(exc, 1)
        if resp is not None:
            if len(resp.shape) == 1:
                resp = np.expand_dims(resp, (0,1))
            if len(resp.shape) == 2:
                resp = np.expand_dims(resp, 1)

        # reshape if signals are added at the beginning (at object initialization) and if different sampling_freqs are specified:
        # both signals have to last the same amount of time
        if exc is not None and resp is not None:
            if (self.exc_sampling_freq != self.resp_sampling_freq) and (exc.shape[2] != resp.shape[2]):
                if self.exc_sampling_freq > self.resp_sampling_freq:
                    exc = scipy.signal.resample(exc, resp.shape[2], axis=2)  # downsample excitation signal
                elif self.resp_sampling_freq > self.exc_sampling_freq:
                    resp = scipy.signal.resample(resp, exc.shape[2], axis=2)  # downsample response signal
        # fft length
        if fft_len is None:
            if exc is None:
                raise Exception('If no excitaton signal is given, fft_len needs to be defined!')
            self.fft_len = exc.shape[2]
        else:
            self.fft_len =  int(fft_len)
        self.nperseg = nperseg
        self.noverlap = noverlap

        # save time data
        self.archive_time_data = archive_time_data
        
         # error checking
        if not (self.frf_type in _FRF_TYPES):
            raise Exception('wrong FRF type given %s (can be %s)'
                            % (self.frf_type, _FRF_TYPES))
        
        if not (self.weighting.split(':')[0] in _WGH_TYPES):
            raise Exception('wrong weighting type given %s (can be %s)'
                            % (self.weighting, _WGH_TYPES))

        if not (self.exc_type in _EXC_TYPES):
            raise Exception('wrong excitation type given %s (can be %s)'
                            % (self.exc_type, _EXC_TYPES))

        if not (self.resp_type in _RESP_TYPES):
            raise Exception('wrong response type given %s (can be %s)'
                            % (self.resp_type, _RESP_TYPES))
        
        if not (self.csd_window.split(':')[0] in _WINDOWS):
            raise Exception('wrong csd window type given %s (can be %s)'
                            % (self.exc_window, _WINDOWS))

        if not (self.exc_window.split(':')[0] in _WINDOWS):
            raise Exception('wrong excitation window type given %s (can be %s)'
                            % (self.exc_window, _WINDOWS))

        if not (self.resp_window.split(':')[0] in _WINDOWS):
            raise Exception('wrong response window type given %s (can be %s)'
                            % (self.resp_window, _WINDOWS))

        self.total_meas = int(0)  # total number of all measurements
        self.ntimes_meas_added = int(0)  # the number of times we added new measuremenets

        if exc is not None and resp is not None:
            self.add_data(exc, resp)
            
            
    def add_data(self, exc, resp):
        """
        Adds new data - called at object creation if excitation and response signals are given.
        Used also for continous data adding.

        :param exc: Excitation array.
        :param resp: Response array.
        :return: True if data is added.
        """
        if len(exc.shape) == 1:
            exc = np.expand_dims(exc, (0,1))
        if len(exc.shape) == 2:
            exc = np.expand_dims(exc, 1)

        if len(resp.shape) == 1:
            resp = np.expand_dims(resp, (0,1))
        if len(resp.shape) == 2:
            resp = np.expand_dims(resp, 1)

        # reshape if signals are added later and if different sampling_freqs are specified at the beginning:
        if (self.exc_sampling_freq != self.resp_sampling_freq) and (exc.shape[2] != resp.shape[2]):
            if self.exc_sampling_freq > self.resp_sampling_freq:
                exc = scipy.signal.resample(exc, resp.shape[2], axis=2)  # downsample excitation signal
            elif self.resp_sampling_freq > self.exc_sampling_freq:
                resp = scipy.signal.resample(resp, exc.shape[2], axis=2)  # downsample response signal

        self._add_to_archive(exc, resp)

        if self.copy:
            self.exc = np.copy(exc)
            self.resp = np.copy(resp)
        else:
            self.exc = exc
            self.resp = resp
            
        samples = exc.shape[2]
        self.fft_len_cutoff = exc.shape[2]
        if self.nperseg is None:
            self.nperseg = samples
            if self.nperseg > self.fft_len:  # for less fft points and no averaging in csd
                self.fft_len_cutoff = self.fft_len
                self.nperseg = self.fft_len
        elif self.nperseg > samples:
            raise ValueError('nperseg must be less than samples.')
        if self.nperseg > self.fft_len:
            raise ValueError('fft_len must be greater than or equal to nperseg when specifying both.')
        if self.noverlap is None:
            self.noverlap = self.nperseg // 2
        elif self.noverlap >= self.nperseg:
            raise ValueError('noverlap must be less than nperseg.')

        self._ini_lengths_and_csd_window(self.nperseg)
        self._ini_exc_and_resp_window()
        self._apply_exc_and_resp_window()

        self.total_meas += exc.shape[0]
        self.ntimes_meas_added += 1
        #print("number of new measurements added: ", exc.shape[0])
        #print("current measurement in row (total measurements): ", self.total_meas)
        #print("number of times we added new measuremets: ", self.ntimes_meas_added)
        
        self._get_frf_av()
        self._data_available = True
        
        # accelerance/mobility/receptance conversion frf multiplication factor:
        if self.resp_type == 'a':
            self.frf_conversion = np.power(-1.j / self.get_w_axis(), 2)
        elif self.resp_type == 'v':
            self.frf_conversion = np.power(-1.j / self.get_w_axis(), 1)
        else:
            self.frf_conversion = np.ones(self.get_w_axis().shape)
        
        return True
    
    def is_data_ok(self, exc, resp, 
                 overflow_samples=3, 
                 double_impact_limit=1e-3, verbose=0):
        """
        Checks the data for overflow and double-impacts.

        :param exc: Excitation array.
        :param resp: Response array.
        :param overflow_samples: Number of samples that need to be equal to max for overflow identification.
        :type overflow_samples: int
        :param double_impact_limit: Ratio of freqency content of the double vs single hit.
            Smaller number means more sensitivity
        :type double_impact_limit: float
        :param verbose: Prints overflow and double impact status. 0 or 1.
        :type verbose: int
        :return: True if data is ok, False if there is overflow or double impact present.
        :rtype: bool
        """
        if exc.ndim != resp.ndim:
            raise Exception("excitation and response array should be the same number of dimensions!")
        
        overflow_exc = self._is_overflow(exc, overflow_samples=overflow_samples, verbose=verbose)
        overflow_resp = self._is_overflow(resp, overflow_samples=overflow_samples, verbose=verbose)
        double_impact = self._is_double_impact(exc, limit=double_impact_limit, verbose=verbose)

        if exc.ndim == 3:
            for measurement_num in range(exc.shape[0]):
                for dof_num in range(exc.shape[1]):
                    if overflow_exc[measurement_num, dof_num] or \
                        overflow_resp[measurement_num, dof_num] or \
                        double_impact[measurement_num, dof_num]:
                        return False
            return True
        elif exc.ndim == 2:
            for measurement_num in range(exc.shape[0]):
                if overflow_exc[measurement_num] or \
                    overflow_resp[measurement_num] or \
                    double_impact[measurement_num]:
                    return False
            return True
        elif exc.ndim == 1:
            if overflow_exc or overflow_resp or double_impact:
                return False
            else:
                return True
        else:
            raise Exception('Incorrect input array shape!')

    
    def _apply_exc_and_resp_window(self):
        """
        Apply windows to excitation and response signals.

        :return:
        """
        
        self.exc *= self.exc_window_data
        self.resp *= self.resp_window_data
    
    def _get_window_sub(self, length, window='none'):
        """
        Returns the window time series and amplitude normalization term.

        :param length: Length of window.
        :type length: int
        :param window: Window type.
        :type window: str
        :return: Window ndarray, amplitude norm
        """
        #print("window: ", window)
        window = window.split(':')
        
        if window[0] in ['hanning', 'hann']:
            w = np.hanning(length)
        elif window[0] in ['hamming']:
            w = np.hamming(length)
        elif window[0] in ['bartlett']:
            w = np.bartlett(length)
        elif window[0] in ['blackman']:
            w = np.blackman(length)
        elif window[0] in ['kaiser']:
            beta = float(window[1])
            w = np.kaiser(length, beta)
        elif window[0] == 'force':
            w = np.zeros(length)
            force_window = float(window[1])
            to1 = int(force_window * length)
            w[:to1] = 1.
        elif window[0] == 'exponential':
            w = np.arange(length)
            exponential_window = float(window[1])
            w = np.exp(np.log(exponential_window) * w / (length - 1))
        elif window[0] in ['none']:
            w = np.ones(length)

        if window[0] == 'force':
            amplitude_norm = 2 / len(w)
        else:
            amplitude_norm = 2 / np.sum(w)

        return w, amplitude_norm

            
    # OK:
    def get_f_axis(self):
        """
        Get frequency series.

        :return: Frequency ndarray in Hz.
        :rtype: ndarray
        """
        if not self._data_available:
            raise Exception('No data has been added yet!')

        return np.fft.rfftfreq(self.fft_len, 1. / self.sampling_freq)

    # OK:
    def get_w_axis(self):
        """
        Get angular freqency series.

        :return: Frequency ndarray in rad/s.
        :rtype: ndarray
        """
        if not self._data_available:
            raise Exception('No data has been added yet!')

        return 2 * np.pi * self.get_f_axis()
    
    # OK:
    def get_df(self):
        """
        Get delta frequency.

        :return: Delta frequency in Hz.
        :rtype: float
        """
        if not self._data_available:
            raise Exception('No data has been added yet!')

        return self.get_f_axis()[1]

    # OK:
    def get_t_axis(self):
        """
        Get time axis.

        :return: Time axis ndarray.
        :rtype: ndarray
        """

        if not self._data_available:
            raise Exception('No data has been added yet!')

        return np.arange(self.samples) / self.sampling_freq
    
    
    def _get_frf_av(self):
        """
        Calculates the averaged FRF based on averaging and weighting type, using scipy.csd.


        Literature:
            [1] Haylen, Lammens, Sas: ISMA 2011 Modal Analysis Theory and Testing page: A.2.27
            [2] http://zone.ni.com/reference/en-XX/help/371361E-01/lvanlsconcepts/average_improve_measure_freq/

        :return:
        """
        # obtain cross and auto spectra for current data
        freq_len = np.fft.rfftfreq(self.fft_len, 1. / self.sampling_freq).shape[0]
        S_XX = np.zeros((self.resp.shape[1], self.resp.shape[1], freq_len), dtype="complex128")
        S_FF = np.zeros((self.exc.shape[1], self.exc.shape[1], freq_len), dtype="complex128")
        S_XF = np.zeros((self.resp.shape[1], self.exc.shape[1], freq_len), dtype="complex128")
        S_FX = np.zeros((self.exc.shape[1], self.resp.shape[1], freq_len), dtype="complex128")

        #print("newly added measurements (csd calculating): ", self.resp.shape[0])
        
        #print("nperseg: ", self.nperseg)
        #print("samples: ", self.samples)
        #print("noverlap: ", self.noverlap)
        #print("fft_len: ", self.fft_len)

        for k in range(self.resp.shape[0]): # for each measurement
            for i in range(S_XX.shape[0]):
                for j in range(S_XX.shape[1]):
                    S_XX[i,j] += scipy.signal.csd(self.resp[k][i][:self.fft_len_cutoff], self.resp[k][j][:self.fft_len_cutoff], 
                                                  self.sampling_freq, window=self.csd_window_data, nperseg=self.nperseg, 
                                                  noverlap=self.noverlap, nfft=self.fft_len)[1]
            for i in range(S_FF.shape[0]):
                for j in range(S_FF.shape[1]):
                    S_FF[i,j] += scipy.signal.csd(self.exc[k][i][:self.fft_len_cutoff], self.exc[k][j][:self.fft_len_cutoff], 
                                                  self.sampling_freq, window=self.csd_window_data, nperseg=self.nperseg, 
                                                  noverlap=self.noverlap, nfft=self.fft_len)[1]
            for i in range(S_XF.shape[0]):
                for j in range(S_XF.shape[1]):
                    S_XF[i,j] += scipy.signal.csd(self.resp[k][i][:self.fft_len_cutoff], self.exc[k][j][:self.fft_len_cutoff], 
                                                  self.sampling_freq, window=self.csd_window_data, nperseg=self.nperseg, 
                                                  noverlap=self.noverlap, nfft=self.fft_len)[1]
            for i in range(S_FX.shape[0]):
                for j in range(S_FX.shape[1]):
                    S_FX[i,j] += scipy.signal.csd(self.exc[k][i][:self.fft_len_cutoff], self.resp[k][j][:self.fft_len_cutoff], 
                                                  self.sampling_freq, window=self.csd_window_data, nperseg=self.nperseg, 
                                                  noverlap=self.noverlap, nfft=self.fft_len)[1]

        if self.ntimes_meas_added == 1:  # if the measurements are added for the first time:
            #print(self.ntimes_meas_added, " time the measurements were added - csd")                        
            self.S_XX = S_XX / self.resp.shape[0]
            self.S_FF = S_FF / self.resp.shape[0]
            self.S_XF = S_XF / self.resp.shape[0]
            self.S_FX = S_FX / self.resp.shape[0]
        else:  # proportionaly add to average, based on total number of measurements and number of newly added measurements
            num_new_meas = self.resp.shape[0]  # number of newly added measurements
            #print(self.ntimes_meas_added, " time the measurements were added - csd") 
            weighting = self.weighting.split(':')
            if weighting[0] == 'linear':
                #print("linear")
                N = np.float64(self.total_meas)
            elif weighting[0] == 'exponential':
                #print("exponential")
                if len(weighting) == 1:  # not specified N of averages for exponential averaging
                    N_exp = np.float64(self.total_meas)  # same as linear
                elif weighting[1] == '':  # not specified N of averages for exponential averaging
                    N_exp = np.float64(self.total_meas)  # same as linear
                else:
                    N_exp = int(weighting[1])
                #print(N_exp)
                if self.total_meas >= N_exp:
                    N = np.float64(N_exp)
                else:
                    N = np.float64(self.total_meas)
            else:
                raise Exception("Incorrect weighting type!")
            
            #print("N: ", N)

            self.S_XX = num_new_meas / N * S_XX + (N - num_new_meas) / N * self.S_XX
            self.S_FF = num_new_meas / N * S_FF + (N - num_new_meas) / N * self.S_FF
            self.S_XF = num_new_meas / N * S_XF + (N - num_new_meas) / N * self.S_XF
            self.S_FX = num_new_meas / N * S_FX + (N - num_new_meas) / N * self.S_FX


    def get_ods_frf(self):
        """
        Operational deflection shape averaged estimator.

        Numerical implementation of Equation (6) in [1].

        Literature:
            [1] Schwarz, Brian, and Mark Richardson. Measurements required for displaying
                operating deflection shapes. Presented at IMAC XXII January 26 (2004): 29.

        :return: ODS FRF estimator.
        :rtype: ndarray
        """

        freq_len = np.fft.rfftfreq(self.fft_len, 1. / self.sampling_freq).shape[0]
        ods_frf = np.zeros((self.resp.shape[1], self.exc.shape[1], freq_len), dtype="complex128")
        
        if self.exc.shape[1] != 1:  # MISO, MIMO
            raise Exception("Currently not implemented for MISO and MIMO systems.")
        else:  # SISO, SIMO
            for i in range(self.resp.shape[1]):
                ods_frf[i,:,:] = np.sqrt(self.S_XX[i,i,:]) * self.S_XF[i,:,:] / np.abs(self.S_XF[i,:,:])
        return ods_frf * self.frf_conversion
    

    def get_resp_spectrum(self, amplitude_spectrum=True):
        """
        Get response amplitude/power spectrum.

        :param amplitude_spectrum: True - get amplitude spectrum, False - power spectrum.
        :return: Response spectrum.
        :rtype: ndarray
        """
        
        freq_len = np.fft.rfftfreq(self.fft_len, 1. / self.sampling_freq).shape[0]
        amp = np.zeros((self.resp.shape[1], freq_len), dtype="complex128")
        
        if self.exc.shape[1] != 1:  # MISO, MIMO
            raise Exception("Currently not implemented for MIMO systems.")
        else:  # SISO, SIMO    
            k = self.resp_window_amp_norm
            
            for i in range(self.resp.shape[1]):
                amp[i,:] = np.sqrt(np.abs(self.S_XX[i,i,:]))

            if amplitude_spectrum:
                return k * amp * self.frf_conversion
            else:
                return k * (amp * self.frf_conversion) ** 2
    

    def get_exc_spectrum(self, amplitude_spectrum=True):
        """
        Get excitation amplitude/power spectrum.

        :param amplitude_spectrum: True - get amplitude spectrum, False - power spectrum.
        :return: Excitation spectrum.
        :rtype: ndarray
        """
        
        freq_len = np.fft.rfftfreq(self.fft_len, 1. / self.sampling_freq).shape[0]
        amp = np.zeros((self.exc.shape[1], freq_len), dtype="complex128")
        
        if self.exc.shape[1] != 1:  # MISO, MIMO
            raise Exception("Currently not implemented for MIMO systems.")
        else:  # SISO, SIMO    
            k = self.exc_window_amp_norm

            amp[0,:] = np.sqrt(np.abs(self.S_FF[0,0,:]))

            if amplitude_spectrum:
                return k * amp
            else:
                return k * amp**2
        
   
    def get_H1(self):
        """
        Get H1 FRF averaged estimator (receptance), preferable call via get_FRF().

        :return: H1 FRF estimator matrix (ndarray) of shape (response DOF, excitation DOF, freqency points).
        :rtype: ndarray
        """
        with np.errstate(invalid='ignore'):
            freq_len = np.fft.rfftfreq(self.fft_len, 1. / self.sampling_freq).shape[0]
            H1 = np.zeros((self.resp.shape[1], self.exc.shape[1], freq_len), dtype="complex128")
            
            if self.exc.shape[1] == 1:  # SISO, SIMO
                #print("single input")
                H1 = self.frf_norm * self.S_XF / self.S_FF
            else:  # MISO, MIMO
                if (self.S_FF[:,:,0].shape == (2,2)) or (self.S_FF[:,:,0].shape == (3,3)):
                    for i in range(freq_len):
                        H1[:,:,i] = self.frf_norm * (self.S_XF[:,:,i] @ self._analytical_matrix_inverse(self.S_FF[:,:,i]))
                else:
                    for i in range(freq_len):
                        H1[:,:,i] = self.frf_norm * (self.S_XF[:,:,i] @ np.linalg.inv(self.S_FF[:,:,i]))
            return  (H1 * self.frf_conversion) / self._correct_time_delay()
        
    # OK:    
    def get_H2(self):
        """
        H2 FRF averaged estimator (receptance), preferable call via get_FRF().

        :return: H2 FRF estimator matrix (ndarray) of shape (response DOF, excitation DOF, freqency points).
        :rtype: ndarray
        """
        with np.errstate(invalid='ignore'):
            freq_len = np.fft.rfftfreq(self.fft_len, 1. / self.sampling_freq).shape[0]
            H2 = np.zeros((self.resp.shape[1], self.exc.shape[1], freq_len), dtype="complex128")
            
            if self.exc.shape[1] == 1:  # SISO, SIMO
                #print("single input")
                for i in range(self.resp.shape[1]):
                    H2[i,:,:] = self.frf_norm * (self.S_XX[i,i,:] / self.S_FX[:,i,:])
            else:
                if (self.S_FX[:,:,0].shape == (2,2)) or (self.S_FX[:,:,0].shape == (3,3)):
                    for i in range(freq_len):
                        H2[:,:,i] = self.frf_norm * (self.S_XX[:,:,i] @ self._analytical_matrix_inverse(self.S_FX[:,:,i]))
                else:
                    if (self.S_FX[:,:,0].shape[0] == self.S_FX[:,:,0].shape[1]):
                        for i in range(freq_len):
                            H2[:,:,i] = self.frf_norm * (self.S_XX[:,:,i] @ np.linalg.inv(self.S_FX[:,:,i]))
                    else:
                        for i in range(freq_len):
                            H2[:,:,i] = self.frf_norm * (self.S_XX[:,:,i] @ np.linalg.pinv(self.S_FX[:,:,i]))
            return (H2 * self.frf_conversion) / self._correct_time_delay()
        
    def get_Hv(self):
        """
        Get Hv FRF averaged estimator (receptance), preferable call via get_FRF().

        Literature:
            [1] Kihong and Hammond: Fundamentals of Signal Processing for
                Sound and Vibration Engineers, page 293.

        :return: Hv FRF estimator matrix (ndarray).
        :rtype: ndarray
        """
        
        with np.errstate(invalid='ignore'):
            freq_len = np.fft.rfftfreq(self.fft_len, 1. / self.sampling_freq).shape[0]
            Hv = np.zeros((self.resp.shape[1], self.exc.shape[1], freq_len), dtype="complex128")
            
            if self.exc.shape[1] != 1:  # MISO, MIMO
                raise Exception("Currently not implemented for MISO and MIMO systems.")
            
            if self.exc.shape[1] == 1:  # SISO, SIMO
                k = 1  # ratio of the spectra of measurement noises
                #print("single input")
                for i in range(self.resp.shape[1]):
                    Hv[i,:,:] = self.frf_norm * ((self.S_XX[i,i,:] - k * self.S_FF[0,0,:] + np.sqrt(\
                        (k * self.S_FF[0,0,:] - self.S_XX[i,i,:]) ** 2 + 4 * k * np.conj(self.S_FX[0,i,:]) * self.S_FX[0,i,:]))\
                                                / (2 * self.S_XF[i,0,:]))
            return (Hv * self.frf_conversion )/ self._correct_time_delay()
    
        
    def get_FRF(self, type='default', form='receptance'):
        """
        Returns the default FRF function set at init.

        :param type: Choose default (as set at init) or H1, H2, Hv or ODS.
        :type type: str
        :param form: Choose receptance, mobility, accelerance.
        :type form: str
        :return: FRF estimator matrix (ndarray).
        :rtype: ndarray
        """
        if type=='default':
            type=self.frf_type
        if not (type in _FRF_TYPES):
            raise Exception('wrong FRF type given %s (can be %s)'
                            % (type, _FRF_TYPES))

        if type == 'H1':
            receptance = self.get_H1()
        if type == 'H2':
            receptance = self.get_H2()
        if type == 'Hv':
            receptance = self.get_Hv()
        if type == 'ODS':
            receptance = self.get_ods_frf()

        if not (form in _FRF_FORM):
            raise Exception('wrong FRF form given %s (can be %s)'
                            % (form, _FRF_FORM))
        
        # convert to requested frf form:
        if form=='accelerance':
            return receptance * np.power(1.j * self.get_w_axis(), 2)
        if form=='mobility':
            return receptance * np.power(1.j * self.get_w_axis(), 1)
        return receptance
        

    def get_coherence(self):
        """
        Get coherence.

        :return: Coherence ndarray of shape (response DOF, freqency points).
        :rtype: ndarray
        """
        with np.errstate(invalid='ignore'):
            if self.exc.shape[1] == 1:  # SISO, SIMO
                return np.abs(self.get_H1()[:,0,:] / self.get_H2()[:,0,:])
            else:  # MIMO, MISO
                freq_len = np.fft.rfftfreq(self.fft_len, 1. / self.sampling_freq).shape[0]
                coh = np.zeros((self.resp.shape[1], freq_len), dtype="complex128")
            
                for i in range(self.resp.shape[1]):
                    if (self.S_FF[:,:,0].shape == (2,2)) or (self.S_FF[:,:,0].shape == (3,3)):
                        for j in range(freq_len):
                            coh[i,j] = ((self.S_XF[i,:,j] @ self._analytical_matrix_inverse(self.S_FF[:,:,j])) @ self.S_FX[:,i,j])
                    else:
                        for j in range(freq_len):
                            coh[i,j] = ((self.S_XF[i,:,j] @ np.linalg.inv(self.S_FF[:,:,j])) @ self.S_FX[:,i,j])
                    coh[i] = coh[i] / self.S_XX[i,i]
                return coh


    def _ini_exc_and_resp_window(self):
        """
        Sets the windows for exc and resp signals.

        :return:
        """
        
        self.exc_window_data, self.exc_window_amp_norm = self._get_window_sub(length=self.exc.shape[2], 
                                                                              window=self.exc_window)
        self.resp_window_data, self.resp_window_amp_norm = self._get_window_sub(length=self.resp.shape[2], 
                                                                                window=self.resp_window)

        self.frf_norm = self.resp_window_amp_norm / self.exc_window_amp_norm
        #print("frf_norm (resp/exc)): ", self.frf_norm)
    
    
    def _ini_lengths_and_csd_window(self, length):
        """
        Sets the lengths used later in csd and sets csd_window.

        :param length: Length of data expected.
        :type length: int
        :return:
        """
        #print("_ini_lengths_and_csd_window self.samples: ", self.samples)
        #print("_ini_lengths_and_csd_window self.exc.shape[2]: ", self.exc.shape[2])
        if self.samples is None:
            self.samples = length
        #elif self.samples != self.exc.shape[2]:
            #raise ValueError('data length changed.')
        
        self.csd_window_data, self.csd_window_amp_norm = self._get_window_sub(length=self.samples, 
                                                                              window=self.csd_window)
            
    def _add_to_archive(self, exc, resp):
        """
        Add time data to the archive for later data analysis.

        :param exc: Excitation array.
        :param resp: Response array.
        :return:
        """
        if self.archive_time_data:
            self.resp_archive.append(resp)
            self.exc_archive.append(exc)
            
    def get_archive(self):
        """
        Returns the time archive. If not available, it returns None.

        :return: (excitation, response) time archive.
        """
        if self.archive_time_data:
            return self.exc_archive, self.resp_archive
        else:
            return None, None
        
    def _is_overflow(self, data, overflow_samples=3, verbose=0):
        """
        Check data for overflow.
    
        :param data: Excitation or response array.
        :param overflow_samples: Number of samples that need to be equal to max or overflow identification.
        :param verbose: 0 or 1 - prints overflow status.
        :return: Overflow status (True/False).
        :rtype: bool
        """
        def _overflow_check(time_series, os):
            s = np.sort(np.abs(time_series))[::-1]
            over = s == np.max(s)
            if np.sum(over) >= os:
                return True
            else:
                return False

        if data.ndim == 3:
            overflow_status_array = np.empty((data.shape[0], data.shape[1]), dtype=bool)
            for measurement_num in range(data.shape[0]):
                for dof_num in range(data.shape[1]):
                    overflow_status = _overflow_check(data[measurement_num, dof_num], overflow_samples)
                    if overflow_status:
                        if verbose == 1:
                            print(f'Overflow - measurement index number {measurement_num}, dof index number {dof_num}!')
                    overflow_status_array[measurement_num, dof_num] = overflow_status
            return overflow_status_array
        elif data.ndim == 2:
            overflow_status_array = np.empty((data.shape[0]), dtype=bool)
            for measurement_num in range(data.shape[0]):
                overflow_status = _overflow_check(data[measurement_num], overflow_samples)
                if overflow_status:
                    if verbose == 1:
                        print(f'Overflow - measurement index number {measurement_num}!')
                overflow_status_array[measurement_num] = overflow_status
            return overflow_status_array
        elif data.ndim == 1:
            overflow_status = _overflow_check(data, overflow_samples)
            if overflow_status:
                if verbose == 1:
                    print("Overflow!")
            return overflow_status
        else:
            raise Exception('Incorrect input array shape!')
        
    def _is_double_impact(self, data, limit=1e-3, verbose=0):
        """
        Check data for double-impact.

        See: at the end of http://scholar.lib.vt.edu/ejournals/MODAL/ijaema_v7n2/trethewey/trethewey.pdf
        
        :param data: Excitation array.
        :param limit: Ratio of freqency content of the double vs single hit - smaller number means more sensitivity.
        :param verbose: 0 or 1 - prints the ratio of the double PSD main lobe vse after lobe, 
            prints double impact status.
        :return: Double-hit status (True/False).
        :rtype: bool
        """

        def _double_impact_check(x):
            skip_low_freq = 10
            # first PSD
            X = np.fft.rfft(x)[skip_low_freq:]  
            X = 2 / self.sampling_freq * np.real(X * np.conj(X))/len(x)
            # second PSD: look for oscillations in PSD
            X2 = np.fft.rfft(X)
            X2 = 2 / self.sampling_freq * np.real(X2 * np.conj(X2))/len(X)
            upto = int(0.01 * len(X2))
            max_impact = np.max(X2[:upto])
            max_after_impact = np.max(X2[upto:])
            if verbose == 1:
                print(f'Ratio of the double PSD main lobe vse after lobe {max_after_impact / max_impact:g}')
            if max_after_impact / max_impact > limit:
                return True
            else:
                return False

        if data.ndim == 3:
            double_hit_array = np.empty((data.shape[0], data.shape[1]), dtype=bool)
            for measurement_num in range(data.shape[0]):
                for dof_num in range(data.shape[1]):
                    double_hit = _double_impact_check(data[measurement_num, dof_num])
                    if double_hit:
                        if verbose == 1:
                            print(f'Double impact - measurement index number {measurement_num}, dof index number {dof_num}!')
                    double_hit_array[measurement_num, dof_num] = double_hit
            return double_hit_array
        elif data.ndim == 2:
            double_hit_array = np.empty((data.shape[0]), dtype=bool)
            for measurement_num in range(data.shape[0]):
                double_hit = _double_impact_check(data[measurement_num])
                if double_hit:
                    if verbose == 1:
                        print(f'Double impact - measurement index number {measurement_num}!')
                double_hit_array[measurement_num] = double_hit
            return double_hit_array
        elif data.ndim == 1:
            double_hit = _double_impact_check(data)
            if double_hit:
                if verbose == 1:
                    print("Double impact!")
            return double_hit
        else:
            raise Exception('Incorrect input array shape!')
        
    def _correct_time_delay(self):
        """
        Corrects the FRF with regards to the ``time_delay``.

        :return: Time delay correction.
        :rtype: ndarray
        """
        return (np.exp(1j * self.get_w_axis() * self.resp_delay))

    def _analytical_matrix_inverse(self, matrix):
        """
        Calculate faster analytical inverse of 2x2 or 3x3 matrix.

        :param matrix: Array of shape (2x2) or (3x3).
        :return: Matrix inverse array.
        :rtype: ndarray
        """
        if matrix.shape == (2,2):
            a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
            det = a * d - b * c
            
            if det == 0:
                raise ValueError("Singular matrix.")
            
            inverse_matrix = [[d / det, -b / det],
                            [-c / det, a / det]]
            
            return np.array(inverse_matrix)
        
        if matrix.shape == (3,3):
            a, b, c = matrix[0]
            d, e, f = matrix[1]
            g, h, i = matrix[2]
            
            det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
            
            if det == 0:
                raise ValueError("Singular matrix.")
            
            inverse_matrix = [
                [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
                [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
                [(d * h - e * g) / det, (g * b - a * h) / det, (a * e - b * d) / det]
            ]

            return np.array(inverse_matrix)
        
if __name__ == '__main__':
    pass