The tutorial
============
A case of typical usage is presented here.

For a more detailed use with real examples visit the :doc:`showcase <Showcase>` page.

Installation
------------
To use pyFRF, first install it using pip:

.. code-block:: console

   $ pip install pyFRF

Instance of the ``FRF`` class
-----------------------------
We start by creating the FRF object the following way:

.. code:: python

    a = pyFRF.FRF(
        sampling_freq,
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
        copy=True
    )

``sampling_freq`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~
This argument determines the sampling frequency of excitation and response signals. If ``sampling_freq`` is of type ``int``,
its is asumed that both excitation and response signals have the same sampling frequency. It is possible to use signals with 
different sampling frequencies for exciatation and response signals if we pass the ``sampling_freq`` argumnet as 
``list[int, int]`` or ``tuple(int, int)``, where the first element represents the excitation sampling frequency and the second 
element represents the response sampling frequency.

``exc`` and ``resp`` arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Excitation and response signal arrays. The arrays can be of different shapes:

    * 1D array (single measurement, single input/output): ``(time_series)``
    * 2D array (multiple/single measurements, single input/output): ``(n_measurements, time_series)``
    * 3D array (general array shape, multiple/single measurements, multiple/single inputs/outputs): ``(n_measurements, exc_dofs or resp_dofs, time_series)``

.. note::
    For multiple inputs or multiple outputs in one measurement, the 3D array shape (**general array shape**) is required. 2D array shape is reserved 
    for single input/output multiple measurements.

``exc_type`` and ``resp_type`` arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used for the correct conversion of the FRF.

.. note::
    | Available ``exc_type`` options: ``'f', 'a', 'v', 'd', 'e'``
    | Available ``resp_type`` options: ``'a', 'v', 'd', 'e'``

``window`` argument
~~~~~~~~~~~~~~~~~~~
This argument determines the window used on the excitation and response signals or window that is used for cross spectral 
density computation. If ``window`` is of type ``list(str, str)`` or ``tuple(str, str)``, the first element represents the 
window that is used on the excitation signal and the second element represents the window that is used on response signal. 
If ``window`` is of type ``str``, the window is used for cross spectral density computation (used for averaging, random 
signals and MIMO systems).

.. note::
    Available windows: ``'none', 'force', 'exponential', 'hann', 'hamming', 'bartlett', 'blackman', 'kaiser'``

.. note::
    | For exponential window the percentage of the starting amplitude (``float``) is required (e.g. 10%: ``"exponential:0.1"``).
    | For force window the length of window (percentage (``float``) of full signal) is required (e.g. 10%: ``"force:0.1"``).

``resp_delay`` argument
~~~~~~~~~~~~~~~~~~~~~~~
Used if there is response time delay present (in seconds) with regards to the excitation - used for FFT (phase) correction.

``weighting`` argument
~~~~~~~~~~~~~~~~~~~~~~
Weighting that is used for newly added measurements to ``FRF`` object or when adding separate contionuous measurements. 
If all the measurements are added to FRF object at the same time the ``weighting`` argument has no effect - linear averaging 
is used.

.. note::
    Available wighting types: ``'linear', 'exponential'``

.. note::
    | For exponential weighting, the number of averages (``int``)  has to be specified (e.g. 5 averages: ``"exponential:5"``).

``fft_len`` argument
~~~~~~~~~~~~~~~~~~~~
The length of the FFT (zero-padding if longer than length of data). If ``None`` then the ``fft_len`` matches the time length.

``nperseg`` argument
~~~~~~~~~~~~~~~~~~~~
Length of each segment used for averaging while computing cross power spectral density. If ``None``, the whole time signal 
(data length) is used (no averaging).

``noverlap`` argument
~~~~~~~~~~~~~~~~~~~~~
Optional segments overlap. By default (if ``None``), then ``noverlap = nperseg // 2``. If ``nperseg`` is not specified (data length 
is used), then ``noverlap``  has no effect.

``archive_time_data`` argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used for archiving time data.

.. note::
    Can consume a lot of memory.

``frf_type`` argument
~~~~~~~~~~~~~~~~~~~~~
Default FRF type returned at ``self.get_frf()``.

``copy`` argument
~~~~~~~~~~~~~~~~~
Determines if the excitation and response arrays are copied (if data is not copied the applied window affects the source arrays).

Adding new data
---------------
Data can be added at object creation, by passing the excitation and response signals into ``exc`` and ``resp`` arguments. Data 
can also be added later via ``add_data()`` method - also useful for continuous measurements:

.. code:: python

    a.add_data(exc, resp)

Single input/output
~~~~~~~~~~~~~~~~~~~
The most **general array shape** of excitation and response signals for all different systems (SISO, SIMO, MISO, MIMO) is a 3D 
``ndarray`` of shape ``(n_measurements, exc_dofs or resp_dofs, time_series)``. 

When dealing with measurements where the excitation and/or response signals are measured at only one location (DOF) at the same time, 
the excitation/response array shape can be of 3 different shapes:

    * 1D array (single measurement): ``(time_series)``
    * 2D array (multiple/single measurements): ``(n_measurements, time_series)``
    * 3D array (general shape, multiple/single measurements): ``(n_measurements, 1, time_series)``

Multiple inputs/outputs
~~~~~~~~~~~~~~~~~~~~~~~
For measurements where the excitation and/or response signals are measured at multiple locations (DOFs) at the same time the **general 
array shape** must be used:

    * 3D array: ``(n_measurements, exc_dofs or resp_dofs, time_series)``

Checking data
-------------
Before adding data, it is useful to check if the measurements are ok. The ``is_data_ok()`` method checks if any of the provided 
excitation signals contains double impacts (for modal hammer impact testing measurements), or if excitation and response signals 
are overflowed. 

Argument ``overflow_samples`` represents the number of samples that need to be equal to max for overflow identification.

Argument ``double_impact_limit`` represents the ratio of freqency content of the double vs single hit. Smaller number means more 
sensitivity.

We can show the status of overflow and double impact measurements via argument ``verbose``.

.. code:: python

    a.is_data_ok(exc, resp, overflow_samples=3, double_impact_limit=1e-3, verbose=0)

Getting frequency and time axis
-------------------------------
To get the frequency axis array, we use the method ``get_f_axis()`` or ``get_w_axis()`` for angular frequency. We can also get delta 
frequency via ``get_df()`` method. For time axis we use the method ``get_t_axis()``:

.. code:: python

    freq = a.get_f_axis()
    w = a.get_w_axis()
    df = a.get_df()
    time = a.get_t_axis()

Obtaining FRF
-------------

``get_FRF()`` method
~~~~~~~~~~~~~~~~~~~~
Preferable way to get the frequency response functions is via ``get_FRF()`` method, where we use the ``type`` argument to specify 
the type of FRF. By default the returned FRF type is the one set at object creation (if no type is specified at object creation then 
the type is set to H1). With the argument ``form`` we control the returned form of the FRF (receptance by default).

As result we get the FRF matrix (``ndarray``) of shape ``(resp_dofs, exc_dofs, frequency_series)``:

.. code:: python

    frf = a.get_FRF(type="default", form="receptance")

.. note::
    | ``type`` argument options: ``'H1', 'H2', 'Hv', 'ODS'``
    | ``form`` argumnet options: ``'accelerance', 'mobility', 'receptance'``

Other (direct) methods
~~~~~~~~~~~~~~~~~~~~~~
We can also directly get the requested FRF via other methods: ``get_H1()``, ``get_H2()``, ``get_Hv()`` and, ``get_ods_frf()``. That 
way we can not control the form of the returned FRF. The returned FRF form of these methods is receptance:

.. code:: python

    H1 = a.get_H1()
    H2 = a.get_H2()
    Hv = a.get_Hv()
    ods_frf = a.get_ods_frf()

Coherence
---------
To get coherence we simply use the methods ``get_coherence()``, which returns the coherence ``ndarray`` of shape 
``(resp_dofs, frequency_series)``:

.. code:: python

    coh = a.get_coherence()