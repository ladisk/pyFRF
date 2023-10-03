pyFRF
======

Frequency response function as used in structural dynamics.
-----------------------------------------------------------
For more information check out the showcase examples and see documentation: LINK

Basic ``pyFRF`` usage:
---------------------

Make an instance of ``FRF`` class:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    a = pyFRF.FRF(
        sampling_freq,
        exc=None,
        resp=None,
        exc_type='f', resp_type='a',
        window='none',
        weighting='linear',
        fft_len=None,
        nperseg=None,
        noverlap=None,
        archive_time_data=False,
        frf_type='H1',
        copy=True
    )

Adding data:
~~~~~~~~~~~~
We can add the excitation and response data at the beginning through ``exc`` and ``resp`` arguments, otherwise, the excitation and response 
data can be added later via ``add_data()`` method:

.. code:: python

    a.add_data(exc, resp)

Computing FRF:
~~~~~~~~~~~~~~
Preferable way to get the frequency response functions is via ``get_FRF()`` method:

.. code:: python

    frf = a.get_FRF(type="default", form="receptance")

We can also directly get the requested FRF via other methods: ``get_H1()``, ``get_H2()``, ``get_Hv()`` and, ``get_ods_frf()``:

.. code:: python

    H1 = a.get_H1()
    H2 = a.get_H2()
    Hv = a.get_Hv()
    ods_frf = a.get_ods_frf()


|pytest|

|binder| to test the *Showcase.ipynb*.

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/ladisk/pyFRF/main
.. |pytest| image:: https://github.com/ladisk/pyFRF/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/ladisk/pyFRF/actions
