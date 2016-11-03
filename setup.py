desc = """\
pyFRF
======

Frequency response function as used in structural dynamics.
-----------------------------------------------------------
This package is part of the www.openmodal.com project.

For a showcase see: https://github.com/openmodal/pyFRF/blob/master/Showcase%20pyFRF.ipynb
"""

#from distutils.core import setup, Extension
from setuptools import setup, Extension
setup(name='pyFRF',
      version='0.2.1',
      author='Janko Slavič',
      author_email='janko.slavic@fs.uni-lj.si',
      description='Frequency response function as used in structural dynamics.',
      url='https://github.com/openmodal/pyFRF',
      py_modules=['pyFRF','fft_tools'],
      #ext_modules=[Extension('lvm_read', ['data/short.lvm'])],
      long_description=desc,
      requires=['numpy']
      )