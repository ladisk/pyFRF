desc = """\
pyFRF
======

Frequency response function as used in structural dynamics.
-----------------------------------------------------------

For a showcase see: https://github.com/ladisk/pyFRF/blob/master/Showcase%20pyFRF.ipynb
"""

#from distutils.core import setup, Extension
from setuptools import setup
setup(name='pyFRF',
      version='0.41',
      author='Janko Slavič et al.',
      author_email='janko.slavic@fs.uni-lj.si',
      description='Frequency response function as used in structural dynamics.',
      url='https://github.com/ladisk/pyFRF',
      py_modules=['pyFRF','fft_tools'],
      #ext_modules=[Extension('lvm_read', ['data/short.lvm'])],
      long_description=desc,
      install_requires=['numpy']
      )