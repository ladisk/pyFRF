
# Copyright (C) 2014-2017 Martin Česnik, Matjaž Mršnik, Miha Pirnat, Janko Slavič, Blaž Starc (in alphabetic order)
# 
# This file is part of pyFRF.
# 
# pyFRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# pyFRF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with pyFRF.  If not, see <http://www.gnu.org/licenses/>.


desc = """\
pyFRF
======

Frequency response function as used in structural dynamics.
-----------------------------------------------------------
This package is part of the www.openmodal.com project.

For a showcase see: https://github.com/openmodal/pyFRF/blob/master/Showcase%20pyFRF.ipynb
"""

#from distutils.core import setup, Extension
from setuptools import setup
setup(name='pyFRF',
      version='0.37',
      author='Janko Slavič et al.',
      author_email='janko.slavic@fs.uni-lj.si',
      description='Frequency response function as used in structural dynamics.',
      url='https://github.com/openmodal/pyFRF',
      py_modules=['pyFRF','fft_tools'],
      #ext_modules=[Extension('lvm_read', ['data/short.lvm'])],
      long_description=desc,
      install_requires=['numpy']
      )