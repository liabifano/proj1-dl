#!/usr/bin/env python
from setuptools import setup, find_packages


setup(name='proj1-dl',
      url='',
      author='',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version='0.0.1',
      install_requires=['pytest==2.9.2'],
      include_package_data=True,
      zip_safe=False)