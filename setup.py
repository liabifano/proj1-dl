#!/usr/bin/env python
from setuptools import setup, find_packages


setup(name='deep',
      url='',
      author='',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version='0.0.1',
      install_requires=['pytest==2.9.2',
                        'scikit-learn==0.19.1',
                        'scipy==1.0.0'],
      include_package_data=True,
      zip_safe=False)