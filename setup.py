#!/usr/bin/env python3

from setuptools import setup

setup(name = 'machine learning',
      version = '1.0',
      description = 'Machine Learning Stanford',
      author = 'Lai, Chih-An',
      author_email = '',
      url = '',
      packages = ['ex1', 'ex2', 'ex3'],
      install_requires=[
          'numpy',
      ],
      py_modules = ['ex1.linearRegression', 'ex2.logisticRegression', 
          'ex3.multiClassification'],
     )
