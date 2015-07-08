#!/usr/bin/env python3

from setuptools import setup, Command

class CleanCommand(Command):
    '''Custom clean command to tidy up the project root.'''
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def rum(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    name = 'machineLearningStanford',
    version = '0.0',
    description = 'Machine Learning Stanford',
    author = 'Lai, Chih-An',
    author_email = 'chihan.lai@gmail.com',
    url = 'https://github.com/ZianLai/machineLearningStanford',
    packages = ['ex1', 'ex2', 'ex3'],
    install_requires=[
        'numpy',
    ],
    py_modules = ['ex1.linearRegression', 'ex2.logisticRegression', 
          'ex3.multiClassification'],
    zip_safe = False, 
    cmdclass={'clean': CleanCommand,}
)
