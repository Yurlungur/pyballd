#!/usr/bin/env python2

"""setup.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-05-13 16:22:08 (jmiller)>

This is the setup.py file for the pyballd package
which is an ellitpic solver for axisymmetric problems
implemented in Python.
"""

from setuptools import setup,find_packages

exec(open('pyballd/version.py','r').read())
with open('README.md','r') as f:
    long_description = f.read()

setup(
    name='pyballd',
    version=__version__,
    description='an ellitpic solver for axisymmetric problems',
    long_description=long_description,
    url='https://github.com/Yurlungur/pyballd',
    author = 'Jonah Miller',
    author_email = 'jonah.maxwell.miller@gmail.com',
    license = 'LGPLv3+',
    classifiers = [
        'Development status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Environment :: Console',
        'License :: GNU Lesser General Public License v3+',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    keywords = 'simulations relativity science computing',
    packages = find_packages(),
    install_requires = ['numpy','scipy','matplotlib']
)
