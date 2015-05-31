#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import PAL2

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


setup(
    name='PAL2',
    version=PAL2.__version__,
    author='Justin A. Ellis',
    author_email='justin.ellis18@gmail.com',
    url='https://github.com/jellis18/PAL2',
    packages=['PAL2'],
    package_dir = {'PAL2': 'PAL2'},
    scripts=['PAL2/PAL2_run.py', 'PAL2/makeH5File.py'],
    zip_save=False,
    license='GPLv3',
    description='PTA analysis software',
    long_description=open('README.md').read() + '\n\n'
                    + '---------\n\n'
                    + open('HISTORY.md').read(),
    package_data={'PAL2': ['pulsarDistances.txt']},
    install_requires=['numpy', 'scipy', 'h5py',
                      'ephem', 'healpy', 'numexpr',
                      'statsmodels',
                      'libstempo>=2.2.2'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
