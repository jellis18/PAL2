import os
import sys
import numpy

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


setup(
    name="PAL2",
    version='2015.02',
    author="Justin A. Ellis",
    author_email="justin.ellis18@gmail.com",
    packages=["PAL2"],
    url="https://github.com/jellis18/PAL2",
    license="GPLv3",
    description="PTA analysis software",
    long_description=open("README.md").read() + "\n\n"
                    + "---------\n\n"
                    + open("HISTORY.md").read(),
    package_data={"": ["README.md", "HISTORY.md"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "h5py"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ]
    # For Cython code, include the following modules
    #,
    #ext_modules = cythonize(Extension('pal2.jitterext', ['pal2/jitterext.pyx'],
    #        include_dirs = [numpy.get_include()]))
)
