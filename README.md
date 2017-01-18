# PAL2 (PTA Algorithm Library) #

[![DOI](https://zenodo.org/badge/16349842.svg)](https://zenodo.org/badge/latestdoi/16349842)

PAL2 is a Bayesian inference package for pulsar timing data. PAL2 is a re-write of 
the original PAL in which we use a very similar signal dictionary
formalism introduced in [piccard] (https://github.com/vhaasteren/piccard). This code is 
meant to perform a large number of analyses on pulsar timing data including:

* Noise characterization 
* Detection and characterization of GWB (isotropic and anisotropic) and continuous GWs
* Non-linear pulsar timing
* Dispersion measure variation modeling

## Dependencies ##

The code has a large number of Python and non-Python dependencies. 
The python dependencies should should be installed automatically when 
following the directions below. You will have to manually install the
following non-Python packages (preferably with a package manager):

* [tempo2] (https://bitbucket.org/psrsoft/tempo2.git)
* hdf5
* healpix
* openmpi

PAL2 also supports MultiNest but it must be installed separately.

* [MultiNest] (http://ccpforge.cse.rl.ac.uk/gf/project/multinest/) (version 2.18 or later])
* [PyMultiNest] (https://github.com/JohannesBuchner/PyMultiNest)

## Installation ##

You will need to have [tempo2](http://www.atnf.csiro.au/research/pulsar/tempo2/index.php?n=Main.Download) and a working C, C++ and Fortran compiler.

### Anaconda Install ###

The easiest way for installation is to use [Anaconda](http://docs.continuum.io/anaconda/install) or [Miniconda](http://conda.pydata.org/miniconda.html).
Anaconda comes with many of the packages that we will need whereas Miniconda only comes with a small base Python installation. Once you have Anaconda 
(or Miniconda) installed you should get a [academic license](https://www.continuum.io/anaconda-academic-subscriptions-available) and install the 
MKL optimizations for fast linear algebra. With either installation it is best to set up an python environment with all of the needed dependencies. 
This can easily be done in Anaconda with

```
conda env create -f environment.yml
source activate pal2_conda
```

Be sure to follow the instructions and activate the Anaconda environment. This will create an Anaconda environment with nearly all of the required dependencies installed except [libstempo](https://github.com/vallis/libstempo). 
The libstempo package can be installed with

```pip install libstempo --install-option="--with-tempo2=$TEMPO2"```

if you have your TEMPO2 environment variable set correctly. To finalize the installation do

```python setup.py install```

### Standard Install ###

You can also install PAL2 with a [python virutal environment](https://virtualenvwrapper.readthedocs.org/en/latest/). However for some clusters this may not be appropriate and you can follow the instructions below but append a --user flag on all of the ``pip`` and ``setup.py`` commands. Once you have the virualenv activated first do:

```
pip install numpy
pip install cython
pip install -r requirements.txt
```

This will install most of the dependencies except [libstempo](https://github.com/vallis/libstempo). The libstempo package can be installed with

```pip install libstempo --install-option="--with-tempo2=$TEMPO2"```

if you have your TEMPO2 environment variable set correctly. To finalize the installation do

```python setup.py install```


## Known Issues ##

There are a few known issues with the PAL2 code that are being addressed:

* During installation ``pip`` may have problems with the healpy package on Mac OSX. To overcome any potential problems you may need to include ``export CC=/path/to/gcc`` and ``export CXX=/path/to/g++``.
* the ``mark6`` likelihood function is currently not compatible with cross correlations in the GWB. You will need to use the ``--noCorrelations`` flag with ``PAL2_run.py`` or use the ``incCorrelations=False`` flag in the likelihood if using your own interface.
* You may have problems installing ``basemap`` with pip. You will first need to install ``geo`` with your package manager. Then you can install ``basemap`` via ``pip install basemap --allow-external basemap --allow-unverified basemap``.

## Example Data ##

We have included two datasets for testing purposes. 

* Open dataset 1 from the [IPTA MDC](http://www.ipta4gw.org/?page_id=126). Good for testing GWB detection and characterization.
* The [5-year data release](http://data.nanograv.org) from NANOGrav. Good for testing realistic noise models.

## Manual ##

Example ipython notebook can be found in [here](https://github.com/jellis18/PAL2/blob/master/demo/PAL2_demo.ipynb).

## Contact ##
[_Justin Ellis_] (mailto:justin.ellis18@gmail.com)
