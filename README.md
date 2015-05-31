# PAL2 (PTA Algorithm Library) #

PAL2 is a Bayesian inference package for pulsar timing data. PAL2 is a re-write of 
the original PAL in which we use a very similar signal dictionary
formalism introduced in [piccard] (https://github.com/vhaasteren/piccard). This code is 
meant to perform a large number of analyses on pulsar timing data including:

* Noise characterization 
* Detection and characterization of GWB (isotropic and anisotropic) and continuous GWs
* Non-linear pulsar timing
* Dispersion measure variation modeling

## Dependencies ##

The code has a large number of dependencies all of which should be installed
automatically when using ``setup.py`` or ``pip``.

* Python 2.7
* [numpy](http://numpy.scipy.org)
* [scipy](http://numpy.scipy.org)
* [matplotlib](http://matplotlib.org), for plotting only
* [h5py](http://www.h5py.org)
* [tempo2](http://www.atnf.csiro.au/research/pulsar/tempo2/index.php?n=Main.Download)
* [libstempo](https://github.com/vallis/mc3pta/tree/master/stempo) (version >= 2.2.2)
* [pyephem](http://rhodesmill.org/pyephem/)
* [statsmodels](http://statsmodels.sourceforge.net)
* [healpy](https://healpy.readthedocs.org)
* [numexpr](https://github.com/pydata/numexpr)

PAL2 also supports MultiNest but it must be installed separately.

* [MultiNest] (http://ccpforge.cse.rl.ac.uk/gf/project/multinest/) (version 2.18 or later])
* [PyMultiNest] (https://github.com/JohannesBuchner/PyMultiNest)

## Installation ##

PAL2 is most easliy installed with 

```python setup.py install --user```

but can also be installed from [Pypi](https://pypi.python.org/pypi/PAL2/) via

```pip install PAL2```

By default the ``PAL2_run.py`` and ``makeH5File.py`` scripts will be installed to ``$HOME/Library/Python/2.7/bin/``. Make sure this is in your path in order to run these scripts. Otherwise you can choose the install directory by using the ``--install-scripts `` flag to `setup.py install`.

## Known Issues ##

There are a few known issues with the PAL2 code that are being addressed:

* During installation ``pip`` may have problems with the healpy package on Mac OSX. To overcome any potential problems you may need to include ``export CC=/path/to/gcc`` and ``export CXX=/path/to/g++``.
* the ``mark9`` and ``mark6`` likelihood functions are currently not compatible with cross correlations in the GWB. You will need to use the ``--noCorrelations`` flag with ``PAL2_run.py`` or use the ``incCorrelations=False`` flag in the likelihood if using your own interface.

## Example Data ##

We have included two datasets for testing purposes. 

* Open dataset 1 from the [IPTA MDC](http://www.ipta4gw.org/?page_id=126). Good for testing GWB detection and characterization.
* The [5-year data release](http://data.nanograv.org) from NANOGrav. Good for testing realistic noise models.

## Manual ##

Example ipython notebooks are under construction...

## Contact ##
[_Justin Ellis_] (mailto:justin.ellis18@gmail.com)
