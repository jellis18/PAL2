#!/usr/bin/env python

from __future__ import division

import numpy as np
import h5py as h5
import os
import sys
import time
import json
import tempfile
import scipy.linalg as sl
import scipy.special as ss
from numpy.polynomial.hermite import hermval
import AnisCoefficientsV2 as ani

from PAL2 import PALutils
from PAL2 import PALdatafile
from PAL2 import PALpsr

# In order to keep the dictionary in order
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import matplotlib.pyplot as plt

"""

PALmodels.py

This file contains the model class that handles all elements or our model on a per pulsar
basis, including noise parameters, GW parameters, and timing model parameters (not implemented yet)

This file aims to retain compatibility with the dictionary types used in Rutger van Haasteren's
Piccard.

"""


class PTAmodels(object):

    """
    Constructor. Read data/model if filenames are given

    @param h5filename:      HDF5 filename with pulsar data
    @param jsonfilename:    JSON file with model
    @param pulsars:         Which pulsars to read ('all' = all, otherwise provide a
                            list: ['J0030+0451', 'J0437-4715', ...])
                            WARNING: duplicates are _not_ checked for.
    """

    def __init__(self, h5filename=None, jsonfilename=None,
                 pulsars='all', auxFromFile=True):
        self.clear()

        if h5filename is not None:
            self.initFromFile(h5filename, pulsars=pulsars)

            if jsonfilename is not None:
                self.initModelFromFile(jsonfilename, auxFromFile=auxFromFile)

    """
    Clear all the structures present in the object
    """
    # TODO: Do we need to delete all with 'del'?

    def clear(self):
        self.h5df = None
        self.psr = []
        self.ptasignals = []

        self.dimensions = 0
        self.pmin = None
        self.pmax = None
        self.pstart = None
        self.pwidth = None
        self.pamplitudeind = None
        self.initialised = False
        self.likfunc = 'mark3'
        self.orderFrequencyLines = False
        self.haveStochSources = False
        self.haveDetSources = False
        self.haveEnvelope = False
        self.haveScat = False
        self.haveExt = False
        self.haveFrequencyLines = False
        self.skipUpdateToggle = False

    """
    Initialise pulsar class from an HDF5 file

    @param filename:    Name of the HDF5 file we will be reading
    @param pulsars:     Which pulsars to read ('all' = all, otherwise provide a
                        list: ['J0030+0451', 'J0437-4715', ...])
                        WARNING: duplicates are _not_ checked for.
    @param append:      If set to True, do not delete earlier read-in pulsars
    """

    def initFromFile(self, filename, pulsars='all', append=False):
        # Retrieve the pulsar list
        self.h5df = PALdatafile.DataFile(filename)
        psrnames = self.h5df.getPulsarList()

        # Determine which pulsars we are reading in
        readpsrs = []
        if pulsars == 'all':
            readpsrs = psrnames
        else:
            # Check if all provided pulsars are indeed in the HDF5 file
            if np.all(
                    np.array([pulsars[ii] in psrnames for ii in \
                              range(len(pulsars))]) == True):
                readpsrs = pulsars
            elif pulsars in destpsrnames:
                pulsars = [pulsars]
                readpsrs = pulsars
            else:
                raise ValueError(
                    "ERROR: Not all provided pulsars in HDF5 file")

        # Free earlier pulsars if we are not appending
        if not append:
            self.psr = []

        # Initialise all pulsars
        for psrname in readpsrs:
            newpsr = PALpsr.Pulsar()
            newpsr.readFromH5(self.h5df, psrname)
            self.psr.append(newpsr)

    """
    Function to easily construct a model dictionary for all pulsars

    TODO: make more functionality for single puslars later
    """

    def makeModelDict(self, nfreqs=20, ndmfreqs=None,
                      incRedNoise=False, noiseModel='powerlaw', fc=None, logf=False,
                      incRedBand=False, incRedGroups=False, redGroups=None,
                      incRedExt=False, redExtModel='powerlaw', redExtFtrans=3e-8,
                      redExtNf=30,
                      incDMBand=False,
                      incORF=False,
                      incDM=False, dmModel='powerlaw',
                      incDMEvent=False, dmEventModel='shapelet', ndmEventCoeffs=3,
                      incRedFourierMode=False, incDMFourierMode=False,
                      incWavelet=False, nWavelets=1,
                      incGWWavelet=False, nGWWavelets=1,
                      incGWFourierMode=False,
                      incScattering=False, scatteringModel='powerlaw',
                      nscatfreqs=None,
                      incGWB=False, gwbModel='powerlaw',
                      incGWBAni=False, lmax=2,
                      incBWM=False, incDMX=False,
                      incDMXKernel=False, DMXKernelModel='linear',
                      incCW=False, incPulsarDistance=False, CWupperLimit=False,
                      mass_ratio=False, CWModel='standard',
                      varyEfac=True, separateEfacs=False, separateEfacsByFreq=False,
                      incEquad=False, separateEquads=False, separateEquadsByFreq=False,
                      incJitter=False, separateJitter=False, separateJitterByFreq=False,
                      incTimingModel=False, nonLinear=False, fulltimingmodel=False,
                      addPars=None, subPars=None,
                      incJitterEpoch=False, nepoch=None,
                      incNonGaussian=False, nnongaussian=3,
                      incEnvelope=False, envelopeModel='powerlaw',
                      incJitterEquad=False, separateJitterEquad=False,
                      separateJitterEquadByFreq=False,
                      efacPrior='uniform', equadPrior='log', jitterPrior='uniform',
                      jitterEquadPrior='log',
                      redAmpPrior='log', redSiPrior='uniform', GWAmpPrior='log',
                      GWSiPrior='uniform',
                      DMAmpPrior='log', DMSiPrior='uniform', redSpectrumPrior='log',
                      DMSpectrumPrior='log',
                      GWspectrumPrior='log',
                      incSingleFreqNoise=False, numSingleFreqLines=1,
                      incSingleFreqDMNoise=False, numSingleFreqDMLines=1,
                      singlePulsarMultipleFreqNoise=None,
                      multiplePulsarMultipleFreqNoise=None,
                      dmFrequencyLines=None,
                      orderFrequencyLines=False,
                      compression='None',
                      Tmax=None,
                      evalCompressionComplement=False,
                      likfunc='mark1'):

        signals = []

        # backwards compatibility for CW signals
        if CWupperLimit:
            CWModel = 'upperLimit'
        if mass_ratio:
            CWModel = 'mass_ratio'

        # start loop over pulsars
        npsr = len(self.psr)
        for ii, p in enumerate(self.psr):

            # how many frequencies
            if incDM:
                if ndmfreqs is None or ndmfreqs == "None":
                    ndmfreqs = nfreqs
            else:
                ndmfreqs = 0

            if incScattering:
                if nscatfreqs is None:
                    nscatfreqs = nfreqs
            else:
                nscatfreqs = 0

            if incDMEvent and dmEventModel == 'shapeletmarg':
                p.ndmEventCoeffs = ndmEventCoeffs
            else:
                p.ndmEventCoeffs = 0

            if separateEfacs or separateEfacsByFreq:
                if separateEfacs and ~separateEfacsByFreq:
                    pass

                # if both set, default to fflags
                else:
                    p.flags = p.fflags  # TODO: make this more elegant

                uflagvals = list(set(p.flags))  # Unique flags
                for flagval in uflagvals:
                    newsignal = OrderedDict({
                        "stype": "efac",
                        "corr": "single",
                        "pulsarind": ii,
                        "flagname": "efacequad",
                        "flagvalue": flagval,
                        "bvary": [varyEfac],
                        "pmin": [0.001],
                        "pmax": [10.0],
                        "pwidth": [0.1],
                        "pstart": [1.0],
                        "prior": efacPrior
                    })
                    signals.append(newsignal)
            else:
                newsignal = OrderedDict({
                    "stype": "efac",
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": [varyEfac],
                    "pmin": [0.001],
                    "pmax": [10.0],
                    "pwidth": [0.1],
                    "pstart": [1.0],
                    "prior": efacPrior
                })
                signals.append(newsignal)

            if incJitter:
                if separateJitter or separateJitterByFreq:
                    if separateJitter and ~separateJitterByFreq:
                        pass

                    # if both set, default to fflags
                    else:
                        p.flags = p.fflags

                    uflagvals = list(set(p.flags))  # Unique flags
                    for flagval in uflagvals:
                        newsignal = OrderedDict({
                            "stype": "jitter",
                            "corr": "single",
                            "pulsarind": ii,
                            "flagname": "jitter",
                            "flagvalue": flagval,
                            "bvary": [True],
                            "pmin": [0],
                            "pmax": [5],
                            "pwidth": [0.1],
                            "pstart": [0.333],
                            "prior": jitterPrior
                        })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype": "jitter",
                        "corr": "single",
                        "pulsarind": ii,
                        "flagname": "pulsarname",
                        "flagvalue": p.name,
                        "bvary": [True],
                        "pmin": [0],
                        "pmax": [5],
                        "pwidth": [0.1],
                        "pstart": [0.333],
                        "prior": jitterPrior
                    })
                    signals.append(newsignal)

            if incJitterEquad:
                if separateJitterEquad or separateJitterEquadByFreq:
                    if separateJitterEquad and ~separateJitterEquadByFreq:
                        pass

                    # if both set, default to fflags
                    else:
                        p.flags = p.fflags

                        # TODO: come up with better way to deal with this
                        avetoas, aveflags, U = \
                            PALutils.exploderMatrixNoSingles(p.toas,
                                                             np.array(p.flags), dt=10)

                    # uflagvals = list(set(p.flags))  # Unique flags
                    uflagvals = np.unique(aveflags)
                    for flagval in uflagvals:
                        newsignal = OrderedDict({
                            "stype": "jitter_equad",
                            "corr": "single",
                            "pulsarind": ii,
                            "flagname": "jitter_equad",
                            "flagvalue": flagval,
                            "bvary": [True],
                            "pmin": [-8.5],
                            "pmax": [-4.0],
                            "pwidth": [0.1],
                            "pstart": [-8.0],
                            "prior": jitterEquadPrior
                        })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype": "jitter_equad",
                        "corr": "single",
                        "pulsarind": ii,
                        "flagname": "pulsarname",
                        "flagvalue": p.name,
                        "bvary": [True],
                        "pmin": [-8.5],
                        "pmax": [-4.0],
                        "pwidth": [0.1],
                        "pstart": [-8.0],
                        "prior": jitterEquadPrior
                    })
                    signals.append(newsignal)

            if incJitterEpoch:
                bvary = [True] * nepoch[ii]
                pmin = [-10.0] * nepoch[ii]
                pmax = [-4.0] * nepoch[ii]
                pstart = [-9.0] * nepoch[ii]
                pwidth = [0.5] * nepoch[ii]
                prior = [jitterEquadPrior] * nepoch[ii]

                newsignal = OrderedDict({
                    "stype": "jitter_epoch",
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior
                })
                signals.append(newsignal)

            if incEquad:
                if separateEquads or separateEquadsByFreq:
                    if separateEquads and ~separateEquadsByFreq:
                        pass

                    # if both set, default to fflags
                    else:
                        p.flags = p.fflags

                    uflagvals = list(set(p.flags))  # Unique flags
                    for flagval in uflagvals:
                        newsignal = OrderedDict({
                            "stype": "equad",
                            "corr": "single",
                            "pulsarind": ii,
                            "flagname": "equad",
                            "flagvalue": flagval,
                            "bvary": [True],
                            "pmin": [-10.0],
                            "pmax": [-4.0],
                            "pwidth": [0.1],
                            "pstart": [-8.0],
                            "prior": equadPrior
                        })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype": "equad",
                        "corr": "single",
                        "pulsarind": ii,
                        "flagname": "pulsarname",
                        "flagvalue": p.name,
                        "bvary": [True],
                        "pmin": [-10.0],
                        "pmax": [-4.0],
                        "pwidth": [0.1],
                        "pstart": [-8.0],
                        "prior": equadPrior
                    })
                    signals.append(newsignal)

            if incRedNoise:
                if noiseModel == 'spectrum':
                    #nfreqs = numNoiseFreqs[ii]
                    bvary = [True] * nfreqs
                    pmin = [-18.0] * nfreqs
                    pmax = [-7.0] * nfreqs
                    pstart = [-18.0] * nfreqs
                    pwidth = [0.1] * nfreqs
                    prior = [redSpectrumPrior] * nfreqs
                elif noiseModel == 'powerlaw':
                    bvary = [True, True, False]
                    pmin = [-20.0, 0.02, 1.0e-11]
                    pmax = [-11.0, 6.98, 3.0e-9]
                    pstart = [-19.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                    prior = [redAmpPrior, redSiPrior, 'log']
                elif noiseModel == 'spectralModel':
                    bvary = [True, True, True]
                    pmin = [-28.0, 0.0, -4.0]
                    pmax = [-14.0, 12.0, 2.0]
                    pstart = [-22.0, 2.0, -1.0]
                    pwidth = [-0.2, 0.1, 0.1]
                    prior = [redAmpPrior, redSiPrior, 'uniform']

                newsignal = OrderedDict({
                    "stype": noiseModel,
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior
                })
                signals.append(newsignal)
            
            if incRedExt:
                if redExtModel == 'spectrum':
                    nf = redExtNf
                    bvary = [True] * nf
                    pmin = [-18.0] * nf
                    pmax = [-7.0] * nf
                    pstart = [-18.0] * nf
                    pwidth = [0.1] * nf
                    prior = [redSpectrumPrior] * nf
                    parids = ['rho_ext_' + str(f) for f in range(nf)]
                    noiseModel = 'ext_spectrum'
                elif redExtModel == 'powerlaw':
                    bvary = [True, True, False]
                    pmin = [-20.0, 1.02, 1.0e-11]
                    pmax = [-11.0, 6.98, 3.0e-9]
                    pstart = [-19.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                    prior = [redAmpPrior, redSiPrior, 'log']
                    parids = ['RN-Amp-ext', 'RN-Si-ext', 'RN-fL-ext']
                    noiseModel='ext_powerlaw'

                newsignal = OrderedDict({
                    "stype": noiseModel,
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior,
                    "parids":parids
                })
                signals.append(newsignal)
            
            if incEnvelope:
                if envelopeModel == 'spectrum':
                    model = 'env_spectrum'
                    bvary = [True] * 2
                    bvary += [True] * nfreqs
                    pmin = [p.toas.min() / 86400, np.log10(14 * 86400)]
                    pmin += [-18.0] * nfreqs
                    pmax = [p.toas.max() / 86400, np.log10(
                        (p.toas.max() + p.toas.min()) / 2)]
                    pmax += [-7.0] * nfreqs
                    pstart = [(p.toas.max() + p.toas.min()) / 2 / 86400,
                              np.log10(100.0*86400)]
                    pstart += [-18.0] * nfreqs
                    pwidth = [10.0, 10.0]
                    pwidth += [0.1] * nfreqs
                    prior = ['uniform', 'log']
                    prior += [redSpectrumPrior] * nfreqs
                    parids = ['env_tmean', 'env_sigma']
                    parids += ['rho_env_' + str(f) for f in range(nfreqs)]
                elif envelopeModel == 'powerlaw':
                    model = 'env_powerlaw'
                    bvary = [True] * 2
                    bvary += [True, True, False]
                    pmin = [p.toas.min() / 86400, np.log10(14 * 86400)]
                    pmin += [-20.0, 1.02, 1.0e-11]
                    pmax = [p.toas.max() / 86400, np.log10(
                        (p.toas.max() - p.toas.min()) / 2)]
                    pmax += [-11.0, 6.98, 3.0e-9]
                    pstart = [(p.toas.max() + p.toas.min()) / 2 / 86400,
                              np.log10(100.0*86400)]
                    pstart += [-19.0, 2.01, 1.0e-10]
                    pwidth = [10.0, 10.0]
                    pwidth += [0.1, 0.1, 5.0e-11]
                    prior = ['uniform', 'log']
                    prior += [redAmpPrior, redSiPrior, 'log']
                    parids = ['env_tmean', 'env_sigma']
                    parids += ['env-RN-Amplitude', 'env-RN-Si', 'env-RN-fL']

                newsignal = OrderedDict({
                    "stype": model,
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior,
                    "parids":parids
                })
                signals.append(newsignal)

            if incRedBand:
                lbands = [0, 1000, 2000]
                lhbands = [1000, 2000, 5000]
                for lb, hb in zip(lbands, lhbands):
                    if np.any(np.logical_and(p.freqs <= hb, p.freqs > lb)):
                        print lb, hb, np.sum(np.logical_and(p.freqs <= hb, p.freqs > lb))
                        bvary = [True, True, False]
                        pmin = [-20.0, 0.02, 1.0e-11]
                        pmax = [-11.0, 6.98, 3.0e-9]
                        pstart = [-19.0, 2.01, 1.0e-10]
                        pwidth = [0.1, 0.1, 5.0e-11]
                        prior = [redAmpPrior, redSiPrior, 'log']
                        parids = ['RN-Amplitude-{0}-{1}'.format(lb, hb),
                                  'RN-Si-{0}-{1}'.format(lb, hb),
                                  'RN-fL-{0}-{1}'.format(lb, hb)]
                        print parids

                        newsignal = OrderedDict({
                            "stype": 'powerlaw_band',
                            "corr": "single",
                            "pulsarind": ii,
                            "flagname": "pulsarname",
                            "flagvalue": p.name,
                            "bvary": bvary,
                            "pmin": pmin,
                            "pmax": pmax,
                            "pwidth": pwidth,
                            "pstart": pstart,
                            "prior": prior,
                            "parids": parids
                        })
                        signals.append(newsignal)

            if incDM:
                if dmModel == 'spectrum':
                    #nfreqs = ndmfreqs
                    bvary = [True] * ndmfreqs
                    pmin = [-14.0] * ndmfreqs
                    pmax = [-3.0] * ndmfreqs
                    pstart = [-7.0] * ndmfreqs
                    pwidth = [0.1] * ndmfreqs
                    prior = [DMSpectrumPrior] * nfreqs
                    DMModel = 'dmspectrum'
                elif dmModel == 'powerlaw':
                    bvary = [True, True, False]
                    pmin = [-20.0, 0.02, 1.0e-11]
                    pmax = [-6.5, 6.98, 3.0e-9]
                    pstart = [-13.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                    prior = [DMAmpPrior, DMSiPrior, 'log']
                    DMModel = 'dmpowerlaw'
                elif dmModel == 'se':
                    bvary = [True, True]
                    pmin = [-12.0, 14.0]
                    pmax = [-5.0, (p.toas.max() - p.toas.min()) / 2]
                    pstart = [-7.0, 36.0]
                    pwidth = [0.5, 10.0]
                    prior = ['log', 'linear']
                    DMModel = 'dmse'

                newsignal = OrderedDict({
                    "stype": DMModel,
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior
                })
                signals.append(newsignal)

            if incDMBand:
                lbands = [0, 1000, 2000]
                lhbands = [1000, 2000, 5000]
                for lb, hb in zip(lbands, lhbands):
                    if np.any(np.logical_and(p.freqs <= hb, p.freqs > lb)):
                        print lb, hb, np.sum(np.logical_and(p.freqs <= hb, p.freqs > lb))
                        bvary = [True, True, False]
                        pmin = [-14.0, 0.02, 1.0e-11]
                        pmax = [-6.5, 6.98, 3.0e-9]
                        pstart = [-13.0, 2.01, 1.0e-10]
                        pwidth = [0.1, 0.1, 5.0e-11]
                        prior = [DMAmpPrior, DMSiPrior, 'log']
                        parids = ['DM-Amplitude-{0}-{1}'.format(lb, hb),
                                  'DM-Si-{0}-{1}'.format(lb, hb),
                                  'DM-fL-{0}-{1}'.format(lb, hb)]
                        print parids

                        newsignal = OrderedDict({
                            "stype": 'dmpowerlaw_band',
                            "corr": "single",
                            "pulsarind": ii,
                            "flagname": "pulsarname",
                            "flagvalue": p.name,
                            "bvary": bvary,
                            "pmin": pmin,
                            "pmax": pmax,
                            "pwidth": pwidth,
                            "pstart": pstart,
                            "prior": prior,
                            "parids": parids
                        })
                        signals.append(newsignal)

            if incScattering:
                if scatteringModel == 'spectrum':
                    bvary = [True]
                    bvary += [True] * nscatfreqs
                    pmin = [0.1]
                    pmin += [-20.0] * nscatfreqs
                    pmax = [7.0]
                    pmax += [-8.0] * nscatfreqs
                    pstart = [4.0]
                    pstart += [-15.0] * nscatfreqs
                    pwidth = [0.1]
                    pwidth += [0.1] * nscatfreqs
                    prior = ['uniform']
                    prior += [DMSpectrumPrior] * nscatfreqs
                    parids = ['scat-beta']
                    parids += ['rho_scat_' + str(f) for f in range(nscatfreqs)]
                    ScatteringModel = 'scatspectrum'
                elif scatteringModel == 'powerlaw':
                    bvary = [True, True, True, False]
                    pmin = [0.01, -20.0, 0.02, 1.0e-11]
                    pmax = [7.0, -11, 6.98, 3.0e-9]
                    pstart = [4.0, -16.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 0.1, 5.0e-11]
                    prior = ['uniform', DMAmpPrior, DMSiPrior, 'log']
                    parids = ['scat-beta', 'scat-Amplitude',
                              'scat-spectral-index', 'scat_fL']
                    ScatteringModel = 'scatpowerlaw'

                newsignal = OrderedDict({
                    "stype": ScatteringModel,
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior,
                    "parids": parids,
                    "nscatfreqs": nscatfreqs * 2,
                    "bwflags": p.bwflags
                })
                signals.append(newsignal)

            if incSingleFreqNoise:

                for jj in range(numSingleFreqLines):
                    newsignal = OrderedDict({
                        "stype": 'frequencyline',
                        "corr": "single",
                        "pulsarind": ii,
                        "flagname": "pulsarname",
                        "flagvalue": p.name,
                        "bvary": [True, True],
                        "pmin": [-9.0, -18.0],
                        "pmax": [-6.0, -9.0],
                        "pwidth": [-0.1, -0.1],
                        "pstart": [-7.0, -10.0],
                        "prior": ['log', 'log']
                    })
                    signals.append(newsignal)

            if incSingleFreqDMNoise:

                for jj in range(numSingleFreqDMLines):
                    newsignal = OrderedDict({
                        "stype": 'dmfrequencyline',
                        "corr": "single",
                        "pulsarind": ii,
                        "flagname": "pulsarname",
                        "flagvalue": p.name,
                        "bvary": [True, True],
                        "pmin": [-9.0, -14.0],
                        "pmax": [-6.0, -3.0],
                        "pwidth": [-0.1, -0.1],
                        "pstart": [-7.0, -10.0],
                        "prior": ['log', 'log']
                    })
                    signals.append(newsignal)

            if incDMEvent:
                if dmEventModel == 'shapeletmarg':
                    stype = 'dmshapeletmarg'
                    bvary = [True] * 2
                    pmin = [p.toas.min() / 86400, 14]
                    pmax = [p.toas.max() / 86400, 500]
                    pwidth = [10, 5]
                    pstart = [(p.toas.max() + p.toas.min()) / 2 / 86400, 30]
                    parids = ['dmShapeTime', 'dmShapeWidth']
                    priors = ['uniform', 'uniform']
                    p.ndmEventCoeffs = ndmEventCoeffs
                elif dmEventModel == 'shapelet':
                    stype = 'dmshapelet'
                    bvary = [True] * 2
                    bvary += [True] * ndmEventCoeffs
                    pmin = [p.toas.min() / 86400, 29]
                    pmin += [-0.01] * ndmEventCoeffs
                    pmax = [p.toas.max() / 86400, 500]
                    pmax += [0.01] * ndmEventCoeffs
                    pwidth = [10, 5]
                    pwidth += [0.0001] * ndmEventCoeffs
                    pstart = [(p.toas.max() - p.toas.min()) / 86400 / 2, 30]
                    pstart += [0.0] * ndmEventCoeffs
                    parids = ['dmShapeTime', 'dmShapeWidth']
                    names = [
                        'dmShapeAmp_{0}'.format(jj) for jj in range(ndmEventCoeffs)]
                    parids += names
                    priors = ['uniform', 'uniform']
                    prior += ['uniform'] * ndmEventCoeffs
                newsignal = OrderedDict({
                    "stype": stype,
                    "corr": "single",
                    "pulsarind": ii,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "parid": parids,
                    "prior": priors
                })
                signals.append(newsignal)

            if incDMXKernel:
                if DMXKernelModel == 'constant':
                    stype = 'DMXconstantKernel'
                    bvary = [True]
                    pmin = [-7.0]
                    pmax = [-1.0]
                    pwidth = [0.1]
                    pstart = [-6.0]
                    parids = ['DMXconstantKernel_amp']
                    priors = ['log']
                elif DMXKernelModel == 'se' or DMXKernelModel == 'SE':
                    stype = 'DMXseKernel'
                    bvary = [True] * 2
                    pmin = [-7.0, 14]
                    pmax = [-1.0, 50]
                    pwidth = [0.1, 1]
                    pstart = [-6.0, 30]
                    parids = ['DMXseKernel_amp', 'DMXseKernel_cts']
                    priors = ['log', 'linear']
                newsignal = OrderedDict({
                    "stype": stype,
                    "corr": "single",
                    "pulsarind": ii,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "parid": parids,
                    "prior": priors
                })
                signals.append(newsignal)

            if incTimingModel:
                if nonLinear:
                    # Get the parameter errors from libstempo. Initialise the
                    # libstempo object
                    p.initLibsTempoObject()

                    # add or subtract pars
                    if addPars is not None:
                        for key in addPars:
                            if key == 'DMX':
                                dmxs = np.array([par for par in p.ptmdescription
                                                 if 'DMX' in par])
                                for dmx in dmxs:
                                    print 'Turning on fit for {0}'.format(dmx)
                                    p.t2psr[dmx].fit = True
                            elif key == 'ORTHO':
                                print 'Using orthometric parameterization!'
                                if p.t2psr.binarymodel == 'DD':
                                    p.t2psr.binarymodel = 'DDH'
                                if p.t2psr.binarymodel == 'ELL1':
                                    p.t2psr.binarymodel = 'ELL1H'
                                sini = p.t2psr['SINI'].val
                                zeta = sini / (1 + np.cos(np.arcsin(sini)))
                                h3val = 4.9e-6 * p.t2psr['M2'].val * zeta ** 3
                                stigval = zeta
                                if h3val == 0 or stigval == 0:
                                    h3val, stigval = 1e-8, 0.6
                                p.t2psr['M2'].fit = False
                                p.t2psr['SINI'].fit = False
                                p.t2psr['M2'].val = 0.0
                                p.t2psr['SINI'].val = 0.0
                                p.t2psr['H3'].fit = True
                                p.t2psr['H3'].val = h3val
                                p.t2psr['STIG'].fit = True
                                p.t2psr['STIG'].val = stigval
                            elif key == 'MTOT':
                                p.t2psr.binarymodel == 'DDGR'
                                p.t2psr[key].fit = True
                            elif key == 'SHAPMAX':
                                sini = p.t2psr['SINI'].val
                                p.t2psr['SINI'].fit = False
                                p.t2psr[key].fit = True
                                p.t2psr[key].val = -np.log(1 - sini)
                                p.t2psr.binarymodel = 'T2'

                            else:
                                print 'Turning on fit for {0}'.format(key)
                                p.t2psr[key].fit = True
                                if key == 'SINI' and p.t2psr['SINI'].val == 0:
                                    p.t2psr[key].val = 0.99
                                if key == 'M2' and p.t2psr['M2'].val == 0:
                                    p.t2psr[key].val = 0.3
                        # p.t2psr.fit(iters=1)
                        p.ptmdescription = ['Offset'] + map(str, p.t2psr.pars())
                        p.ptmpars = np.array([0] + list(p.t2psr.vals()))
                        p.ptmparerrs = np.array([0] + list(p.t2psr.errs()))
                        #p.residuals = p.t2psr.residuals()
                        p.Mmat = p.t2psr.designmatrix(fixunits=True)

                    if subPars is not None:
                        for key in subPars:
                            if key == 'DMX':
                                dmxs = np.array([par for par in p.ptmdescription
                                                 if 'DMX' in par])
                                for dmx in dmxs:
                                    print 'Turning off fit for {0}'.format(dmx)
                                    p.t2psr[dmx].fit = False
                                    p.t2psr[dmx].val = 0.0
                            else:
                                print 'Turning off fit for {0}'.format(key)
                                p.t2psr[key].fit = False
                                p.t2psr[key].val = 0.0
                        # p.t2psr.fit(iters=1)
                        p.ptmdescription = ['Offset'] + map(str, p.t2psr.pars())
                        p.ptmpars = np.array([0] + list(p.t2psr.vals()))
                        p.ptmparerrs = np.array([0] + list(p.t2psr.errs()))
                        #p.residuals = p.t2psr.residuals()
                        p.Mmat = p.t2psr.designmatrix(fixunits=True)

                # Just do the timing-model fit ourselves here, in order to set
                # the prior.
                w = 1.0 / p.toaerrs ** 2
                Sigi = np.dot(p.Mmat.T, (w * p.Mmat.T).T)
                try:
                    cf = sl.cho_factor(Sigi)
                    Sigma = sl.cho_solve(cf, np.eye(Sigi.shape[0]))
                except np.linalg.LinAlgError:
                    U, s, Vh = sl.svd(Sigi)
                    if not np.all(s > 0):
                        raise ValueError("Sigi singular according to SVD")
                    Sigma = np.dot(Vh.T, np.dot(np.diag(1.0 / s), U.T))

                tmperrs = np.sqrt(np.diag(Sigma))
                tmpest = p.ptmpars
                p.ptmparerrs = tmperrs
                #tmperrs = p.ptmparerrs
                #tmpest2 = np.dot(Sigma, np.dot(p.Mmat.T, w*p.detresiduals))

                # Figure out which parameters we'll keep in the design matrix
                jumps = []
                dmx = []
                fds = []
                for tmpar in p.ptmdescription:
                    if tmpar[:4] == 'JUMP':
                        jumps += [tmpar]
                    if tmpar[:3] == 'DMX':
                        dmx += [tmpar]
                    if tmpar[:2] == 'FD':
                        fds += [tmpar]

                if fulltimingmodel:
                    newptmdescription = p.getNewTimingModelParameterList(keep=True,
                                                                         tmpars=[])
                else:
                    newptmdescription = p.getNewTimingModelParameterList(keep=True,
                                        tmpars=['Offset', 'F0', 'F1', 'RAJ', 'DECJ',
                                                'LAMBDA','ELONG', 'ELAT',
                                                'BETA', 'PMELONG', 'PMRA', 'PMDEC',
                                                'PMELAT', 'DM', 'DM1','DM2'] + \
                                                jumps + dmx + fds)
                    # newptmdescription = p.getNewTimingModelParameterList(keep=True, \
                    #    tmpars=['Offset', 'F0', 'F1', 'RAJ', 'DECJ', 'LAMBDA', \
                    #            'ELONG', 'ELAT', \
                    #            'BETA' ,'DM', 'DM1', 'DM2'] + jumps + dmx + fds)

                # Select the numerical parameters. These are the ones not
                # present in the quantities that getModifiedDesignMatrix
                # returned
                parids = []
                bvary = []
                pmin = []
                pmax = []
                pwidth = []
                pstart = []
                for jj, parid in enumerate(p.ptmdescription):
                    if not parid in newptmdescription:
                        parids += [parid]
                        bvary += [True]

                        if tmperrs[jj] == 0:
                            tmperrs[jj] = tmpest[jj]

                        # physical priors
                        # if parid == 'SINI':
                        #    pmin += [-tmpest[jj] - 1]
                        #    pmax += [1-tmpest[jj]]
                        #    pwidth += [tmperrs[jj]]
                        #    pstart += [0.0]
                        # elif parid == 'STIG':
                        #    pmin += [-1.0]
                        #    pmax += [1.0]
                        #    pwidth += [tmperrs[jj]]
                        #    if tmpest[jj] <= -1.0 or tmpest[jj] >= 1.0:
                        #        pstart += [0.5]
                        #    else:
                        #        pstart += [tmpest[jj]]
                        # elif parid == 'ECC' or parid == 'E':
                        #    pmin += [-tmpest[jj]]
                        #    pmax += [1.0-tmpest[jj]]
                        #    pwidth += [tmperrs[jj]]
                        #    pstart += [0.0]
                        # elif parid == 'KOM':
                        #    pmin += [-tmpest[jj]]
                        #    pmax += [360.0-tmpest[jj]]
                        #    pwidth += [tmperrs[jj]]
                        #    pstart += [0.0]
                        # elif parid == 'PX':
                        #    if tmpest[jj] < 0:
                        #        tmpest[jj] = 0.001
                        #    pmin += [-tmpest[jj]]
                        #    pmax += [500.0 * tmperrs[jj]]
                        #    pwidth += [tmperrs[jj]]
                        #    pstart += [0.0]
                        # elif parid == 'M2':
                        #    if tmpest[jj] < 0:
                        #        tmpest[jj] = 0.001
                        #    pmin += [-tmpest[jj]]
                        #    pmax += [500.0 * tmperrs[jj]]
                        #    pwidth += [tmperrs[jj]]
                        #    pstart += [0.0]
                        # elif parid == 'GAMMA':
                        #    pmin += [0.0]
                        #    pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                        #    pwidth += [tmperrs[jj]]
                        #    pstart += [tmpest[jj]]
                        # elif parid == 'SHAPMAX':
                        #    pmin += [0.0]
                        #    pmax += [50]
                        #    pstart += [tmpest[jj]]
                        #    pwidth += [tmperrs[jj]]
                        # elif parid == 'EDOT':
                        ##    pmin += [-500.0 * tmperrs[jj] + tmpest[jj]]
                        ##    pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                        ##    pwidth += [tmperrs[jj]]
                        ##    pstart += [0]
                        #
                        # make parameter be the 'parameter offset'
                        # elif parid == 'Offset':
                        ##    pmin += [-500]
                        ##    pmax += [500]
                        ##    pwidth += [0.1]
                        ##    pstart += [0]

                        # elif parid == 'F0':
                        ##    pmin += [-500]
                        ##    pmax += [500]
                        ##    pwidth += [0.1]
                        ##    pstart += [0]
                        ##
                        # elif parid == 'F1':
                        ##    pmin += [-500]
                        ##    pmax += [500]
                        ##    pwidth += [0.1]
                        ##    pstart += [0]

                        # else:
                        #    pmin += [-500.0 * tmperrs[jj]]
                        #    pmax += [500.0 * tmperrs[jj]]
                        #    pwidth += [tmperrs[jj]]
                        #    pstart += [0.0]

                        # physical priors
                        if parid == 'SINI':
                            pmin += [0.0]
                            pmax += [1.0]
                            pwidth += [tmperrs[jj]]
                            if tmpest[jj] <= -1.0 or tmpest[jj] >= 1.0:
                                pstart += [0.99]
                            else:
                                pstart += [tmpest[jj]]
                        elif parid == 'STIG':
                            pmin += [0.0]
                            pmax += [1.0]
                            pwidth += [tmperrs[jj]]
                            if tmpest[jj] <= -1.0 or tmpest[jj] >= 1.0:
                                pstart += [0.5]
                            else:
                                pstart += [tmpest[jj]]
                        elif parid == 'H3':
                            pmin += [0.0]
                            pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                            pwidth += [tmperrs[jj]]
                            if tmpest[jj] <= 0.0:
                                pstart += [0.5]
                            else:
                                pstart += [tmpest[jj]]
                        elif parid == 'ECC' or parid == 'E':
                            pmin += [0.0]
                            pmax += [1.0]
                            pwidth += [tmperrs[jj]]
                            pstart += [tmpest[jj]]
                        elif parid == 'KOM':
                            pmin += [0.0]
                            pmax += [360.0]
                            pwidth += [tmperrs[jj]]
                            pstart += [tmpest[jj]]
                        elif parid == 'PX':
                            if tmpest[jj] < 0:
                                tmpest[jj] = 0.001
                            pmin += [0.0]
                            pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                            pwidth += [tmperrs[jj]]
                            pstart += [tmpest[jj]]
                        elif parid == 'M2':
                            if tmpest[jj] < 0:
                                tmpest[jj] = 0.001
                            pmin += [0.0]
                            pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                            pwidth += [tmperrs[jj]]
                            pstart += [tmpest[jj]]
                        elif parid == 'GAMMA':
                            pmin += [0.0]
                            pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                            pwidth += [tmperrs[jj]]
                            pstart += [tmpest[jj]]
                        elif parid == 'SHAPMAX':
                            pmin += [0.0]
                            pmax += [50]
                            pstart += [tmpest[jj]]
                            pwidth += [tmperrs[jj]]
                        # elif parid == 'EDOT':
                        #    pmin += [-500.0 * tmperrs[jj] + tmpest[jj]]
                        #    pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                        #    pwidth += [tmperrs[jj]]
                        #    pstart += [0]

                        # make parameter be the 'parameter offset'
                        # elif parid == 'Offset':
                        #    pmin += [-500]
                        #    pmax += [500]
                        #    pwidth += [0.1]
                        #    pstart += [0]

                        # elif parid == 'F0':
                        #    pmin += [-500]
                        #    pmax += [500]
                        #    pwidth += [0.1]
                        #    pstart += [0]
                        #
                        # elif parid == 'F1':
                        #    pmin += [-500]
                        #    pmax += [500]
                        #    pwidth += [0.1]
                        #    pstart += [0]

                        else:
                            pmin += [-500.0 * tmperrs[jj] + tmpest[jj]]
                            pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                            pwidth += [tmperrs[jj]]
                            pstart += [tmpest[jj]]
                        print parid, pmin[-1], pmax[-1], pwidth[-1], pstart[-1], tmpest[jj]

                if nonLinear:
                    stype = 'nonlineartimingmodel'
                else:
                    stype = 'lineartimingmodel'

                newsignal = OrderedDict({
                    "stype": stype,
                    "corr": "single",
                    "pulsarind": ii,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "parid": parids,
                    "prior": 'flat'
                })
                signals.append(newsignal)

            if incRedFourierMode:
                bvary = [True] * 2 * nfreqs
                pmin = [-1e-5] * 2 * nfreqs
                pmax = [1e-5] * 2 * nfreqs
                pstart = [1e-9] * 2 * nfreqs
                pwidth = [1e-8] * 2 * nfreqs
                prior = [redSpectrumPrior] * 2 * nfreqs

                newsignal = OrderedDict({
                    "stype": 'redfouriermode',
                    "corr": "single",
                    "pulsarind": ii,
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior,
                })
                signals.append(newsignal)

            if incDMFourierMode:
                bvary = [True] * 2 * ndmfreqs
                pmin = [-1e-2] * 2 * ndmfreqs
                pmax = [1e-2] * 2 * ndmfreqs
                pstart = [0.0] * 2 * ndmfreqs
                pwidth = [1e-3] * 2 * ndmfreqs
                prior = [DMSpectrumPrior] * 2 * ndmfreqs

                newsignal = OrderedDict({
                    "stype": 'dmfouriermode',
                    "corr": "single",
                    "pulsarind": ii,
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior
                })
                signals.append(newsignal)

            if incGWFourierMode:
                bvary = [True] * 2 * nfreqs
                pmin = [-1e-5] * 2 * nfreqs
                pmax = [1e-5] * 2 * nfreqs
                pstart = [0.0] * 2 * nfreqs
                pwidth = [1e-7] * 2 * nfreqs
                prior = [GWSpectrumPrior] * 2 * nfreqs

                newsignal = OrderedDict({
                    "stype": 'gwfouriermode',
                    "corr": "gr",
                    "pulsarind": ii,
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior
                })
                signals.append(newsignal)

            if incNonGaussian:
                bvary = [True] * nnongaussian
                pmin = [-1.0] * nnongaussian
                pmax = [1.0] * nnongaussian
                pstart = [0.0] * nnongaussian
                pwidth = [0.1] * nnongaussian
                prior = ['uniform'] * nnongaussian

                newsignal = OrderedDict({
                    "stype": 'nongausscoeff',
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior
                })
                signals.append(newsignal)

            # pulsar distance for CW signal
            if incCW and incPulsarDistance:
                bvary = [True]
                pmin = [0]
                pmax = [10]
                pstart = [p.pdist]
                pwidth = [p.pdistErr]
                prior = ['gaussian']

                newsignal = OrderedDict({
                    "stype": 'pulsardistance',
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior
                })
                signals.append(newsignal)

            # separate pulsar phase and frequency
            if incCW and CWModel == 'free':
                bvary = [True] * 2
                pmin = [0, -9]
                pmax = [2*np.pi, -7]
                pstart = [np.pi, -8]
                pwidth = [0.1, 0.1]
                prior = ['uniform', 'log']
                parids = ['pphase_' + str(p.name), 'lpfgw_' + str(p.name)]

                newsignal = OrderedDict({
                    "stype": 'pulsarTerm',
                    "model": 'free',
                    "corr": "single",
                    "pulsarind": ii,
                    "flagname": "pulsarname",
                    "flagvalue": p.name,
                    "bvary": bvary,
                    "pmin": pmin,
                    "pmax": pmax,
                    "pwidth": pwidth,
                    "pstart": pstart,
                    "prior": prior,
                    "parid": parids
                })
                signals.append(newsignal)

        if incGWWavelet:
            bvary = [True] * 4
            pmin = [0, 0, 0, 0]
            pmax = [np.pi, 2*np.pi, np.pi, 1]
            pstart = [np.pi/2, np.pi, np.pi/2, 0.5]
            pwidth = [0.1, 0.1, 0.1, 0.1]
            prior = ['cos', 'uniform', 'uniform', 'uniform']
            parids = ['wavetheta', 'wavephi', 'wavepsi', 'waveeps']
            mu = [None] * 4
            sigma = [None] * 4
            for ww in range(nGWWavelets):
                bvary += [True] * 5
                pmin += [-10, -9, p.toas.min()/86400, 2, 0]
                pmax += [-5, -7, p.toas.max()/86400, 100, 2*np.pi]
                pstart += [-7, -8, (p.toas.max() + p.toas.min())/2/86400,
                         30, np.pi]
                pwidth += [0.1, 0.1, 10, 2, 0.1]
                prior += ['log', 'log', 'uniform', 'uniform', 'uniform']
                parids += ['waveAmp_'+str(ww), 'waveFreq_'+str(ww),
                         'waveT0_'+str(ww), 'waveQ_'+str(ww),
                         'wavePhase_'+str(ww)]
                mu += [None] * 5
                sigma += [None] * 5

            newsignal = OrderedDict({
                "stype": "gwwavelet",
                "corr": "gr",
                "pulsarind": -1,
                "mu": mu,
                "sigma": sigma,
                "bvary": bvary,
                "pmin": pmin,
                "pmax": pmax,
                "pwidth": pwidth,
                "pstart": pstart,
                "parid": parids,
                "prior": prior,
            })
            signals.append(newsignal)


        if incCW:
            if CWModel == 'standard':
                bvary = [True] * 8
                pmin = [0, 0, 7, 0.1, -9, 0, 0, 0]
                pmax = [np.pi, 2 * np.pi, 10, 3, -7, 2*np.pi, np.pi, np.pi]
                pstart = [np.pi / 2, np.pi / 2, 8, 1, -8, np.pi,
                          np.pi, np.pi / 2]
                pwidth = [0.1, 0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1]
                prior = ['cos', 'uniform', 'log', 'log',
                         'log', 'uniform', 'uniform', 'cos']
                parids = ['theta', 'phi', 'logmc', 'logd',
                          'logf', 'phase', 'pol', 'inc']
                mu = [None] * 8
                sigma = [None] * 8

            elif CWModel == 'upperLimit':
                bvary = [True] * 8
                pmin = [0, 0, 7, -17, -9, 0, 0, 0]
                pmax = [np.pi, 2 * np.pi, 10, -12, -7, 2*np.pi, np.pi, np.pi]
                pstart = [np.pi / 2, np.pi / 2, 8, -14, -8, np.pi,
                          np.pi, np.pi / 2]
                pwidth = [0.1, 0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1]
                prior = ['cos', 'uniform', 'log', 'uniform',
                         'log', 'uniform', 'uniform', 'cos']
                parids = ['theta', 'phi', 'logmc', 'logh',
                          'logf', 'phase', 'pol', 'inc']
                mu = [None] * 8
                sigma = [None] * 8

            elif CWModel == 'mass_ratio':
                bvary = [True] * 9
                pmin = [0, 0, 7, 0.1, -9, 0, 0, 0, -3]
                pmax = [np.pi, 2 * np.pi, 12, 3, -7, 2*np.pi, np.pi, np.pi, 0]
                pstart = [np.pi / 2, np.pi / 2, 8, 1, -8, np.pi, np.pi,
                          np.pi / 2, -1]
                pwidth = [0.1, 0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1, 0.1]
                prior = ['cos', 'uniform', 'log', 'log',
                         'log', 'uniform', 'uniform', 'cos', 'log']
                mu = [None] * 9
                sigma = [None] * 9
                parids = ['theta', 'phi', 'logmc', 'logd',
                          'logf', 'phase','pol', 'inc', 'logq']

            elif CWModel == 'free':
                bvary = [True] * 7
                pmin = [0, 0, -18, -9, 0, 0, 0]
                pmax = [np.pi, 2 * np.pi, -11, -7, 2*np.pi, np.pi, np.pi]
                pstart = [np.pi/2, np.pi/2, -15, -8, np.pi/2, np.pi/2, np.pi/2]
                pwidth = [0.1, 0.1, 0.1, 1e-4, 0.1, 0.1, 0.1]
                prior = ['cos', 'uniform', 'log', 'log', 'uniform', 'uniform',
                         'cos']
                mu = [None] * 7
                sigma = [None] * 7
                parids = ['theta', 'phi', 'logh', 'logf', 'phase',
                          'pol', 'inc']

            newsignal = OrderedDict({
                "stype": "cw",
                "model": CWModel,
                "corr": "gr",
                "pulsarind": -1,
                "mu": mu,
                "sigma": sigma,
                "bvary": bvary,
                "pmin": pmin,
                "pmax": pmax,
                "pwidth": pwidth,
                "pstart": pstart,
                "parid": parids,
                "prior": prior,
            })
            signals.append(newsignal)

        #if incCW and not CWupperLimit and not mass_ratio:
        #    bvary = [True] * 8
        #    pmin = [0, 0, 7, 0.1, -9, 0, 0, 0]
        #    pmax = [np.pi, 2 * np.pi, 10, 3, -7, 2*np.pi, np.pi, np.pi]
        #    pstart = [np.pi / 2, np.pi / 2, 8, 1, -8, np.pi, np.pi, np.pi / 2]
        #    pwidth = [0.1, 0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1]
        #    prior = ['cos', 'uniform', 'log', 'log',
        #             'log', 'uniform', 'uniform', 'cos']
        #    parids = ['theta', 'phi', 'logmc', 'logd',
        #              'logf', 'phase', 'pol', 'inc']
        #    mu = [None] * 8
        #    sigma = [None] * 8

        #    newsignal = OrderedDict({
        #        "stype": "cw",
        #        "corr": "gr",
        #        "pulsarind": -1,
        #        "mu": mu,
        #        "sigma": sigma,
        #        "bvary": bvary,
        #        "pmin": pmin,
        #        "pmax": pmax,
        #        "pwidth": pwidth,
        #        "pstart": pstart,
        #        "parid": parids,
        #        "prior": prior,
        #        "upperlimit": CWupperLimit,
        #        "mass_ratio": mass_ratio
        #    })
        #    signals.append(newsignal)

        #elif incCW and CWupperLimit and not mass_ratio:
        #    bvary = [True] * 8
        #    pmin = [0, 0, 7, -17, -9, 0, 0, 0]
        #    pmax = [np.pi, 2 * np.pi, 10, -12, -7, 2*np.pi, np.pi, np.pi]
        #    pstart = [
        #        np.pi / 2, np.pi / 2, 8, -14, -8, np.pi, np.pi, np.pi / 2]
        #    pwidth = [0.1, 0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1]
        #    prior = ['cos', 'uniform', 'log', 'uniform',
        #             'log', 'uniform', 'uniform', 'cos']
        #    parids = [
        #        'theta', 'phi', 'logmc', 'logh', 'logf', 'phase', 'pol', 'inc']

        #    newsignal = OrderedDict({
        #        "stype": "cw",
        #        "corr": "gr",
        #        "pulsarind": -1,
        #        "bvary": bvary,
        #        "pmin": pmin,
        #        "pmax": pmax,
        #        "pwidth": pwidth,
        #        "pstart": pstart,
        #        "parid": parids,
        #        "prior": prior,
        #        "upperlimit": CWupperLimit,
        #        "mass_ratio": mass_ratio
        #    })
        #    signals.append(newsignal)

        #if incCW and mass_ratio:
        #    bvary = [True] * 9
        #    pmin = [0, 0, 7, 0.1, -9, 0, 0, 0, -3]
        #    pmax = [np.pi, 2 * np.pi, 12, 3, -7, 2*np.pi, np.pi, np.pi, 0]
        #    pstart = [np.pi / 2, np.pi / 2, 8, 1, -8, np.pi, np.pi, np.pi / 2, -1]
        #    pwidth = [0.1, 0.1, 0.1, 0.1, 0.0001, 0.1, 0.1, 0.1, 0.1]
        #    prior = ['cos', 'uniform', 'log', 'log',
        #             'log', 'uniform', 'uniform', 'cos', 'log']
        #    mu = [None] * 9
        #    sigma = [None] * 9
        #    parids = ['theta', 'phi', 'logmc', 'logd',
        #              'logf', 'phase','pol', 'inc', 'logq']

        #    newsignal = OrderedDict({
        #        "stype": "cw",
        #        "corr": "gr",
        #        "pulsarind": -1,
        #        "bvary": bvary,
        #        "pmin": pmin,
        #        "pmax": pmax,
        #        "mu": mu,
        #        "sigma": sigma,
        #        "pwidth": pwidth,
        #        "pstart": pstart,
        #        "parid": parids,
        #        "prior": prior,
        #        "upperlimit": CWupperLimit,
        #        "mass_ratio": mass_ratio,
        #    })
        #    signals.append(newsignal)

        if incGWB:
            if gwbModel == 'spectrum':
                bvary = [True] * nfreqs
                pmin = [-18.0] * nfreqs
                pmax = [-8.0] * nfreqs
                pstart = [-10.0] * nfreqs
                pwidth = [0.1] * nfreqs
                prior = [GWspectrumPrior] * nfreqs
            elif gwbModel == 'powerlaw':
                bvary = [True, True, False]
                pmin = [-18.0, 1.02, 1.0e-11]
                pmax = [np.log10(4e-12), 6.98, 3.0e-9]
                pstart = [-15.0, 2.01, 1.0e-10]
                pwidth = [0.1, 0.1, 5.0e-11]
                prior = [GWAmpPrior, GWSiPrior, 'log']
            elif gwbModel == 'turnover':
                bvary = [True, True, True, True, False]
                pmin = [-18.0, 1.02, -9, 0.01, 0.2]
                pmax = [-11.0, 6.98, -7, 6.98, 5.0]
                pstart = [-15.0, 2.01, -8, 2.01, 0.5]
                pwidth = [0.1, 0.1, 0.1, 0.1, 0.1]
                prior = [GWAmpPrior, GWSiPrior, 'uniform', 'uniform', 'uniform']

            newsignal = OrderedDict({
                "stype": gwbModel,
                "corr": "gr",
                "pulsarind": -1,
                "bvary": bvary,
                "pmin": pmin,
                "pmax": pmax,
                "pwidth": pwidth,
                "pstart": pstart,
                "prior": prior
            })
            signals.append(newsignal)

        if incORF:
            ncoeff = int(npsr * (npsr - 1) / 2)
            bvary = [True] * ncoeff
            pmin = [0.0] * ncoeff
            pmax = [np.pi] * ncoeff
            pstart = [np.pi/2] * ncoeff
            pwidth = [0.1] * ncoeff
            prior = ['uniform'] * ncoeff
            parids = ['Phi_' + str(l) + str(m) for l in range(npsr)
                       for m in range(l + 1, npsr)]
            
            newsignal = OrderedDict({
                "stype": "ORF",
                "corr": "gr",
                "pulsarind": -1,
                "bvary": bvary,
                "pmin": pmin,
                "pmax": pmax,
                "pwidth": pwidth,
                "pstart": pstart,
                "parid": parids,
                "prior": prior,
            })
            signals.append(newsignal)

        if incGWBAni:
            if gwbModel == 'spectrum':
                ncoeff = (lmax + 1) ** 2 - 1
                bvary = [True] * (nfreqs + ncoeff)
                pmin = [-18.0] * nfreqs
                pmin += [-3.0] * (ncoeff)
                pmax = [-8.0] * nfreqs
                pmax += [3.0] * (ncoeff)
                pstart = [-17.0] * nfreqs
                pstart += [0.0] * (ncoeff)
                pwidth = [0.1] * nfreqs
                pwidth += [0.1] * (ncoeff)
                prior = [GWspectrumPrior] * nfreqs
                prior += ['uniform'] * (ncoeff)
                print nfreqs
                parids = ['rho_' + str(f) for f in range(nfreqs)]
                parids += ['c_' + str(l) + str(m) for l in range(lmax + 1)
                           for m in range(-l, l + 1) if l != 0]

            elif gwbModel == 'powerlaw':
                ncoeff = (lmax + 1) ** 2 - 1
                bvary = [True, True, False]
                bvary += [True] * (ncoeff)
                pmin = [-17.0, 1.02, 1.0e-11]
                pmin += [-3] * (ncoeff)
                pmax = [-11.0, 6.98, 3.0e-9]
                pmax += [3] * (ncoeff)
                pstart = [-15.0, 2.01, 1.0e-10]
                pstart += [0.0] * (ncoeff)
                pwidth = [0.1, 0.1, 5.0e-11]
                pwidth += [0.1] * (ncoeff)
                prior = [GWAmpPrior, GWSiPrior, 'log']
                prior += ['uniform'] * (ncoeff)
                parids = ['aGWB-Amplitude', 'aGWB-SpectralIndex', 'fL']
                parids += ['c_' + str(l) + str(m) for l in range(lmax + 1)
                           for m in range(-l, l + 1) if l != 0]
                print len(parids), np.sum(bvary)

            newsignal = OrderedDict({
                "stype": gwbModel,
                "corr": "gr_sph",
                "pulsarind": -1,
                "bvary": bvary,
                "pmin": pmin,
                "pmax": pmax,
                "pwidth": pwidth,
                "pstart": pstart,
                "parid": parids,
                "prior": prior,
                "lmax": lmax
            })
            signals.append(newsignal)

        # The list of signals
        modeldict = OrderedDict({
            "file version": 2014.02,
            "author": "PAL-makeModel",
            "numpulsars": len(self.psr),
            "pulsarnames": [self.psr[ii].name for ii in range(len(self.psr))],
            "numNoiseFreqs": [nfreqs for ii in range(len(self.psr))],
            "numDMFreqs": [ndmfreqs for ii in range(len(self.psr))],
            "numScatFreqs":nscatfreqs,
            "compression": compression,
            "logfrequencies": logf,
            "incDMX": incDMX,
            "orderFrequencyLines": orderFrequencyLines,
            "evalCompressionComplement": evalCompressionComplement,
            "likfunc": likfunc,
            "Tmax":Tmax,
            "redExtNf":redExtNf,
            "incRedExt":incRedExt,
            "signals": signals
        })

        return modeldict

    """
    Add a signal to the internal description data structures, based on a signal
    dictionary

    @param signal:  The signal dictionary we will add to the list
    @param index:   The index of the first par in the global par list
    @param Tmax:    The total time-baseline we use for this signal
    """

    def addSignal(self, signal, index=0, Tmax=None):
        # Assert that the necessary keys are present
        keys = ['pulsarind', 'stype', 'corr', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))

        # Determine the time baseline of the array of pulsars
        #if not 'Tmax' in signal and Tmax is None:
        #    Tstart = np.min(self.psr[0].toas)
        #    Tfinish = np.max(self.psr[0].toas)
        #    for p in self.psr:
        #        Tstart = np.min([np.min(p.toas), Tstart])
        #        Tfinish = np.max([np.max(p.toas), Tfinish])
        #    Tmax = Tfinish - Tstart

        # Adjust some basic details about the signal
        signal['Tmax'] = Tmax
        signal['parindex'] = index

        # Convert a couple of values
        signal['bvary'] = np.array(signal['bvary'], dtype=np.bool)
        signal['npars'] = np.sum(signal['bvary'])
        signal['ntotpars'] = len(signal['bvary'])
        signal['pmin'] = np.array(signal['pmin'])
        signal['pmax'] = np.array(signal['pmax'])
        signal['pwidth'] = np.array(signal['pwidth'])
        signal['pstart'] = np.array(signal['pstart'])

        # Add the signal
        if signal['stype'] == 'efac':
            # Efac
            self.addSignalEfac(signal)

        elif signal['stype'] == 'equad':
            # Equad
            self.addSignalEquad(signal)

        elif signal['stype'] == 'jitter':
            # Jitter
            self.addSignalJitter(signal)

        elif signal['stype'] == 'jitter_equad':
            # Jitter equad
            self.addSignalJitterEquad(signal)

        elif signal['stype'] == 'jitter_epoch':
            # Jitter by epoch
            self.addSignalJitterEpoch(signal)

        elif signal['stype'] == 'nongausscoeff':
            # non-gaussian coefficients
            self.addSignalNonGaussian(signal)
            self.haveNonGaussian = True

        elif signal['stype'] == 'pulsardistance':
            # pulsar distance parameters used with CW sigal
            self.ptasignals.append(signal)

        elif signal['stype'] == 'pulsarTerm':
            # pulsar phase and frequency
            self.ptasignals.append(signal)

        elif signal['stype'] in ['env_powerlaw', 'env_spectrum']:
            self.haveEnvelope = True
            for pp in self.psr:
                pp.kappa_env = np.zeros(len(pp.Ffreqs))
            self.ptasignals.append(signal)

        elif signal['stype'] in ['scatpowerlaw', 'scatspectrum']:
            self.haveScat = True
            for pp in self.psr:
                pp.kappa_scat = np.zeros(signal['nscatfreqs'] + 3)
            self.ptasignals.append(signal)

        elif signal['stype'] in ['powerlaw', 'spectrum', 'spectralModel',
                                 'powerlaw_band', 'turnover', 'ext_powerlaw',
                                 'ext_spectrum']:
            # Any time-correlated signal
            self.addSignalTimeCorrelated(signal)
            self.haveStochSources = True
            if signal['corr'] in ['gr_sph']:
                psr_locs = np.array([[p.phi[0], p.theta[0]] for p in self.psr])
                lmax = signal['lmax']
                self.harm_sky_vals = PALutils.SetupPriorSkyGrid(lmax)
                self.AniBasis = ani.CorrBasis(psr_locs, lmax)

            if signal['stype'] in ['ext_powerlaw', 'ext_spectrum']:
                for pp in self.psr:
                    pp.kappa_ext = np.zeros(len(pp.Fextfreqs))
                self.haveExt = True

        elif signal['stype'] in ['dmpowerlaw', 'dmspectrum', 'dmpowerlaw_band']:
            # A DM variation signal
            self.addSignalDMV(signal)
            self.haveStochSources = True

        elif signal['stype'] in ['redfouriermode', 'dmfouriermode', 'gwfouriermode']:
            # fourier amplitudes
            self.ptasignals.append(signal)
            self.haveDetSources = True

        elif signal['stype'] == 'cw':
            # a continuous GW signal
            self.ptasignals.append(signal)
            self.haveDetSources = True

        elif signal['stype'] == 'gwwavelet':
            # a GW wavelet signal
            self.ptasignals.append(signal)
            self.haveDetSources = True

        elif signal['stype'] == 'lineartimingmodel':
            # A Tempo2 linear timing model, except for (DM)QSD parameters
            self.addSignalTimingModel(signal)
            self.haveDetSources = True

        elif signal['stype'] == 'nonlineartimingmodel':
            # A Tempo2 timing model, except for (DM)QSD parameters
            # Note: libstempo must be installed
            self.addSignalTimingModel(signal, linear=False)
            self.haveDetSources = True

        elif signal['stype'] == 'frequencyline':
            # Single free-floating frequency line
            psrSingleFreqs = self.getNumberOfSignals(stype='frequencyline',
                                                     corr='single')
            signal['npsrfreqindex'] = psrSingleFreqs[signal['pulsarind']]
            self.addSignalFrequencyLine(signal)
            self.haveFrequencyLines = True

        elif signal['stype'] == 'dmfrequencyline':
            # Single free-floating frequency line
            psrSingleFreqs = self.getNumberOfSignals(stype='dmfrequencyline',
                                                     corr='single')
            signal['npsrfreqindex'] = psrSingleFreqs[signal['pulsarind']]
            self.addSignalFrequencyLine(signal)
            self.haveFrequencyLines = True

        elif signal['stype'] == 'dmshapeletmarg':
            self.ptasignals.append(signal)

        elif signal['stype'] in ['DMXconstantKernel', 'DMXseKernel']:
            self.ptasignals.append(signal)

        elif signal['stype'] == 'dmshapelet':
            self.ptasignals.append(signal)
            self.haveDetSources = True

        else:
            # Some other unknown signal
            self.ptasignals.append(signal)

    """
    Add an EFAC signal

    Required keys in signal
    @param psrind:      Index of the pulsar this efac applies to
    @param index:       Index of first parameter in total parameters array
    @param flagname:    Name of the flag this efac applies to (field-name)
    @param flagvalue:   Value of the flag this efac applies to (e.g. CPSR2)
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters

    """

    def addSignalEfac(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in efac signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))

        signal['Nvec'] = self.psr[signal['pulsarind']].toaerrs ** 2

        if signal['flagname'] != 'pulsarname':
            # This efac only applies to some TOAs, not all of 'm
            ind = np.array(self.psr[signal['pulsarind']].flags) != signal[
                'flagvalue']
            signal['Nvec'][ind] = 0.0

        self.ptasignals.append(signal.copy())

    """
    Add an Jitter signal (can be split up based on backend like efac)

    Required keys in signal
    @param psrind:      Index of the pulsar this efac applies to
    @param index:       Index of first parameter in total parameters array
    @param flagname:    Name of the flag this efac applies to (field-name)
    @param flagvalue:   Value of the flag this efac applies to (e.g. CPSR2)
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters

    """

    def addSignalJitter(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in jitter signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))

        # eq 5 from cordes and shannon measurement model paper
        Wims = 0.1 * self.psr[signal['pulsarind']].period * 1e3
        mI = 1
        if self.likfunc == 'mark2':
            N6 = self.psr[signal['pulsarind']].avetobs / \
                self.psr[signal['pulsarind']].period / 1e6
            sigmaJ = 0.28e-6 * Wims * 1 / \
                np.sqrt(N6) * np.sqrt((1 + mI ** 2) / 2)
            signal['Jvec'] = sigmaJ ** 2
        else:
            N6 = self.psr[signal['pulsarind']].tobsflags / \
                self.psr[signal['pulsarind']].period / 1e6
            sigmaJ = 0.28e-6 * Wims * 1 / \
                np.sqrt(N6) * np.sqrt((1 + mI ** 2) / 2)
            signal['Nvec'] = sigmaJ ** 2

        if signal['flagname'] != 'pulsarname':
            # This jitter only applies to some average TOAs, not all of 'm
            if self.likfunc in ['mark2', 'mark6']:
                ind = np.array(self.psr[signal['pulsarind']].aveflags) != signal[
                    'flagvalue']
                signal['Jvec'][ind] = 0.0
            else:
                ind = np.array(self.psr[signal['pulsarind']].flags) != signal[
                    'flagvalue']
                signal['Nvec'][ind] = 0.0

        self.ptasignals.append(signal.copy())

    """
    Add an Jitter by Epoch signal (one free jitter parameter per epoch)

    Required keys in signal
    @param psrind:      Index of the pulsar this efac applies to
    @param index:       Index of first parameter in total parameters array
    @param flagname:    Name of the flag this efac applies to (field-name)
    @param flagvalue:   Value of the flag this efac applies to (e.g. CPSR2)
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters

    """

    def addSignalJitterEpoch(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in jitter equad signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))

        self.ptasignals.append(signal.copy())

    """
    Add an Jitter Equad signal (can be split up based on backend like efac)

    Required keys in signal
    @param psrind:      Index of the pulsar this efac applies to
    @param index:       Index of first parameter in total parameters array
    @param flagname:    Name of the flag this efac applies to (field-name)
    @param flagvalue:   Value of the flag this efac applies to (e.g. CPSR2)
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters

    """

    def addSignalJitterEquad(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in jitter equad signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))

        # This is the 'jitter' that we have used before
        signal['Jvec'] = np.ones(len(self.psr[signal['pulsarind']].avetoas))

        if signal['flagname'] != 'pulsarname':
            # This jitter only applies to some average TOAs, not all of 'm
            ind = np.array(self.psr[signal['pulsarind']].aveflags) != signal[
                'flagvalue']
            signal['Jvec'][ind] = 0.0

        self.ptasignals.append(signal.copy())

    """
    Add an EQUAD signal

    Required keys in signal
    @param stype:       Either or 'equad'
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param flagname:    Name of the flag this efac applies to (field-name)
    @param flagvalue:   Value of the flag this efac applies to (e.g. CPSR2)
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    """

    def addSignalEquad(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in equad signal. Keys: {0}. \
                             Required: {1}".format(signal.keys(), keys))

        signal['Nvec'] = np.ones(len(self.psr[signal['pulsarind']].toaerrs))

        if signal['flagname'] != 'pulsarname':
            # This equad only applies to some TOAs, not all of 'm
            ind = np.array(self.psr[signal['pulsarind']].flags) != signal[
                'flagvalue']
            signal['Nvec'][ind] = 0.0

        self.ptasignals.append(signal.copy())

    """
    Add a single frequency line signal

    Required keys in signal
    @param stype:       Either 'frequencyline' or 'dmfrequencyline'
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param freqindex:   If there are several of these sources, which is this?
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    """

    def addSignalFrequencyLine(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in frequency line signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))

        self.ptasignals.append(signal.copy())

    """
    Add non gaussian coefficients

    Required keys in signal
    @param stype:       Either 'frequencyline' or 'dmfrequencyline'
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param freqindex:   If there are several of these sources, which is this?
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    """

    def addSignalNonGaussian(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in non gaussian coeffieicnt signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))

        self.ptasignals.append(signal.copy())

    """
    Add some time-correlated signal

    Required keys in signal
    @param stype:       Either 'spectrum', 'powerlaw', or 'spectralModel'
    @param corr:        Either 'single', 'uniform', 'dipole', 'gr', ...
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param Tmax         Time baseline of the entire experiment
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    @param lAniGWB:     In case of an anisotropic GWB, this sets the order of
                        anisotropy (default=2, also for all other signals)
    """

    def addSignalTimeCorrelated(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex', 'Tmax']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in signal. Keys: {0}. \
                             Required: {1}".format(signal.keys(), keys))

        if signal['corr'] == 'gr':
            # Correlated with the Hellings \& Downs matrix
            signal['corrmat'] = PALutils.computeORFMatrix(self.psr) / 2
        elif signal['corr'] == 'uniform':
            # Uniformly correlated (Clock signal)
            signal['corrmat'] = np.ones((len(self.psr), len(self.psr)))

        if signal['corr'] != 'single':
            # Also fill the Ffreqs array, since we are dealing with
            # correlations
            numfreqs = np.array([len(self.psr[ii].Ffreqs)
                                 for ii in range(len(self.psr))])
            ind = np.argmax(numfreqs)
            signal['Ffreqs'] = self.psr[ind].Ffreqs.copy()

        self.ptasignals.append(signal.copy())

    """
    Add some DM variation signal

    Required keys in signal
    @param stype:       Either 'spectrum', 'powerlaw', or 'spectralModel'
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param Tmax         Time baseline of the entire experiment
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    """

    def addSignalDMV(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex', 'Tmax']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in DMV signal. Keys: {0}. \
                             Required: {1}".format(signal.keys(), keys))

        self.ptasignals.append(signal.copy())

    """
    Add a signal that represents a numerical tempo2 timing model

    Required keys in signal
    @param stype:       Basically always 'lineartimingmodel' (TODO: include nonlinear)
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    @param parid:       The identifiers (as used in par-file) that identify
                        which parameters are included
    """

    def addSignalTimingModel(self, signal, linear=True):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary', 'parid',
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in TimingModel signal. Keys: {0}. Required: {1}".format(
                signal.keys(), keys))

        # Assert that this signal applies to a pulsar
        if signal['pulsarind'] < 0 or signal['pulsarind'] >= len(self.psr):
            raise ValueError(
                "ERROR: timingmodel signal applied to non-pulsar ({0})".format(signal['pulsarind']))

        # Check that the parameters included here are also present in the design
        # matrix
        for ii, parid in enumerate(signal['parid']):
            if not parid in self.psr[signal['pulsarind']].ptmdescription:
                raise ValueError(
                    "ERROR: timingmodel signal contains non-valid parameter id")

        # If this is a non-linear signal, make sure to initialise the libstempo
        # object
        # if linear == False:
        #    self.psr[signal['pulsarind']].initLibsTempoObject()

        self.ptasignals.append(signal.copy())

    # TODO: add CW signal

    """
    Re-calculate the number of varying parameters per signal, and the number of
    dimensions in total.
    """

    def setDimensions(self):
        self.dimensions = 0
        for sig in self.ptasignals:
            sig['npars'] = np.sum(sig['bvary'])
            self.dimensions += sig['npars']

    """
    Before being able to run the likelihood, we need to initialise the prior

    """

    def initPrior(self):
        self.setDimensions()

        self.pmin = np.zeros(self.dimensions)
        self.pmax = np.zeros(self.dimensions)
        self.pstart = np.zeros(self.dimensions)
        self.pwidth = np.zeros(self.dimensions)

        index = 0
        for sig in self.ptasignals:
            for ii in range(sig['ntotpars']):
                if sig['bvary'][ii]:
                    self.pmin[index] = sig['pmin'][ii]
                    self.pmax[index] = sig['pmax'][ii]
                    self.pwidth[index] = sig['pwidth'][ii]
                    self.pstart[index] = sig['pstart'][ii]
                    index += 1

    """
    Find the number of signals per pulsar matching some criteria, given a list
    of signal dictionaries. Main use is, for instance, to find the number of
    free frequency lines per pulsar given the signal model dictionary.

    @param signals: Dictionary of all signals
    @param stype:   The signal type that must be matched
    @param corr:    Signal correlation that must be matched
    """

    def getNumberOfSignalsFromDict(
            self, signals, stype='powerlaw', corr='single'):
        psrSignals = np.zeros(len(self.psr), dtype=np.int)

        for ii, signal in enumerate(signals):
            if signal['stype'] == stype and signal['corr'] == corr:
                if signal['pulsarind'] == -1:
                    psrSignals[:] += 1
                else:
                    psrSignals[signal['pulsarind']] += 1

        return psrSignals

    """
    Find the number of signals per pulsar matching some criteria in the current
    signal list.

    @param stype:   The signal type that must be matched
    @param corr:    Signal correlation that must be matched
    """

    def getNumberOfSignals(self, stype='powerlaw', corr='single'):
        return self.getNumberOfSignalsFromDict(self.ptasignals, stype, corr)

    """
    Find the signal numbers of a certain type and correlation

    @param signals: Dictionary of all signals
    @param stype:   The signal type that must be matched
    @param corr:    Signal correlation that must be matched
    @param psrind:  Pulsar index that must be matched (-2 means all)

    @return:        Index array with signals that qualify
    """

    def getSignalNumbersFromDict(self, signals, stype='powerlaw',
                                 corr='single', psrind=-2):
        signalNumbers = []

        for ii, signal in enumerate(signals):
            if signal['stype'] == stype and signal['corr'] == corr:
                if psrind == -2:
                    signalNumbers.append(ii)
                elif signal['pulsarind'] == psrind:
                    signalNumbers.append(ii)

        return np.array(signalNumbers, dtype=np.int)

    """
    Initialise the model.
    @param numNoiseFreqs:       Dictionary with the full model
    @param fromFile:            Try to read the necessary Auxiliaries quantities
                                from the HDF5 file
    @param verbose:             Give some extra information about progress
    """

    def initModel(self, fullmodel, fromFile=False, write=True,
                  verbose=False, memsave=True):
        numNoiseFreqs = fullmodel['numNoiseFreqs']
        numDMFreqs = fullmodel['numDMFreqs']
        compression = fullmodel['compression']
        evalCompressionComplement = fullmodel['evalCompressionComplement']
        orderFrequencyLines = fullmodel['orderFrequencyLines']
        likfunc = fullmodel['likfunc']
        signals = fullmodel['signals']
        incDMX = fullmodel['incDMX']
        dTmax = fullmodel['Tmax']
        nfredExt = fullmodel['redExtNf']
        incRedExt = fullmodel['incRedExt']
        numScatFreqs = fullmodel['numScatFreqs']

        if len(self.psr) < 1:
            raise IOError("No pulsars loaded")

        if fullmodel['numpulsars'] != len(self.psr):
            raise IOError("Model does not have the right number of pulsars")

        # if not self.checkSignalDictionary(signals):
        #    raise IOError, "Signal dictionary not properly defined"

        # Details about the likelihood function
        self.likfunc = likfunc
        self.orderFrequencyLines = orderFrequencyLines

        # Determine the time baseline of the array of pulsars
        Tstart = np.min(self.psr[0].toas)
        Tfinish = np.max(self.psr[0].toas)
        for p in self.psr:
            Tstart = np.min([np.min(p.toas), Tstart])
            Tfinish = np.max([np.max(p.toas), Tfinish])
        Tmax = Tfinish - Tstart
        #Tmax = 1/1.33950638e-09
        self.Tref = Tstart

        # print 'WARNING: Using seperate Tmax for each pulsar'
        if dTmax is None:
            print 'WARNING: Using seperate Tmax for each pulsar'
            for p in self.psr:
                p.Tmax = p.toas.max() - p.toas.min()
                #p.Tmax = self.Tmax
        elif dTmax == 0:
            print 'Using longest timespan of {0} yr for Tmax'.format(Tmax/3.16e7)
            for p in self.psr:
                p.Tmax = Tmax
        else:
            print 'Using {0} yr for Tmax'.format(dTmax / 3.16e7)
            for p in self.psr:
                p.Tmax = dTmax

        # If the compressionComplement is defined, overwrite the default
        if evalCompressionComplement != 'None':
            self.evallikcomp = evalCompressionComplement
            self.compression = compression
        elif compression == 'None':
            self.evallikcomp = False
            self.compression = compression
        else:
            self.evallikcomp = True
            self.compression = compression

        # Find out how many single-frequency modes there are
        numSingleFreqs = self.getNumberOfSignalsFromDict(signals,
                                                         stype='frequencyline',
                                                         corr='single')
        numSingleDMFreqs = self.getNumberOfSignalsFromDict(signals,
                                                           stype='dmfrequencyline',
                                                           corr='single')

        # Find out how many efac signals there are, and translate that to a
        # separateEfacs boolean array (for two-component noise analysis)
        numEfacs = self.getNumberOfSignalsFromDict(signals,
                                                   stype='efac', corr='single')
        numEquads = self.getNumberOfSignalsFromDict(signals,
                                                    stype='equad', corr='single')
        numJitter = self.getNumberOfSignalsFromDict(signals,
                                                    stype='jitter_equad', corr='single')
        numRedBand = self.getNumberOfSignalsFromDict(signals,
                                                     stype='powerlaw_band', corr='single')
        numDMBand = self.getNumberOfSignalsFromDict(signals,
                                                    stype='dmpowerlaw_band', corr='single')
        incJitter = np.array(numJitter) > 0
        incRedBand = np.array(numRedBand) > 0
        incDMBand = np.array(numDMBand) > 0
        print incRedExt
        if self.likfunc == 'mark9':
            incJitter = [False] * numJitter
        separateEfacs = np.logical_or(numEfacs > 1, numEquads > 1)

        # Modify design matrices, and create pulsar Auxiliary quantities
        for pindex, p in enumerate(self.psr):
            # If we model DM variations, we will need to include QSD
            # marginalisation for DM. Modify design matrix accordingly
            # if dmModel[pindex] != 'None':
            # if numDMFreqs[pindex] > 0:
            #    p.addDMQuadratic()

            # get timing model parameters
            tmpars = None
            linsigind = self.getSignalNumbersFromDict(signals,
                                                      stype='lineartimingmodel', psrind=pindex)
            nlsigind = self.getSignalNumbersFromDict(signals,
                                                     stype='nonlineartimingmodel', psrind=pindex)

            if len(linsigind) + len(nlsigind) > 0:

                tmpars = []    # All the timing model parameters of this pulsar
                for ss in np.append(linsigind, nlsigind):
                    tmpars += signals[ss]['parid']

            # We'll try to read the necessary quantities from the HDF5 file
            try:
                if not fromFile:
                    raise Exception(
                        'Requested to re-create the Auxiliaries')
                # Read Auxiliaries
                if verbose:
                    print "Reading Auxiliaries for {0}".format(p.name)
                p.readPulsarAuxiliaries(self.h5df, p.Tmax, numNoiseFreqs[pindex],
                                        numDMFreqs[pindex], ~separateEfacs[
                                            pindex],
                                        nSingleFreqs=numSingleFreqs[pindex],
                                        nSingleDMFreqs=numSingleDMFreqs[
                                            pindex],
                                        likfunc=likfunc, compression=compression,
                                        memsave=memsave, tmpars=tmpars)
            except (Exception, ValueError, KeyError, IOError, RuntimeError) as err:
                # Create the Auxiliaries ourselves

                # For every pulsar, construct the auxiliary quantities like the Fourier
                # design matrix etc
                if verbose:
                    print str(err)
                    print "Creating Auxiliaries for {0}".format(p.name)
                p.createPulsarAuxiliaries(self.h5df, p.Tmax, numNoiseFreqs[pindex],
                                          numDMFreqs[pindex], ~separateEfacs[
                                              pindex],
                                          nSingleFreqs=numSingleFreqs[pindex],
                                          nSingleDMFreqs=numSingleDMFreqs[
                                              pindex],
                                          likfunc=likfunc, compression=compression,
                                          write=write, tmpars=tmpars, memsave=memsave,
                                          incJitter=incJitter[
                                              pindex], incDMX=incDMX,
                                          incRedBand=incRedBand[pindex],
                                          incDMBand=incDMBand[pindex],
                                         incRedExt=incRedExt,
                                         nfredExt=nfredExt)

        # Initialise the ptasignal objects
        self.ptasignals = []
        index = 0
        for ii, signal in enumerate(signals):
            self.addSignal(signal, index, p.Tmax)
            index += self.ptasignals[-1]['npars']

        self.initPrior()
        self.allocateLikAuxiliaries()
        #self.pardes = self.getModelParameterList()

    """
    Get a list of all the model parameters, the parameter indices, and the
    descriptions

    TODO: insert these descriptions in the signal dictionaries
    """

    def getModelParameterList(self):
        pardes = []

        for ii, sig in enumerate(self.ptasignals):
            pindex = 0
            for jj in range(sig['ntotpars']):
                if sig['bvary'][jj]:
                    # This parameter is in the mcmc
                    index = sig['parindex'] + pindex
                    pindex += 1
                else:
                    index = -1

                psrindex = sig['pulsarind']
                if sig['stype'] == 'efac':
                    flagname = sig['flagname']
                    flagvalue = 'efac_' + sig['flagvalue']

                elif sig['stype'] == 'equad':
                    flagname = sig['flagname']
                    flagvalue = 'equad_' + sig['flagvalue']

                elif sig['stype'] == 'jitter':
                    flagname = sig['flagname']
                    flagvalue = 'jitter_' + sig['flagvalue']

                elif sig['stype'] == 'jitter_equad':
                    flagname = sig['flagname']
                    flagvalue = 'jitter_q_' + sig['flagvalue']

                elif sig['stype'] == 'jitter_epoch':
                    flagname = sig['flagname']
                    flagvalue = 'jitter_p_' + str(jj)

                elif sig['stype'] == 'nongausscoeff':
                    flagname = sig['flagname']
                    flagvalue = 'alpha_' + str(jj + 1)

                elif sig['stype'] == 'pulsardistance':
                    flagname = sig['flagname']
                    flagvalue = 'pdist_' + sig['flagvalue']

                elif sig['stype'] == 'spectrum' and sig['corr'] != 'gr_sph':
                    flagname = 'frequency'
                    #flagvalue = 'rho' + str(jj)
                    if sig['corr'] == 'single':
                        flagvalue = 'red_' + \
                            str(self.psr[psrindex].Ffreqs[2 * jj])
                    elif sig['corr'] == 'gr':
                        flagvalue = 'gwb_' + \
                            str(self.psr[psrindex].Ffreqs[2 * jj])

                elif sig['stype'] == 'dmspectrum':
                    flagname = 'dmfrequency'
                    flagvalue = 'dm_' + \
                        str(self.psr[psrindex].Fdmfreqs[2 * jj])

                elif sig['stype'] == 'cw':
                    flagname = 'cw'
                    flagvalue = sig['parid'][jj]

                elif sig['stype'] == 'gwwavelet':
                    flagname = 'gwwavelet'
                    flagvalue = sig['parid'][jj]

                elif sig['stype'] == 'pulsarTerm':
                    flagname = 'pulsarTerm'
                    flagvalue = sig['parid'][jj]

                elif sig['stype'] == 'ORF':
                    flagname = 'ORF'
                    flagvalue = sig['parid'][jj]

                elif sig['stype'] == 'powerlaw_band':
                    flagname = 'powerlaw_band'
                    flagvalue = sig['parids'][jj]

                elif sig['stype'] == 'dmpowerlaw_band':
                    flagname = 'dmpowerlaw_band'
                    flagvalue = sig['parids'][jj]

                elif sig['stype'] == 'powerlaw' and sig['corr'] != 'gr_sph':
                    flagname = 'powerlaw'

                    if sig['corr'] == 'gr':
                        flagvalue = ['GWB-Amplitude', 'GWB-spectral-index',
                                     'low-frequency-cutoff'][jj]
                    elif sig['corr'] == 'uniform':
                        flagvalue = ['CLK-Amplitude', 'CLK-spectral-index',
                                     'low-frequency-cutoff'][jj]
                    elif sig['corr'] == 'dipole':
                        flagvalue = ['DIP-Amplitude', 'DIP-spectral-index',
                                     'low-frequency-cutoff'][jj]
                    else:
                        flagvalue = ['RN-Amplitude', 'RN-spectral-index',
                                     'low-frequency-cutoff'][jj]

                elif sig['stype'] == 'turnover' and sig['corr'] == 'gr':
                    flagvalue = ['GWB-Amplitude', 'GWB-spectral-index',
                                 'GWB-f0', 'GWB-kappa', 'GWB-beta'][jj]

                elif sig['stype'] == 'dmpowerlaw':
                    flagname = 'dmpowerlaw'
                    flagvalue = ['DM-Amplitude', 'DM-spectral-index',
                                 'low-frequency-cutoff'][jj]

                elif sig['stype'] == 'dmse':
                    flagname = 'dmse'
                    flagvalue = ['DM-Amplitude', 'DM-tau'][jj]

                elif sig['stype'] == 'spectralModel':
                    flagname = 'spectralModel'
                    flagvalue = ['SM-Amplitude', 'SM-spectral-index',
                                 'SM-corner-frequency'][jj]

                elif sig['stype'] == 'frequencyline':
                    flagname = 'frequencyline'
                    flagvalue = ['Line-Freq', 'Line-Ampl'][jj]

                elif sig['stype'] == 'dmfrequencyline':
                    flagname = 'dmfrequencyline'
                    flagvalue = ['DM-Line-Freq', 'DM-Line-Ampl'][jj]

                elif sig['stype'] == 'lineartimingmodel' or \
                        sig['stype'] == 'nonlineartimingmodel':
                    flagname = sig['stype']
                    flagvalue = sig['parid'][jj]

                elif sig['stype'] in ['env_powerlaw', 'env_spectrum']:
                    flagname = sig['stype']
                    flagvalue = sig['parids'][jj]

                elif sig['stype'] in ['scatpowerlaw', 'scatspectrum']:
                    flagname = sig['stype']
                    flagvalue = sig['parids'][jj]

                elif sig['stype'] in ['ext_powerlaw', 'ext_spectrum']:
                    flagname = sig['stype']
                    flagvalue = sig['parids'][jj]

                elif sig['stype'] == 'dmshapeletmarg':
                    flagname = sig['stype']
                    flagvalue = sig['parid'][jj]

                elif sig['stype'] == 'dmshapelet':
                    flagname = sig['stype']
                    flagvalue = sig['parid'][jj]

                elif sig['corr'] == 'gr_sph':
                    if sig['bvary'][jj]:
                        flagname = sig['stype']
                        flagvalue = sig['parid'][jj]

                elif sig['stype'] in ['DMXconstantKernel', 'DMXseKernel']:
                    flagname = sig['stype']
                    flagvalue = sig['parid'][jj]

                elif sig['stype'] == 'redfouriermode':
                    flagname = 'redfourier'
                    if jj % 2 == 0:
                        flagvalue = 'ared_s_' + \
                            str(self.psr[psrindex].Ffreqs[jj])
                    else:
                        flagvalue = 'ared_c_' + \
                            str(self.psr[psrindex].Ffreqs[jj])

                elif sig['stype'] == 'dmfouriermode':
                    flagname = 'dmfourier'
                    flagvalue = sig['flagvalue'] + '_adm_' + \
                        str(self.psr[psrindex].Fdmfreqs[jj])

                elif sig['stype'] == 'gwfouriermode':
                    flagname = 'gwfourier'
                    flagvalue = sig['flagvalue'] + \
                        '_agw_' + str(self.gwfreqs[jj])

                else:
                    flagname = 'none'
                    flagvalue = 'none'

                pardes.append(
                    {'index': index, 'pulsar': psrindex, 'sigindex': ii,
                     'sigtype': sig['stype'], 'correlation': sig['corr'],
                     'name': flagname, 'id': flagvalue})

        return pardes

    """
    Determine intial parameters drawn from prior ranges

    """

    def initParameters(self, startEfacAtOne=True, fixpstart=False):

        p0 = []
        for ct, sig in enumerate(self.ptasignals):
            if np.any(sig['bvary']):
                for min, max, pstart, pwidth in zip(sig['pmin'][sig['bvary']],
                                                    sig['pmax'][sig['bvary']], sig[
                                                        'pstart'][sig['bvary']],
                                                    sig['pwidth'][sig['bvary']]):
                    if startEfacAtOne and sig['stype'] == 'efac':
                        p0.append(1)
                    else:
                        if fixpstart:
                            p0.append(np.double(pstart))
                        else:
                            p0.append(min + np.random.rand() * (max - min))
                            #p0.append(pstart + np.random.randn()*pwidth*10)

        return np.array(p0)

    """
    Determine intial covariance matrix for jumps

    """

    def initJumpCovariance(self):

        cov_diag = []
        for ct, sig in enumerate(self.ptasignals):
            if np.any(sig['bvary']):
                for step in sig['pwidth'][sig['bvary']]:
                    cov_diag.append((step) ** 2)

        cov = np.diag(cov_diag)

        # if we have timing model parameters use fisher matrix
        for ct, sig in enumerate(self.ptasignals):
            if 'timingmodel' in sig['stype']:
                parinds = np.arange(
                    sig['parindex'], sig['parindex'] + sig['npars'])
                pulsarind = sig['pulsarind']
                cov[parinds[0]:parinds[-1] + 1, parinds[0]:parinds[-1] + 1] = \
                    self.psr[pulsarind].fisher
                # for ct1, ii in enumerate(parinds):
                #    for ct2, jj in enumerate(parinds):
                #        cov[ii,jj] = self.psr[pulsarind].fisher[ct1,ct2]

        return cov

    """
    Allocate memory for the ptaLikelihood attribute matrices that we'll need in
    the likelihood function.  This function does not perform any calculations,
    although it does initialise the 'counter' integer arrays like npf and npgs.
    """

    def allocateLikAuxiliaries(self):
        # First figure out how large we have to make the arrays
        self.npsr = len(self.psr)
        self.npf = np.zeros(self.npsr, dtype=np.int)
        self.npftot = np.zeros(self.npsr, dtype=np.int)
        self.npu = np.zeros(self.npsr, dtype=np.int)
        self.npff = np.zeros(self.npsr, dtype=np.int)
        self.npfdm = np.zeros(self.npsr, dtype=np.int)
        self.npffdm = np.zeros(self.npsr, dtype=np.int)
        self.npobs = np.zeros(self.npsr, dtype=np.int)
        self.npgs = np.zeros(self.npsr, dtype=np.int)
        self.npgos = np.zeros(self.npsr, dtype=np.int)
        self.gradient = np.zeros(self.dimensions)
        self.hessian = np.zeros((self.dimensions, self.dimensions))
        self.ntmpars = 0
        nphiTmat = 0
        self.logdet_Sigma = 0

        for ii, p in enumerate(self.psr):

            # number of red and DM frequencies
            self.npf[ii] = len(p.Ffreqs)
            self.npfdm[ii] = len(p.Fdmfreqs)
            self.npftot[ii] = self.npf[ii] + self.npfdm[ii] + \
                p.nSingleFreqs * 2 + p.nSingleDMFreqs * 2
            self.ntmpars += len(p.ptmdescription)

            if self.likfunc == 'mark6' or self.likfunc == 'mark8' or \
                self.likfunc == 'mark9' or self.likfunc == 'mark10':
                p.Ttmat = p.Tmat.copy()
                if p.nSingleFreqs != 0:
                    p.Ttmat = np.append(p.Ttmat, np.zeros((p.Tmat.shape[0], 
                                                           p.nSingleFreqs * 2)),
                                       axis=1)
                if self.haveEnvelope:
                    p.Ttmat = np.append(p.Ttmat, np.zeros((p.Tmat.shape[0], self.npf[ii])),
                                        axis=1)
                    nphiTmat += self.npf[ii]

                if self.haveScat:
                    p.Ttmat = np.append(p.Ttmat, np.zeros((
                        p.Tmat.shape[0], len(p.kappa_scat))), axis=1)
                    nphiTmat += len(p.kappa_scat) 

                nphiTmat += p.Tmat.shape[1] + p.nSingleFreqs * 2 + p.nSingleDMFreqs * 2 + \
                    p.ndmEventCoeffs

            # noise vectors
            p.Nvec = np.zeros(len(p.toas))
            p.Nwvec = np.zeros(p.nbasis)
            p.Nwovec = np.zeros(p.nobasis)

        # number of GW frequencies
        self.ngwf = np.max(self.npf)
        self.gwfreqs = self.psr[np.argmax(self.npf)].Ffreqs
        nftot = np.max(self.npftot)
        if self.likfunc == 'mark6' or self.likfunc == 'mark9' or self.likfunc == 'mark10':
            self.Phiinv = np.zeros((nphiTmat, nphiTmat))
            self.TNT = np.zeros((nphiTmat, nphiTmat))
            self.Phi = np.zeros((nphiTmat, nphiTmat))
            self.Sigma = np.zeros((nphiTmat, nphiTmat))
        else:
            self.Phiinv = np.zeros((nftot * self.npsr, nftot * self.npsr))
            self.Phi = np.zeros((nftot * self.npsr, nftot * self.npsr))
            self.Sigma = np.zeros((nftot * self.npsr, nftot * self.npsr))



    """
    Fill in vector or Fourier amplitudes for all sources

    TODO: clean this up a bit

    """

    def setFourierAmplitudes(self, parameters, returnIndices=False):

        # loop over signals
        ared, adm, agw, indices = [], [], [], []
        for ss, sig in enumerate(self.ptasignals):
            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind + npars]

            if sig['stype'] == 'redfouriermode':
                ared.append(sparameters)
                indices.append(np.arange(parind, parind + npars))

            if sig['stype'] == 'dmfouriermode':
                adm.append(sparameters)
                indices.append(np.arange(parind, parind + npars))

            if sig['stype'] == 'gwfouriermode':
                agw.append(sparameters)
                indices.append(np.arange(parind, parind + npars))

        # loop over all pulsars and fill in total a array
        aarray = []
        for ct, p in enumerate(self.psr):

            if len(ared) != 0 and len(adm) != 0:
                atot = np.concatenate((ared[ct], adm[ct]))
            elif len(ared) != 0 and len(adm) == 0:
                atot = ared[ct]
            elif len(ared) == 0 and len(adm) != 0:
                atot = adm[ct]

            # GW piece
            if len(adm) > 0:
                gwamp = np.concatenate((agw[ct], np.zeros(len(adm[ct]))))
            elif len(agw) == 0:
                gwamp = 0
            else:
                gwamp = agw[ct]

            # append to diagonal elements
            if len(atot) > 0:
                aarray.append(atot + gwamp)
            else:
                aarrayappend(gwamp)

        if returnIndices:
            ret = (np.array(aarray).flatten(), np.array(indices).flatten())
        else:
            ret = np.array(aarray).flatten()

        return ret


    def computeAniORF(self, clms):
        """
        Construct anisitropic ORF basis functions

        """

        ncoeff = len(clms)
        ret = np.zeros((self.npsr, self.npsr))
        for ii in range(ncoeff):
            ret += clms[ii] * self.AniBasis[ii]

        return ret

    def computeSphORF(self, phi_corr):
        """
        Compute ORF using spherical basis?
        """
        upper_triang = np.zeros((self.npsr, self.npsr))
        phi_els = np.array([[0.0]*ii for ii in range(1, self.npsr)])
        k = 0
        for ii in range(len(phi_els)):
            for jj in range(len(phi_els[ii])):
                phi_els[ii][jj] = phi_corr[k]
                k += 1

        upper_triang[0, 0] = 1.0
        for ii in range(1, upper_triang.shape[1]):
            upper_triang[0, ii] = np.cos(phi_els[ii-1][0])
            upper_triang[ii,ii] = np.prod(np.sin(phi_els[ii-1]))

            for jj in range(ii+1, upper_triang.shape[1]):
                upper_triang[ii, jj] = np.cos(
                    phi_els[jj-1][ii]) * \
                    np.prod(np.sin(np.array(phi_els[jj-1])[0:ii]))   

        return np.dot(upper_triang.T, upper_triang) 


    """
    Construct the gradient of various parameters for use
    in HMC and MALA.

    Does not include GWB correlations yet

    TODO: make more robust, very specific at the moment
    """

    def constructGradients(self, parameters, incCorrelations=False):

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations)

        # frequency lines
        # self.updateSpectralLines(parameters)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # first do fourier modes as they are a bit different
        a, ind = self.setFourierAmplitudes(parameters, returnIndices=True)
        Phiinva = np.diag(self.Phiinv) * a
        for ct, p in enumerate(self.psr):

            # F^T N^{-1} (dt-Fa-Me)
            if ct == 0:
                d = np.dot(p.Ftot.T, p.detresiduals / p.Nvec)
            else:
                d = np.append(d, np.dot(p.Ftot.T, p.detresiduals / p.Nvec))

        # fill in gradient for fourier modes
        self.gradient[ind] = -d + Phiinva

        # loop over all signals
        for ss, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']
            p = self.psr[psrind]

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind + npars]

            # linear timing model parameters (not doing non-linear yet)
            if sig['stype'] == 'lineartimingmodel':

                self.gradient[parind:parind + npars] = - \
                    np.dot(p.Mmat.T, p.detresiduals / p.Nvec)

                # check for special parameterizations
                pindex = 0
                for jj in range(sig['ntotpars']):
                    if sig['bvary'][jj]:

                        if sig['parid'][jj] in ['F0', 'F1', 'Offset']:
                            self.gradient[parind + jj] *= p.ptmparerrs[jj]

                        pindex += 1

            # efac
            if sig['stype'] == 'efac':

                efac = sparameters

                # this efac only applies to some TOAs
                ind = np.flatnonzero(sig['Nvec'])

                self.gradient[parind] = len(ind) / efac - \
                    1 / efac * \
                    np.dot(p.detresiduals[ind] ** 2, 1 / p.Nvec[ind])

            # equad
            if sig['stype'] == 'equad':

                equad = 10 ** sparameters

                # this equad only applies to some TOAs
                ind = np.flatnonzero(sig['Nvec'])

                self.gradient[parind] = 0.5 * len(ind) * np.log(10) - \
                    0.5 * \
                    np.dot(
                        p.detresiduals[ind] ** 2, 1 / p.Nvec[ind]) * np.log(10)

            # red spectral components
            if sig['stype'] == 'spectrum':

                if sig['corr'] == 'single':

                    # get power spectrum coefficients
                    rho = np.array([sparameters, sparameters]).T.flatten()

                    # get corresponding fourier modes
                    ind = self.getSignalNumbersFromDict(self.ptasignals,
                                                        stype='redfouriermode', psrind=psrind)
                    fourierdict = self.ptasignals[ind]
                    psrind_a = fourierdict['pulsarind']
                    parind_a = fourierdict['parindex']
                    npars_a = fourierdict['npars']

                    avals = fourierdict['pstart'].copy()
                    avals[fourierdict['bvary']] = parameters[
                        psrind_a:psrind_a + npars_a]

                    # loop over coefficients
                    for jj in range(npars):

                        self.gradient[parind + jj] = np.log(10) - 0.5 * np.log(10) * \
                            np.sum(avals[2 * jj:2 * jj + 2] ** 2 /
                                   10 ** rho[2 * jj:2 * jj + 2])

            # red powerlaw
            if sig['stype'] == 'powerlaw':

                if sig['corr'] == 'single':

                    # get amplitude and spectral index
                    Amp = 10 ** sparameters[0]
                    gamma = sparameters[1]

                    freqpy = p.Ffreqs
                    f1yr = 1 / 3.16e7
                    #f1yr = 1/self.psr[psrind].Tmax
                    rho = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                   freqpy ** (-gamma) / self.psr[psrind].Tmax)

                    # get corresponding fourier modes
                    ind = self.getSignalNumbersFromDict(self.ptasignals,
                                                        stype='redfouriermode', psrind=psrind)
                    fourierdict = self.ptasignals[ind]
                    psrind_a = fourierdict['pulsarind']
                    parind_a = fourierdict['parindex']
                    npars_a = fourierdict['npars']

                    avals = fourierdict['pstart'].copy()
                    avals[fourierdict['bvary']] = parameters[
                        psrind_a:psrind_a + npars_a]

                    # loop over coefficients
                    self.gradient[parind], self.gradient[parind + 1] = 0, 0
                    for jj in range(int(len(p.Ffreqs) / 2)):

                        # logA gradient
                        self.gradient[parind] += (np.log(10) - 0.5 * np.log(10) *
                                                  np.sum(avals[2 * jj:2 * jj + 2] ** 2 /
                                                         10 ** rho[2 * jj:2 * jj + 2])) * 2

                        self.gradient[parind + 1] += 0.5 * np.log10(f1yr / p.Ffreqs[jj * 2]) * \
                            self.gradient[parind]

        return self.gradient

    """
    Loop over all signals, and fill the diagonal pulsar noise covariance matrix
    (based on efac/equad)
    For two-component noise model, fill the total weights vector

    @param parameters:  The total parameters vector
    @param selection:   Boolean array, indicating which parameters to include
    @param incJitter:   Whether or not to include Jitter in the noise vector,
                        Should only be included if using 'average' compression

    """

    def setPsrNoise(self, parameters, incJitter=False, twoComponent=True):

        # For every pulsar, set the noise vector to zero
        for p in self.psr:
            if p.twoComponentNoise:
                p.Nwvec[:] = 0
                p.Nwovec[:] = 0

            p.Nvec[:] = 0
            p.Qamp = 0

        # Loop over all white noise signals, and fill the pulsar Nvec
        for ss, sig in enumerate(self.ptasignals):

            # short hand
            parind = sig['parindex']    # parameter index
            psrind = sig['pulsarind']   # pulsar index

            # efac signal
            if sig['stype'] == 'efac':

                # is this parameter being varied
                if sig['npars'] == 1:
                    pefac = parameters[parind]

                # if not, use reference value
                else:
                    pefac = sig['pstart'][0]

                # if two component noise, fill weighted noise vectors
                if self.psr[psrind].twoComponentNoise and twoComponent:
                    self.psr[
                        psrind].Nwvec += self.psr[psrind].Wvec * pefac ** 2
                    self.psr[
                        psrind].Nwovec += self.psr[psrind].Wovec * pefac ** 2
                    self.psr[psrind].Nvec += sig['Nvec'] * pefac ** 2

                else:   # use Nvec stored in dictionary
                    self.psr[psrind].Nvec += sig['Nvec'] * pefac ** 2

            # equad signal
            elif sig['stype'] == 'equad':

                # is this parameter being varied
                if sig['npars'] == 1:
                    pequadsqr = 10 ** (2 * parameters[parind])

                # if not use reference value
                else:
                    pequadsqr = 10 ** (2 * sig['pstart'][0])

                # if two component noise, use weighted noise vectors
                if self.psr[psrind].twoComponentNoise and twoComponent:
                    self.psr[psrind].Nwvec += pequadsqr
                    self.psr[psrind].Nwovec += pequadsqr
                    self.psr[psrind].Nvec += sig['Nvec'] * pequadsqr

                else:   # use Nvec stored in dictionary
                    self.psr[psrind].Nvec += sig['Nvec'] * pequadsqr

            # jitter signal
            elif sig['stype'] == 'jitter':

                # is this parameter being varied
                if sig['npars'] == 1:
                    pequadsqr = parameters[parind] ** 2

                # if not use reference value
                else:
                    pequadsqr = sig['pstart'][0] ** 2

                if self.likfunc in ['mark2', 'mark6']:
                    self.psr[psrind].Qamp += sig['Jvec'] * pequadsqr
                else:
                    self.psr[psrind].Nvec += sig['Nvec'] * pequadsqr

            # jitter equad signal
            elif sig['stype'] == 'jitter_equad':

                # is this parameter being varied
                if sig['npars'] == 1:
                    pequadsqr = 10 ** (2 * parameters[parind])

                # if not use reference value
                else:
                    pequadsqr = 10 ** (2 * sig['pstart'][0])

                self.psr[psrind].Qamp += sig['Jvec'] * pequadsqr

                if incJitter:
                    self.psr[psrind].Nvec += sig['Nvec'] * pequadpsr

            # jitter by epoch signal
            elif sig['stype'] == 'jitter_epoch':

                # short hand
                npars = sig['npars']

                # parameters for this signal
                sparameters = sig['pstart'].copy()

                # which ones are varying
                sparameters[sig['bvary']] = parameters[parind:parind + npars]

                self.psr[psrind].Qamp += 10 ** (2 * sparameters)

    """
    Update T matrix to include additional linear terms

    """

    def updateTmatrix(self, parameters):

        for ss, sig in enumerate(self.ptasignals):
            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']
            psr = self.psr[psrind]

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind + npars]

            if sig['stype'] == 'dmshapeletmarg':
                t0, width = sparameters[0], sparameters[1]
                for ii in range(psr.ndmEventCoeffs):
                    amps = [
                        1 if jj == ii else 0 for jj in range(psr.ndmEventCoeffs)]
                    sig = PALutils.constructShapelet(psr.toas / 86400, t0, width, amps) * \
                        4.15e3 / psr.freqs ** 2
                    sig = sig[..., None]
                    if ii == 0:
                        psr.Ttmat = np.append(psr.Tmat, sig, axis=1)
                    else:
                        psr.Ttmat = np.append(psr.Ttmat, sig, axis=1)

            if sig['stype'] == 'frequencyline':
                findex = sig['npsrfreqindex']
                psr.SFfreqs[findex] = 10 ** sparameters[0]
                SFmat = PALutils.singlefourierdesignmatrix(
                        psr.toas, psr.SFfreqs)
                ntm = psr.Mmat_reduced.shape[1]
                nf = len(psr.Ffreqs)
                psr.Ttmat[:,ntm+nf:ntm+nf+2] = SFmat

            if sig['stype'] in ['env_powerlaw', 'env_spectrum']:
                nf = len(psr.Ffreqs)
                t0, width = sparameters[0] * 86400, 10**sparameters[1]
                window = np.exp(-(psr.toas - t0)**2 / 2 / width**2)
                psr.Ttmat[:,-nf:] = (psr.Fmat.T * window).T

            if sig['stype'] in ['scatpowerlaw', 'scatspectrum']:
                nfscat = sig['nscatfreqs']
                nf = psr.Tmat.shape[1]
                Fmat, psr.Fscatfreqs = \
                        PALutils.createfourierdesignmatrix(
                            psr.toas, int(nfscat/2), freq=True,
                            Tspan=psr.Tmax)
                bw = np.array(sig['bwflags']) / 16
                Svec =  (psr.freqs  / bw / 1400.0 / 4.0) ** -sparameters[0]
                Fscatmat = (Svec * Fmat.T).T

                # add Quadratic
                psr.Ttmat[:, nf] = Svec
                psr.Ttmat[:, nf+1] = Svec * (psr.toas - self.Tref)
                psr.Ttmat[:, nf+2] = Svec * (psr.toas - self.Tref)**2

                norm = np.sqrt(np.sum(psr.Ttmat[:,nf:nf+3] ** 2, axis=0))
                psr.Ttmat[:,nf:nf+3] /= norm

                # add fourier basis
                psr.Ttmat[:, nf+3:nf+nfscat+3] = Fscatmat


    """
    Update fourier design matrices to include free floating spectral lines.

    """

    def updateSpectralLines(self, parameters):

        # get frequencies
        addedSingle, addedDMSingle = False, False
        for ss, sig in enumerate(self.ptasignals):
            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind + npars]

            if sig['stype'] == 'frequencyline':
                findex = sig['npsrfreqindex']
                self.psr[psrind].SFfreqs[findex] = 10 ** sparameters[0]
                addedSingle = True

            if sig['stype'] == 'dmfrequencyline':
                findex = sig['npsrfreqindex']
                self.psr[psrind].DMSFfreqs[findex] = 10 ** sparameters[0]
                addedDMSingle = True

        # loop over all pulsars and re-construct F matrices and needed
        # auxiliaries
        for ct, p in enumerate(self.psr):

            # if using average likelihood
            if self.likfunc == 'mark2':
                Ftemp = p.UtF.copy()

                # added frequency independent line
                if addedSingle:
                    SFmat = PALutils.singlefourierdesignmatrix(
                        p.avetoas, p.SFfreqs)
                    Ftemp = np.append(Ftemp, SFmat, axis=1)

                # added DM line
                if addedDMSingle:
                    SdmFmat = PALutils.singlefourierdesignmatrix(
                        p.avetoas, p.DMSFfreqs)
                    Dmat = 4.15e3 / (p.avefreqs ** 2)
                    SDMFmat = (Dmat * SdmFmat.T).T
                    Ftemp = np.append(Ftemp, SDMFmat, axis=1)

                # final product
                p.UtFF = Ftemp

            elif np.any([addedSingle, addedDMSingle]) and self.likfunc != 'mark2':
                Ftemp = p.Ftot.copy()

                # added frequency independent line
                if addedSingle:
                    SFmat = PALutils.singlefourierdesignmatrix(
                        p.toas, p.SFfreqs)
                    Ftemp = np.append(Ftemp, SFmat, axis=1)
                # added DM line
                if addedDMSingle:
                    SdmFmat = PALutils.singlefourierdesignmatrix(
                        p.toas, p.DMSFfreqs)
                    Dmat = 4.15e3 / (p.freqs ** 2)
                    SDMFmat = (Dmat * SdmFmat.T).T
                    Ftemp = np.append(Ftemp, SDMFmat, axis=1)

                # two component noise stuff
                if p.twoComponentNoise:
                    GtF = np.dot(p.Hmat.T, Ftemp)
                    p.AGFF = np.dot(p.Amat.T, GtF)
            else:
                p.FFtot = p.Ftot
                if p.twoComponentNoise:
                    p.AGFF = p.AGF

    """
    Construct complete Phi inverse matrix, including DM corrections if needed

    TODO: this code only works if all pulsars have the same number of frequencies
    want to make this more flexible
    """

    def constructPhiMatrix(self, parameters, constructPhi=False,
                           incCorrelations=True, incTM=False, incJitter=False):

        # Loop over all signals and determine rho (GW signals) and kappa (red +
        # DM signals)
        rho = None
        incDMshapelet = False
        incDMXconstantKernel = False
        incDMXseKernel = False
        for p in self.psr:
            p.band = []
            p.dmband = []

        for ss, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind + npars]

            # spectrum
            if sig['stype'] == 'spectrum':

                # pulsar independent red noise spectrum
                if sig['corr'] == 'single':

                    # doubled amplitudes
                    pcdoubled = np.array(
                        [sparameters, sparameters]).T.flatten()

                    # fill in kappa
                    self.psr[psrind].kappa = pcdoubled

                # correlated signals
                if sig['corr'] in ['gr', 'uniform', 'dipole']:

                    # correlation matrix
                    self.corrmat = sig['corrmat']

                    # define rho
                    rho = np.array([sparameters, sparameters]).T.flatten()

                if sig['corr'] in ['gr_sph']:

                    # correlation matrix
                    nf = int(len(self.psr[psrind].Ffreqs) / 2)
                    clms = np.append(2 * np.sqrt(np.pi), sparameters[nf:])
                    rhovals = sparameters[:nf]
                    self.corrmat = self.computeAniORF(clms)

                    # define rho
                    rho = np.array([rhovals, rhovals]).T.flatten()

            # spectral Model
            if sig['stype'] == 'spectralModel':

                # pulsar independent
                if sig['corr'] == 'single':
                    PAL_spy = 3.16e7
                    Amp = 10 ** sparameters[0]
                    alpha = sparameters[1]
                    fc = 10 ** sparameters[2] / PAL_spy
                    freqpy = self.psr[psrind].Ffreqs

                    pcdoubled = np.log10((Amp * PAL_spy ** 3 / self.psr[psrind].Tmax) *
                                         ((1 + (freqpy / fc) ** 2) ** (-0.5 * alpha)))

                    # fill in kappa
                    self.psr[psrind].kappa = pcdoubled

            # powerlaw spectrum
            if sig['stype'] == 'powerlaw':

                # pulsar independend red noise powerlaw
                if sig['corr'] == 'single':

                    # get Amplitude and spectral index
                    Amp = 10 ** sparameters[0]
                    gamma = sparameters[1]

                    freqpy = self.psr[psrind].Ffreqs
                    f1yr = 1 / 3.16e7
                    pcdoubled = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                         freqpy ** (-gamma) / self.psr[psrind].Tmax)

                    # pcdoubled = np.log10(Amp**2/12/np.pi**2 * f1yr**(gamma-3) * \
                    #                     freqpy**(-gamma))
                    #print 1/self.psr[psrind].Ffreqs[0], self.psr[psrind].Tmax
                    # fill in kappa
                    self.psr[psrind].kappa = pcdoubled

                # correlated signals
                if sig['corr'] in ['gr', 'uniform', 'dipole']:

                    # correlation matrix
                    self.corrmat = sig['corrmat']

                    # number of GW frequencies is the max from all pulsars
                    fgw = self.gwfreqs

                    # get Amplitude and spectral index
                    Amp = 10 ** sparameters[0]
                    gamma = sparameters[1]

                    f1yr = 1 / 3.16e7
                    rho = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                   fgw ** (-gamma) / self.psr[psrind].Tmax)
                    # rho = np.log10(Amp**2/12/np.pi**2 * f1yr**(gamma-3) * \
                    #                     fgw**(-gamma))
                    

                if sig['corr'] in ['gr_sph']:

                    # correlation matrix
                    clms = np.append(2 * np.sqrt(np.pi), sparameters[3:])
                    self.corrmat = self.computeAniORF(clms)

                    # number of GW frequencies is the max from all pulsars
                    fgw = self.gwfreqs

                    # get Amplitude and spectral index
                    Amp = 10 ** sparameters[0]
                    gamma = sparameters[1]

                    f1yr = 1 / 3.16e7
                    rho = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                   fgw ** (-gamma) / self.psr[psrind].Tmax)
                    # rho = np.log10(Amp**2/12/np.pi**2 * f1yr**(gamma-3) * \
                    #                     fgw**(-gamma))

            # band limited powerlaw spectrum
            if sig['stype'] == 'powerlaw_band':

                # pulsar independend red noise powerlaw
                if sig['corr'] == 'single':

                    # get Amplitude and spectral index
                    Amp = 10 ** sparameters[0]
                    gamma = sparameters[1]

                    freqpy = self.psr[psrind].Ffreqs
                    f1yr = 1 / 3.16e7
                    pcdoubled = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                         freqpy ** (-gamma) / self.psr[psrind].Tmax)

                    # fill in kappa
                    self.psr[psrind].band.append(pcdoubled)

            # GWB turnover model
            if sig['stype'] == 'turnover' and sig['corr'] == 'gr':

                    # correlation matrix
                self.corrmat = sig['corrmat']

                # get Amplitude and spectral index
                Amp = 10 ** sparameters[0]
                gamma = sparameters[1]
                f0 = 10 ** sparameters[2]
                kappa = sparameters[3]
                beta = sparameters[4]

                freqpy = self.gwfreqs
                f1yr = 1 / 3.16e7
                hcf = Amp * (freqpy / f1yr) ** ((3 - gamma) / 2) / \
                    (1 + (f0 / freqpy) ** kappa) ** beta
                rho = np.log10(
                    hcf ** 2 / 12 / np.pi ** 2 / freqpy ** 3 / self.psr[psrind].Tmax)
                #rho = np.log10(hcf**2/12/np.pi**2 / freqpy**3)

      # DM spectrum
            if sig['stype'] == 'dmspectrum':

                # pulsar independent DM noise spectrum
                if sig['corr'] == 'single':

                    # doubled amplitudes
                    pcdoubled = np.array(
                        [sparameters, sparameters]).T.flatten()

                    # fill in kappa
                    self.psr[psrind].kappadm = pcdoubled

            # powerlaw DM spectrum
            if sig['stype'] == 'dmpowerlaw':

                # pulsar independend red noise powerlaw
                if sig['corr'] == 'single':

                    # get Amplitude and spectral index
                    Amp = 10 ** sparameters[0]
                    gamma = sparameters[1]

                    freqpy = self.psr[psrind].Fdmfreqs
                    f1yr = 1 / 3.16e7
                    pcdoubled = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                         freqpy ** (-gamma) / self.psr[psrind].Tmax)

                    # fill in kappa
                    self.psr[psrind].kappadm = pcdoubled

            # band limited dmpowerlaw spectrum
            if sig['stype'] == 'dmpowerlaw_band':

                # pulsar independend red noise powerlaw
                if sig['corr'] == 'single':

                    # get Amplitude and spectral index
                    Amp = 10 ** sparameters[0]
                    gamma = sparameters[1]

                    freqpy = self.psr[psrind].Fdmfreqs
                    f1yr = 1 / 3.16e7
                    pcdoubled = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                         freqpy ** (-gamma) /self.psr[psrind].Tmax)

                    # fill in kappa
                    self.psr[psrind].dmband.append(pcdoubled)

            # squared exponential DM spectrum
            if sig['stype'] == 'dmse':

                # pulsar independend red noise powerlaw
                if sig['corr'] == 'single':

                    # get Amplitude and tau
                    Amp = 10 ** sparameters[0]
                    tau = sparameters[1] * 86400

                    freqpy = self.psr[psrind].Fdmfreqs
                    pcdoubled = np.log10(Amp * np.sqrt((2 * np.pi * tau ** 2)) *
                                         np.exp(-2 * np.pi * tau ** 2 * freqpy ** 2))
                    pcdoubled[np.isinf(pcdoubled)] = -100
                    pcdoubled[np.isnan(pcdoubled)] = -100

                    # fill in kappa
                    self.psr[psrind].kappadm = pcdoubled

            # frequency line
            if sig['stype'] == 'frequencyline':

                # doubled amplitudes
                pcdoubled = np.array(
                    [sparameters[1], sparameters[1]]).T.flatten()

                # fill in kappa
                self.psr[psrind].kappasingle = pcdoubled

            # dm frequency line
            if sig['stype'] == 'dmfrequencyline':

                # doubled amplitudes
                pcdoubled = np.array(
                    [sparameters[1], sparameters[1]]).T.flatten()

                # fill in kappa
                self.psr[psrind].kappadmsingle = pcdoubled

            if sig['stype'] == 'dmshapeletmarg':
                incDMshapelet = True

            if sig['stype'] == 'DMXconstantKernel':
                incDMXconstantKernel = True
                lDMXkernelAmp = sparameters[0]

            if sig['stype'] == 'DMXseKernel':
                incDMXseKernel = True
                amp = 10 ** sparameters[0]
                lam = sparameters[1]

            # general ORF function
            if sig['stype'] == 'ORF':
                phi_corr = sparameters
                self.corrmat = self.computeSphORF(phi_corr)

            if sig['stype'] == 'env_powerlaw':
                # get Amplitude and spectral index
                Amp = 10 ** sparameters[2]
                gamma = sparameters[3]

                freqpy = self.psr[psrind].Ffreqs
                f1yr = 1 / 3.16e7
                pcdoubled = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                     freqpy ** (-gamma) / self.psr[psrind].Tmax)

                # fill in kappa
                self.psr[psrind].kappa_env = pcdoubled

            if sig['stype'] == 'env_spectrum':
                # doubled amplitudes
                pcdoubled = np.array(
                    [sparameters[2:], sparameters[2:]]).T.flatten()

                # fill in kappa
                self.psr[psrind].kappa_env = pcdoubled

            if sig['stype'] == 'ext_powerlaw':
                # get Amplitude and spectral index
                Amp = 10 ** sparameters[0]
                gamma = sparameters[1]

                freqpy = self.psr[psrind].Fextfreqs
                f1yr = 1 / 3.16e7
                Tmax = 1 / self.psr[psrind].Fextfreqs[0]
                pcdoubled = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                     freqpy ** (-gamma) / Tmax)

                # fill in kappa
                self.psr[psrind].kappa_ext = pcdoubled

            if sig['stype'] == 'ext_spectrum':
                # doubled amplitudes
                pcdoubled = np.array(
                    [sparameters[:], sparameters[:]]).T.flatten()

                # fill in kappa
                self.psr[psrind].kappa_ext = pcdoubled

            if sig['stype'] == 'scatpowerlaw':
                # get Amplitude and spectral index
                Amp = 10 ** sparameters[1]
                gamma = sparameters[2]

                freqpy = self.psr[psrind].Fscatfreqs
                f1yr = 1 / 3.16e7
                Tmax = self.psr[psrind].Tmax
                pcdoubled = np.log10(Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) *
                                     freqpy ** (-gamma) / Tmax)

                # fill in kappa
                self.psr[psrind].kappa_scat[:3] = 80
                self.psr[psrind].kappa_scat[3:] = pcdoubled

            if sig['stype'] == 'scatspectrum':
                # doubled amplitudes
                pcdoubled = np.array(
                    [sparameters[1:], sparameters[1:]]).T.flatten()

                # fill in kappa
                self.psr[psrind].kappa_scat[:3] = 80
                self.psr[psrind].kappa_scat[3:] = pcdoubled

        # now that we have obtained rho and kappa, we can construct Phiinv
        sigdiag = []
        sigoffdiag = []
        self.gwamp = 0

        # no correlated signals (easy)
        if rho is None:

            # loop over all pulsars
            for ii, p in enumerate(self.psr):

                if p.band != []:
                    p.kappa = np.append(p.kappa, np.hstack(p.band))

                if p.dmband != []:
                    p.kappadm = np.append(p.kappadm, np.hstack(p.dmband))

                # have both red noise and DM variations
                if p.incRed and p.incDM:
                    # if np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = np.concatenate((p.kappa, p.kappadm))

                # red noise but no dm
                elif p.incRed and not(p.incDM):
                    # elif np.any(p.kappa) and ~np.any(p.kappadm):
                    p.kappa_tot = p.kappa.copy()

                # dm but no red noise
                elif not(p.incRed) and p.incDM:
                    # elif ~np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = p.kappadm.copy()

                # neither
                else:
                    p.kappa_tot = np.ones(p.kappa.shape) * -40

                if p.nSingleFreqs > 0:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappasingle))

                if p.nSingleDMFreqs > 0:
                    p.kappa_tot = np.concatenate(
                        (p.kappa_tot, p.kappadmsingle))

                # append to signal diagonal
                if self.haveExt:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappa_ext))

                if incTM:
                    p.kappa_tot = np.concatenate((np.ones(p.Mmat_reduced.shape[1]) * 80,
                                                  p.kappa_tot))
                    if incJitter:
                        p.kappa_tot = np.concatenate(
                            (p.kappa_tot, np.log10(p.Qamp)))
                    if incDMshapelet:
                        p.kappa_tot = np.concatenate(
                            (p.kappa_tot, np.ones(p.ndmEventCoeffs) * 80))
                    if p.nDMX:
                        if incDMXconstantKernel:
                            p.kappa_tot = np.concatenate((p.kappa_tot,
                                                          np.ones(p.nDMX) * lDMXkernelAmp * 2))
                        if incDMXseKernel:
                            K = PALutils.constructSEkernel(
                                p.DMXtimes / 86400, lam, amp)
                        else:
                            p.kappa_tot = np.concatenate(
                                (p.kappa_tot, np.ones(p.nDMX) * 80))
                            p.kappa_tot[-p.nDMX] = -20
                    if self.haveEnvelope:
                        p.kappa_tot = np.concatenate((p.kappa_tot, p.kappa_env))

                if self.haveScat:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappa_scat))

                sigdiag.append(10 ** p.kappa_tot)

            # convert to array and flatten
            self.Phi = np.hstack(sigdiag)
            np.fill_diagonal(self.Phiinv, 1 / self.Phi)
            if incDMXseKernel:
                ndmx = K.shape[0]
                u, s, v = np.linalg.svd(K)
                sinv = 1 / s
                sinv[s[0] / s > 1e-16] = 0
                Kinv = np.dot(u, (sinv * u).T)
                logdetK = np.sum(np.log(s))
                #cf = sl.cho_factor(K)
                #Kinv = sl.cho_solve(cf, np.eye(K.shape[0]))
                self.Phiinv[-ndmx:, -ndmx:] = Kinv
                #self.Phiinv[-ndmx, -ndmx] = 1e20
                #logdetK = np.sum(2*np.log(np.diag(cf[0])))
                self.logdetPhi = np.sum(np.log(self.Phi[:-ndmx])) + logdetK
            else:
                self.logdetPhi = np.sum(np.log(self.Phi[:]))

        # Do not include correlations but include GWB in red noise
        if rho is not None and not(incCorrelations):

            # loop over all pulsars
            for ii, p in enumerate(self.psr):

                # have both red noise and DM variations
                if p.incRed and p.incDM:
                    # if np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = np.concatenate((p.kappa, p.kappadm))

                # red noise but no dm
                elif p.incRed and not(p.incDM):
                    # elif np.any(p.kappa) and ~np.any(p.kappadm):
                    p.kappa_tot = p.kappa.copy()

                # dm but no red noise
                elif not(p.incRed) and p.incDM:
                    # elif ~np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = p.kappadm.copy()

                # neither
                else:
                    p.kappa_tot = np.ones(p.kappa.shape) * -40

                if p.nSingleFreqs > 0:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappasingle))

                if p.nSingleDMFreqs > 0:
                    p.kappa_tot = np.concatenate(
                        (p.kappa_tot, p.kappadmsingle))

                # get number of DM freqs (not included in GW spectrum)
                ndmfreq = np.sum(p.kappadm != 0)

                # append to rho
                if ndmfreq > 0:
                    self.gwamp = np.concatenate((10 ** rho, np.zeros(ndmfreq)))
                else:
                    self.gwamp = 10 ** rho

                # append to signal diagonal
                p.kappa_tot = np.log10(10 ** p.kappa_tot + self.gwamp)
                if self.haveExt:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappa_ext))
                if incTM:
                    p.kappa_tot = np.concatenate((np.ones(p.Mmat.shape[1]) * 80,
                                                  p.kappa_tot))
                    if incJitter:
                        p.kappa_tot = np.concatenate(
                            (p.kappa_tot, np.log10(p.Qamp)))
                    if incDMshapelet:
                        p.kappa_tot = np.concatenate(
                            (p.kappa_tot, np.ones(p.ndmEventCoeffs) * 80))

                    if self.haveEnvelope:
                        p.kappa_tot = np.concatenate((p.kappa_tot, p.kappa_env))

                if self.haveScat:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappa_scat))

                # append to signal diagonal
                sigdiag.append(10 ** p.kappa_tot)

            # convert to array and flatten
            self.Phi = np.hstack(sigdiag)
            np.fill_diagonal(self.Phiinv, 1 / self.Phi)
            #self.logdetPhi = np.sum(np.log(self.Phi[-self.npftot[ii]:]))
            self.logdetPhi = np.sum(np.log(self.Phi))

        # correlated signals (not as easy)
        if rho is not None and incCorrelations:

            for ii, p in enumerate(self.psr):

                # have both red noise and DM variations
                if p.incRed and p.incDM:
                    # if np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = np.concatenate((p.kappa, p.kappadm))

                # red noise but no dm
                elif p.incRed and not(p.incDM):
                    # elif np.any(p.kappa) and ~np.any(p.kappadm):
                    p.kappa_tot = p.kappa.copy()

                # dm but no red noise
                elif not(p.incRed) and p.incDM:
                    # elif ~np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = p.kappadm.copy()

                # for now, assume that GW freqs is the same as
                # number of freqs per pulsar

                # get number of DM freqs (not included in GW spectrum)
                ndmfreq = np.sum(p.kappadm != 0)

                # append to rho
                if ndmfreq > 0:
                    self.gwamp = np.concatenate((10 ** rho, np.zeros(ndmfreq)))
                else:
                    self.gwamp = 10 ** rho

                # append to signal diagonal
                p.kappa_tot = np.log10(10 ** p.kappa_tot + self.gwamp)
                sigdiag.append(10**p.kappa_tot)
                if incTM:
                    p.kappa_tot = np.concatenate((np.ones(p.Mmat.shape[1]) * 80,
                                                  p.kappa_tot))
                    self.logdetPhi += np.sum(np.log(10**np.ones(p.Mmat.shape[1]) * 80))
                    if incJitter:
                        p.kappa_tot = np.concatenate(
                            (p.kappa_tot, np.log10(p.Qamp)))
                        self.logdetPhi += np.sum(np.log(p.Qamp))

                # append to off diagonal elements
                sigoffdiag.append(self.gwamp)

            # compute Phi inverse from Lindley's code
            nftot = self.ngwf + np.max(self.npfdm)
            smallMatrix = np.zeros((nftot, self.npsr, self.npsr))
            for ii in range(self.npsr):

                for jj in range(ii, self.npsr):
                    if ii == jj:
                        smallMatrix[:, ii, jj] = self.corrmat[ii, jj] * sigdiag[jj]
                    else:
                        smallMatrix[:, ii, jj] = self.corrmat[ii, jj] * sigoffdiag[jj]
                        smallMatrix[:, jj, ii] = smallMatrix[:, ii, jj]

            # invert them
            self.logdetPhi = 0
            for ii in range(nftot):
                L = sl.cho_factor(smallMatrix[ii, :, :])
                smallMatrix[ii, :, :] = sl.cho_solve(L, np.eye(self.npsr))
                self.logdetPhi += np.sum(2 * np.log(np.diag(L[0])))

            # now fill in real covariance matrix
            if self.likfunc in ['mark6', 'mark9']:
                ind2 = []
                start, stop = 0, 0
                for ct, p in enumerate(self.psr):
                    ntmpars = len(p.ptmdescription)
                    if incJitter:
                        njitter = len(p.avetoas)
                    else:
                        njitter = 0
                    start += ntmpars
                    stop += ntmpars + nftot
                    ind2.append(np.arange(start, stop))
                    start += njitter + nftot
                    stop += njitter

                start, stop = 0, 0
                for ii, p in enumerate(self.psr):
                    ntmpars = len(p.ptmdescription)
                    if incJitter:
                        njitter = len(p.avetoas)
                    else:
                        njitter = 0
                    start += ntmpars
                    stop += ntmpars + nftot
                    ind1 = np.arange(start, stop)
                    start += njitter + nftot
                    stop += njitter
                    self.Phiinv[ind1, ind2[jj]] = smallMatrix[:, ii, jj]
            else:
                ind2 = [np.arange(jj * nftot, jj * nftot + nftot)
                        for jj in range(self.npsr)]
                for ii in range(self.npsr):
                    ind1 = np.arange(ii * nftot, ii * nftot + nftot)
                    for jj in range(0, self.npsr):
                        self.Phiinv[ind1, ind2[jj]] = smallMatrix[:, ii, jj]

    
    def construct_dense_cov_matrix(self, parameters, incJitter=False):
        """
        WARNING: ONLY WORKS FOR A SINGLE PULSAR
        """

        # fill in timing model part
        ntmpars = self.psr[0].Mmat.shape[1]
        nepoch = len(self.psr[0].avetoas)
        self.Phiinv[:ntmpars, :ntmpars] = 0
        self.Phi[:] = 0

        for ss, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']

            psr = self.psr[psrind]

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind + npars]

            # powerlaw spectrum
            if sig['stype'] == 'powerlaw':

                # pulsar independend red noise powerlaw
                if sig['corr'] == 'single':

                    # get Amplitude and spectral index
                    Amp = 10 ** sparameters[0]
                    gamma = sparameters[1]

                    self.Phi[ntmpars:nepoch+ntmpars, ntmpars:nepoch+ntmpars] += \
                            PALutils.createRedNoiseCovarianceMatrix(
                                psr.tm, Amp, gamma, fast=True, fH=1.3e-7)


                # correlated signals
                if sig['corr'] in ['gr', 'uniform', 'dipole']:

                    # correlation matrix
                    self.corrmat = sig['corrmat']

                    # get Amplitude and spectral index
                    Amp = 10 ** sparameters[0]
                    gamma = sparameters[1]
                    
                    self.Phi[ntmpars:nepoch+ntmpars, ntmpars:nepoch+ntmpars] += \
                            PALutils.createRedNoiseCovarianceMatrix(
                                psr.tm, Amp, gamma, fast=True, fH=1.3e-7)

        # add jitter
        if incJitter:
            self.Phi[ntmpars:nepoch+ntmpars, ntmpars:nepoch+ntmpars] += \
                    np.diag(self.psr[0].Qamp)

        # invert C
        cf = sl.cho_factor(self.Phi[ntmpars:nepoch+ntmpars, ntmpars:nepoch+ntmpars])
        self.Phiinv[ntmpars:nepoch+ntmpars, ntmpars:nepoch+ntmpars] = \
                sl.cho_solve(cf, np.eye(nepoch))

        # log determinant
        self.logdetPhi = np.sum(2 * np.log(np.diag(cf[0])))


    """
    Function to contruct non-gaussian signal
    coefficients.
    """

    def getNonGaussianComponents(self, parameters):

        for ss, sig in enumerate(self.ptasignals):

            # Create a parameters array for this particular signal
            sparameters = sig['pstart'].copy()
            sparameters[sig['bvary']] = \
                parameters[sig['parindex']:sig['parindex'] + sig['npars']]

            # shorthand
            psrind = sig['pulsarind']

            if sig['stype'] == 'nongausscoeff':
                self.psr[psrind].nalpha = sig['npars'] + 1
                self.psr[psrind].alphacoeff = sparameters

                # prepend 0th component
                val = np.sqrt(1 - np.sum(self.psr[psrind].alphacoeff ** 2))
                self.psr[psrind].alphacoeff = np.insert(
                    self.psr[psrind].alphacoeff, 0, val)

    """
    Update deterministic sources
    """

    def updateDetSources(self, parameters):

        # Set all the detresiduals equal to residuals
        for ct, p in enumerate(self.psr):
            p.detresiduals = p.residuals.copy()

        # In the case we have numerical timing model (linear/nonlinear)
        for ss, sig in enumerate(self.ptasignals):

            # Create a parameters array for this particular signal
            sparameters = sig['pstart'].copy()
            sparameters[sig['bvary']] = \
                parameters[sig['parindex']:sig['parindex'] + sig['npars']]

            if sig['stype'] == 'lineartimingmodel':
                # This one only applies to one pulsar at a time
                ind = []
                pp = sig['pulsarind']
                newdes = sig['parid']
                psr = self.psr[pp]

                if len(newdes) == psr.Mmat.shape[1]:
                    Mmat = psr.Mmat
                else:
                    raise ValueError('ERROR: Number of timing model parameters \
                                     does not match size of design matrix')

                # determine parameter offsets
                offset = []
                pindex = 0
                for jj in range(sig['ntotpars']):
                    if sig['bvary'][jj]:
                        # check for direct offset parameters
                        # if sig['parid'][jj] == 'Offset':
                        #    offset += [sparameters[pindex]*psr.ptmparerrs[jj]]
                        # elif sig['parid'][jj] == 'F0':
                        #    offset += [sparameters[pindex]*psr.ptmparerrs[jj]]
                        # elif sig['parid'][jj] == 'F1':
                        #    offset += [sparameters[pindex]*psr.ptmparerrs[jj]]
                        if sig['parid'][jj] == 'Offset':
                            offset += [sparameters[pindex] * psr.norm[jj]]
                        elif sig['parid'][jj] == 'F0':
                            offset += [sparameters[pindex] * psr.norm[jj]]
                        elif sig['parid'][jj] == 'F1':
                            offset += [sparameters[pindex] * psr.norm[jj]]
                        else:
                            offset += [(sparameters[pindex] -
                                        sig['pstart'][jj]) * psr.norm[jj]]

                        pindex += 1

                # residuals = M * pars
                #eps = np.dot(psr.Vmat.T, np.array(offset))*psr.Svec
                psr.detresiduals -= np.dot(Mmat, np.array(offset))
                #psr.detresiduals -= np.dot(Mmat, eps)
                # print  np.dot(Mmat, np.array(offset))

            elif sig['stype'] == 'nonlineartimingmodel':
                # The t2psr libstempo object has to be set. Assume it is.
                pp = sig['pulsarind']
                psr = self.psr[pp]

                # For each varying parameter, update the libstempo object
                # parameter with the new value
                pindex = 0
                offset = np.zeros(len(psr.detresiduals))
                for jj in range(sig['ntotpars']):
                    if sig['bvary'][jj]:
                        # If this parameter varies, update the parameter
                        if sig['parid'][jj] in ['F0', 'F1']:
                            psr.t2psr[sig['parid'][jj]].val =\
                                psr.ptmpars[
                                    jj] + psr.ptmparerrs[jj] * sparameters[pindex]
                        elif sig['parid'][jj] == 'Offset':
                            offset[:] = psr.ptmparerrs[
                                jj] * sparameters[pindex]
                        else:
                            psr.t2psr[sig['parid'][jj]].val = \
                                np.longdouble(sparameters[pindex])
                        pindex += 1

                # Generate the new residuals
                psr.detresiduals = np.array(psr.t2psr.residuals(updatebats=True),
                                            dtype=np.double)[psr.isort] + offset

            # fourier modes
            if sig['stype'] in ['redfouriermode', 'gwfouriermode']:
                pp = sig['pulsarind']
                psr = self.psr[pp]

                # fourier amplitudes
                a = sparameters
                psr.detresiduals -= np.dot(psr.Fmat, a)

            if sig['stype'] == 'dmfouriermode':
                pp = sig['pulsarind']
                psr = self.psr[pp]

                # fourier amplitudes
                a = sparameters

                psr.detresiduals -= np.dot(psr.DF, a)

            # dm shapelet
            if sig['stype'] == 'dmshapelet':
                psr = self.psr[sig['pulsarind']]
                t0, width, amps = sparameters[
                    0], sparameters[1], sparameters[2:]
                sig = PALutils.constructShapelet(psr.toas / 86400, t0, width, amps) * \
                    4.15e3 / psr.freqs ** 2

                psr.detresiduals -= sig

            # GW wavelet signal
            if sig['stype'] == 'gwwavelet':
                theta = sparameters[0]
                phi = sparameters[1]
                psi = sparameters[2]
                eps = sparameters[3]
                A = 10**sparameters[4::5]
                f0 = 10**sparameters[5::5]
                t0 = sparameters[6::5] * 86400
                Q = sparameters[7::5]
                phase0 = sparameters[8::5]

                wsig = PALutils.construct_gw_wavelet(
                    self.psr, theta, phi, psi, eps, A, t0, f0,
                    Q, phase0)

                for ct, p in enumerate(self.psr):
                    p.detresiduals -= wsig[ct]

            # continuous wave signal
            if sig['stype'] == 'cw':

                # get pulsar distances
                nsigs = self.getNumberOfSignalsFromDict(self.ptasignals,
                                                        stype='pulsardistance',
                                                        corr='single')

                # including pulsar term
                pdist = []
                if np.any(nsigs):
                    incPterm = True
                    signum = self.getSignalNumbersFromDict(self.ptasignals,
                                                           stype='pulsardistance',
                                                           corr='single')

                    # check to make sure we have all distances
                    if len(signum) != self.npsr:
                        raise ValueError(
                            'ERROR: Number of pulsar distances != number of pulsars!')

                    # current timing model parameters
                    for signum0 in signum:
                        sig0 = self.ptasignals[signum0]
                        pdist.append(parameters[sig0['parindex']:(sig0['parindex'] +
                                                                  sig0['npars'])])
                    pdist = np.array(pdist).copy()

                else:
                    incPterm = False
                    pdist = None

                # get pulsar term parameters
                nsigs = self.getNumberOfSignalsFromDict(
                    self.ptasignals, stype='pulsarTerm',
                    corr='single')
                if np.any(nsigs):
                    signum = self.getSignalNumbersFromDict(
                        self.ptasignals, stype='pulsarTerm',
                        corr='single')

                    pphase, pfgw = [], []
                    for s0 in signum:
                        sig0 = self.ptasignals[s0]
                        pphase.append(parameters[sig0['parindex']])
                        pfgw.append(parameters[sig0['parindex']+1])

                # upper limit (h=2M^{5/3}(\pi f)^{2/3}/d_L)
                if sig['model'] == 'upperLimit':
                    dist = 2 * (10 ** sparameters[2] * 4.9e-6) ** (5 / 3) \
                        * (np.pi * 10 ** sparameters[4]) ** (2 / 3) / 10 ** sparameters[3]
                    dist /= 1.0267e14
                    mc = 10**sparameters[2]
                elif sig['model'] == 'mass_ratio':
                    dist = 10**sparameters[3]
                    q = 10**sparameters[-1]
                    Mtot = 10**sparameters[2]
                    mc = Mtot * (q / (1+q)**2)**(3/5)
                else:
                    dist = 10 ** sparameters[3]
                    mc = 10**sparameters[2]

                # construct CW signal
                if sig['model'] != 'free':
                    cwsig = PALutils.createResidualsFast(
                        self.psr, sparameters[0], sparameters[1],
                        mc, dist, 10 ** sparameters[4],
                        sparameters[5], sparameters[6], sparameters[7],
                        pdist=pdist, psrTerm=incPterm, phase_approx=True,
                        tref=self.Tref)
                else:
                    cwsig = PALutils.createResidualsFree(
                        self.psr, sparameters[0], sparameters[1],
                        10**sparameters[2], 10**sparameters[3],
                        sparameters[4], sparameters[5], sparameters[6],
                        np.array(pphase), 10**np.array(pfgw), psrTerm=True,
                        tref=self.Tref)

                # loop over all pulsars and subtract off CW signal
                for ct, p in enumerate(self.psr):
                    p.detresiduals -= cwsig[ct]

        # If necessary, transform these residuals to two-component basis
        for pp, p in enumerate(self.psr):
            if p.twoComponentNoise:
                Gr = np.dot(p.Hmat.T, p.detresiduals)
                p.AGr = np.dot(p.Amat.T, Gr)

    """
    Simulate residuals for a single pulsar
    """

    def simData(self, parameters, setup=False, turnover=False, f0=1e-9,
                beta=1, power=1):

        # only need to do this if parameters change
        self.setPsrNoise(parameters, incJitter=False, twoComponent=False)
        if setup:

            # set red noise, DM and GW parameters
            self.gwamp = 0
            self.corrmat = np.eye(self.npsr)
            self.constructPhiMatrix(parameters, incCorrelations=False)

            # construct cholesky decomp on ORF
            self.corrmatCho = sl.cholesky(self.corrmat)

        if np.any(self.gwamp):
            #y = np.random.randn(self.npsr, self.ngwf)
            #ypsr = np.dot(self.corrmatCho, y)
            gwbs = PALutils.createGWB(self.psr, 10 ** parameters[-2], parameters[-1],
                                      DM=False, noCorr=False, seed=None,
                                      turnover=turnover, f0=f0, beta=beta,
                                      power=power)

        # begin loop over all pulsars
        findex = 0
        res = []
        for ct, p in enumerate(self.psr):

            # number of frequencies
            npftot = self.npftot[ct]

            # white noise
            n = np.sqrt(p.Nvec)
            w = np.random.randn(len(p.toas))
            white = n * w

            # jitter noise
            if p.Umat is not None:
                j = np.sqrt(p.Qamp)
                w = np.random.randn(len(p.avetoas))
                white += np.dot(p.Umat, j * w)

            # red noise
            phi = np.sqrt(10 ** p.kappa_tot)
            x = np.random.randn(npftot)
            red = np.dot(p.Ftot, phi * x)

            # gwb noise
            if np.any(self.gwamp):
                gwb = gwbs[ct]
                #gwphi = np.sqrt(self.gwamp)
                #phiy = gwphi*ypsr[ct,:]
                #gwb = np.dot(p.Ftot, phiy)
            else:
                gwb = 0

            # add residuals
            res.append(white + red + gwb)

            # increment frequency index
            findex += npftot

        return res

    """
    Optimal Statistic

    """

    def optimalStatistic(self, parameters):

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=False)

        # get correlation matrix
        ORF = PALutils.computeORF(self.psr)

        # loop over all pulsars
        Y = []
        X = []
        Z = []
        FGGNGGF = []
        for ct, p in enumerate(self.psr):

            if p.twoComponentNoise:

                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component
                # basis
                X.append(np.dot(p.AGF.T, p.AGr / p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1 / p.Nwvec) * p.AGF.T).T
                FGGNGGF.append(np.dot(p.AGF.T, right))

            else:

                # G(G^TNG)^{-1}G^T = N^{-1} -
                # N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0 / p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiF = ((1.0 / p.Nvec) * p.Ftot.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiF = np.dot(NiGc.T, p.Ftot)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(
                        np.log(p.Nvec)) + 2 * np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                # F^TG(G^TNG)^{-1}G^T\delta t
                X.append(np.dot(p.Ftot.T, Nir - NiGcNiGcr))

                # compute F^TG(G^TNG)^{-1}G^TF
                FGGNGGF.append(
                    np.dot(NiF.T, p.Ftot) - np.dot(GcNiF.T, GcNiGcF))

            # compute relevant quantities
            nf = len(p.Ffreqs) + len(p.Fdmfreqs)
            #phiinv = 1/self.Phi[ct*nf:(ct*nf+nf)]
            phiinv = np.diag(self.Phiinv)[ct * nf:(ct * nf + nf)]
            Sigma = np.diag(phiinv) + FGGNGGF[ct]

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Sigma)
                right = sl.cho_solve(cf, FGGNGGF[ct])
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                right = np.dot(
                    Vh.T, np.dot(np.diag(1.0 / s), np.dot(U.T, FGGNGGF[ct])))

            Y.append(X[ct] - np.dot(X[ct], right))

            Z.append(FGGNGGF[ct] - np.dot(FGGNGGF[ct], right))

        # cross correlations
        top = 0
        bot = 0
        rho, sig, xi = [], [], []
        for ii in range(self.npsr):
            for jj in range(ii + 1, self.npsr):

                fgw = self.psr[ii].Ffreqs

                # get Amplitude and spectral index
                Amp = 1
                gamma = 13 / 3

                f1yr = 1 / 3.16e7
                pcdoubled = Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) * \
                    fgw ** (-gamma) / np.sqrt(self.psr[ii].Tmax
                                              * self.psr[jj].Tmax)

                phiIJ = 0.5 * np.concatenate((pcdoubled,
                                              np.zeros(len(self.psr[ii].Fdmfreqs))))

                top = np.dot(Y[ii], phiIJ * Y[jj])
                bot = np.trace(
                    np.dot((Z[ii] * phiIJ.T).T, (Z[jj] * phiIJ.T).T))

                # cross correlation and uncertainty
                rho.append(top / bot)
                sig.append(1 / np.sqrt(bot))
                xi.append(PALutils.angularSeparation(self.psr[ii].theta[0],
                                                     self.psr[ii].phi[0],
                                                     self.psr[jj].theta[0],
                                                     self.psr[jj].phi[0]))

        # return Opt, sigma, snr

        # return top/bot, 1/np.sqrt(bot), top/np.sqrt(bot)
        return np.array(xi), np.array(rho), np.array(sig), \
            np.sum(np.array(rho) * ORF / np.array(sig) ** 2) / np.sum(ORF ** 2 / np.array(sig) ** 2), \
            1 / np.sqrt(np.sum(ORF ** 2 / np.array(sig) ** 2))

    """
    Optimal Statistic

    """

    def optimalStatisticCoarse(self, parameters):

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=False)

        # get correlation matrix
        ORF = PALutils.computeORF(self.psr)

        # compute the white noise terms in the log likelihood
        Y = []
        X = []
        Z = []
        UGGNGGU = []
        for ct, p in enumerate(self.psr):

            if p.twoComponentNoise:

                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component
                # basis
                X.append(np.dot(p.AGU.T, p.AGr / p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1 / p.Nwvec) * p.AGU.T).T
                UGGNGGU.append(np.dot(p.AGU.T, right))

            else:

                # G(G^TNG)^{-1}G^T = N^{-1} -
                # N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0 / p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiU = ((1.0 / p.Nvec) * p.Umat.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiU = np.dot(NiGc.T, p.Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(
                        np.log(p.Nvec)) + 2 * np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                # F^TG(G^TNG)^{-1}G^T\delta t
                X.append(np.dot(p.Umat.T, Nir - NiGcNiGcr))

                # compute F^TG(G^TNG)^{-1}G^TF
                UGGNGGU.append(
                    np.dot(NiU.T, p.Umat) - np.dot(GcNiU.T, GcNiGcU))

            # construct modified phi matrix
            nf = len(p.Ffreqs) + len(p.Fdmfreqs)
            Phi0 = np.diag(self.Phi[ct * nf:(ct * nf + nf)])
            UPhiU = np.dot(p.UtF, np.dot(Phi0, p.UtF.T))
            Phi = UPhiU + np.diag(p.Qamp)

            try:
                cf = sl.cho_factor(Phi)
                phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Phi)
                if not np.all(s > 0):
                    return -np.inf
                    #raise ValueError("ERROR: Phi singular according to SVD")
                phiinv = np.dot(Vh.T, np.dot(np.diag(1.0 / s), U.T))

            Sigma = phiinv + UGGNGGU[ct]

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Sigma)
                right = sl.cho_solve(cf, UGGNGGU[ct])
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                right = np.dot(
                    Vh.T, np.dot(np.diag(1.0 / s), np.dot(U.T, UGGNGGU[ct])))

            Y.append(X[ct] - np.dot(X[ct], right))

            Z.append(UGGNGGU[ct] - np.dot(UGGNGGU[ct], right))

        # cross correlations
        top = 0
        bot = 0
        rho, sig, xi = [], [], []
        for ii in range(self.npsr):
            for jj in range(ii + 1, self.npsr):

                fgw = self.psr[ii].Ffreqs
                nf = len(fgw)
                nfdm = len(self.psr[ii].Fdmfreqs)

                # get Amplitude and spectral index
                Amp = 1
                gamma = 13 / 3

                f1yr = 1 / 3.16e7
                pcdoubled = Amp ** 2 / 12 / np.pi ** 2 * f1yr ** (gamma - 3) * \
                    fgw ** (-gamma) / \
                    np.sqrt(self.psr[ii].Tmax * self.psr[jj].Tmax)

                Phi = np.zeros(nf + nfdm)
                #di = np.diag_indices(nf)
                Phi[:nf] = pcdoubled

                phiIJ = 0.5 * \
                    np.dot(self.psr[ii].UtF, (Phi * self.psr[jj].UtF).T)
                #phiIJ = np.dot(self.psr[ii].UtF, (Phi * self.psr[jj].UtF).T)

                top = np.dot(Y[ii], np.dot(phiIJ, Y[jj]))
                bot = np.trace(
                    np.dot(Z[ii], np.dot(phiIJ, np.dot(Z[jj], phiIJ.T))))

                # cross correlation and uncertainty
                rho.append(top / bot)
                sig.append(1 / np.sqrt(bot))
                xi.append(PALutils.angularSeparation(
                    self.psr[ii].theta, self.psr[ii].phi, self.psr[jj].theta, self.psr[jj].phi))

                if np.isnan(sig[-1]):
                    print self.psr[ii].name, self.psr[jj].name, rho[-1], sig[-1]

        # return Opt, sigma, snr

        # return top/bot, 1/np.sqrt(bot), top/np.sqrt(bot)
        return np.array(xi), np.array(rho), np.array(sig), \
            np.sum(np.array(rho) * ORF / np.array(sig) ** 2) / np.sum(ORF ** 2 / np.array(sig) ** 2), \
            1 / np.sqrt(np.sum(ORF ** 2 / np.array(sig) ** 2))

    """
    mark 1 log likelihood. Note that this is not the same as mark1 in piccard

    EFAC + EQUAD + Red noise + DMV + GWs

    No jitter or frequency lines

    Uses Woodbury lemma

    """

    def mark1LogLikelihood(self, parameters, incCorrelations=True):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # frequency lines
        self.updateSpectralLines(parameters)

        # set red noise, DM and GW parameters
        try:
            self.constructPhiMatrix(
                parameters, incCorrelations=incCorrelations)
        except np.linalg.LinAlgError:
            return -np.inf

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        FGGNGGF = []
        nfref = 0
        for ct, p in enumerate(self.psr):

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(
                    np.isinf(p.detresiduals)):
                return -np.inf

            if p.twoComponentNoise:

                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component
                # basis
                if ct == 0:
                    d = np.dot(p.AGFF.T, p.AGr / p.Nwvec)
                else:
                    d = np.append(d, np.dot(p.AGFF.T, p.AGr / p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1 / p.Nwvec) * p.AGFF.T).T
                FGGNGGF.append(np.dot(p.AGFF.T, right))

                # log determinant of G^TNG
                logdet_N = np.sum(np.log(p.Nwvec))

                # triple product in likelihood function
                rGGNGGr = np.sum(p.AGr ** 2 / p.Nwvec)

            elif not(p.twoComponentNoise) and self.compression == 'average':

                GNG = np.dot(p.Hmat.T * p.Nvec, p.Hmat)
                cf = sl.cho_factor(GNG)
                logdet_N = 2 * np.sum(np.log(np.diag(cf[0])))
                GNGGdt = sl.cho_solve(cf, np.dot(p.Hmat.T, p.detresiduals))

                if ct == 0:
                    d = np.dot(p.GtF.T, GNGGdt)
                else:
                    d = np.append(d, np.dot(p.GtF.T, GNGGdt))

                GNGGF = sl.cho_solve(cf, p.GtF)
                FGGNGGF.append(np.dot(p.GtF.T, GNGGF))

                rGGNGGr = np.dot(p.detresiduals, np.dot(p.Hmat, GNGGdt))

            else:

                # G(G^TNG)^{-1}G^T = N^{-1} -
                # N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0 / p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiF = ((1.0 / p.Nvec) * p.FFtot.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiF = np.dot(NiGc.T, p.FFtot)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(
                        np.log(p.Nvec)) + 2 * np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                # F^TG(G^TNG)^{-1}G^T\delta t
                if ct == 0:
                    d = np.dot(p.FFtot.T, Nir - NiGcNiGcr)
                else:
                    d = np.append(d, np.dot(p.FFtot.T, Nir - NiGcNiGcr))

                # triple product in likelihood function
                rGGNGGr = np.dot(p.detresiduals, Nir) - np.dot(GcNir, GcNiGcr)

                # compute F^TG(G^TNG)^{-1}G^TF
                FGGNGGF.append(
                    np.dot(NiF.T, p.FFtot) - np.dot(GcNiF.T, GcNiGcF))

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rGGNGGr)

            # calculate red noise piece
            if not incCorrelations:

                # compute sigma
                logdet_Sigma = 0
                nf = self.npftot[ct]
                Sigma = FGGNGGF[
                    ct] + self.Phiinv[nfref:(nfref + nf), nfref:(nfref + nf)]
                dd = d[nfref:(nfref + nf)]

                # cholesky decomp for maximum likelihood fourier components
                try:
                    cf = sl.cho_factor(Sigma)
                    expval2 = sl.cho_solve(cf, dd)
                    logdet_Sigma += np.sum(2 * np.log(np.diag(cf[0])))
                except np.linalg.LinAlgError:
                    raise ValueError("ERROR: Sigma singular according to SVD")

                loglike += -0.5 * logdet_Sigma + 0.5 * (np.dot(dd, expval2))

                # increment frequency counter
                nfref += nf

        if not incCorrelations:
            loglike += -0.5 * self.logdetPhi

        # compute the red noise, DMV and GWB terms in the log likelihood
        if incCorrelations:
            # compute sigma
            Sigma = sl.block_diag(*FGGNGGF) + self.Phiinv

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Sigma)
                expval2 = sl.cho_solve(cf, d)
                logdet_Sigma = np.sum(2 * np.log(np.diag(cf[0])))
            except np.linalg.LinAlgError:
                print 'Error'
                U, s, Vh = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                logdet_Sigma = np.sum(np.log(s))
                sinv = 1 / s
                sinv[s / s[0] < 1e-15] = 0
                expval2 = np.dot(Vh.T * sinv, U.T, d)

            loglike += -0.5 * \
                (self.logdetPhi + logdet_Sigma) + 0.5 * (np.dot(d, expval2))

        return loglike

    """
    Older version of mark2 likelihood. Does not include option for multiple pulsars

    """

    def mark2LogLikelihood_old(self, parameters):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        UGGNGGU = []
        for ct, p in enumerate(self.psr):

            if p.twoComponentNoise:

                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component
                # basis
                if ct == 0:
                    d = np.dot(p.AGU.T, p.AGr / p.Nwvec)
                else:
                    d = np.append(d, np.dot(p.AGU.T, p.AGr / p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1 / p.Nwvec) * p.AGU.T).T
                UGGNGGU.append(np.dot(p.AGU.T, right))

                # log determinant of G^TNG
                logdet_N = np.sum(np.log(p.Nwvec))

                # triple product in likelihood function
                rGGNGGr = np.sum(p.AGr ** 2 / p.Nwvec)

            else:

                # G(G^TNG)^{-1}G^T = N^{-1} -
                # N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0 / p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiU = ((1.0 / p.Nvec) * p.Umat.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiU = np.dot(NiGc.T, p.Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(
                        np.log(p.Nvec)) + 2 * np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                # F^TG(G^TNG)^{-1}G^T\delta t
                if ct == 0:
                    d = np.dot(p.Umat.T, Nir - NiGcNiGcr)
                else:
                    d = np.append(d, np.dot(p.Umat.T, Nir - NiGcNiGcr))

                # triple product in likelihood function
                rGGNGGr = np.dot(p.detresiduals, Nir) - np.dot(GcNir, GcNiGcr)

                # compute F^TG(G^TNG)^{-1}G^TF
                UGGNGGU.append(
                    np.dot(NiU.T, p.Umat) - np.dot(GcNiU.T, GcNiGcU))

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rGGNGGr)

        # cheat for now
        # TODO: make this more general
        if self.npsr == 1:
            Phi0 = np.diag(1 / np.diag(self.Phiinv))
            UPhiU = np.dot(self.psr[0].UtFF, np.dot(Phi0, self.psr[0].UtFF.T))
            Phi = UPhiU + np.diag(self.psr[0].Qamp)

            try:
                cf = sl.cho_factor(Phi)
                self.logdetPhi = 2 * np.sum(np.log(np.diag(cf[0])))
                self.Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                print 'ERROR: Cholesky Failed when inverting Phi0'
                print parameters
                #U, s, Vh = sl.svd(Phi)
                # if not np.all(s > 0):
                # print "ERROR: Sigma singular according to SVD when inverting
                # Phi0"
                return -np.inf
                #raise ValueError("ERROR: Phi singular according to SVD")
                #self.logdetPhi = np.sum(np.log(s))
                #self.Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

        else:
            raise ValueError(
                "ERROR: Have not yet implemented jitter for multiple pulsars")

        # compute the red noise, DMV and GWB terms in the log likelihood

        # compute sigma
        Sigma = sl.block_diag(*UGGNGGU) + self.Phiinv

        # cholesky decomp for second term in exponential
        try:
            cf = sl.cho_factor(Sigma)
            logdet_Sigma = 2 * np.sum(np.log(np.diag(cf[0])))
            expval2 = sl.cho_solve(cf, d)
        except np.linalg.LinAlgError:
            print 'Cholesky failed when inverting Sigma'
            print parameters
            #U, s, Vh = sl.svd(Sigma)
            # if not np.all(s > 0):
            # print "ERROR: Sigma singular according to SVD when inverting
            # Sigma"
            return -np.inf
            #raise ValueError("ERROR: Sigma singular according to SVD")
            logdet_Sigma = np.sum(np.log(s))
            expval2 = np.dot(Vh.T, np.dot(np.diag(1.0 / s), np.dot(U.T, d)))

        loglike += -0.5 * (self.logdetPhi + logdet_Sigma) + \
            0.5 * (np.dot(d, expval2))

        return loglike

    """
    mark 2 log likelihood. Note that this is not the same as mark1 in piccard

    EFAC + EQUAD + Jitter +Red noise + DMV + GWs

    No frequency lines

    Uses Woodbury lemma

    """

    def mark2LogLikelihood(self, parameters, incCorrelations=True):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # frequency lines
        self.updateSpectralLines(parameters)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        UGGNGGU = []
        FJ = []
        FJF = []
        for ct, p in enumerate(self.psr):

            if p.twoComponentNoise:

                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component
                # basis
                if ct == 0:
                    d = np.dot(p.AGU.T, p.AGr / p.Nwvec)
                else:
                    d = np.append(d, np.dot(p.AGU.T, p.AGr / p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1 / p.Nwvec) * p.AGU.T).T
                UGGNGGU.append(np.dot(p.AGU.T, right))

                # log determinant of G^TNG
                logdet_N = np.sum(np.log(p.Nwvec))

                # triple product in likelihood function
                rGGNGGr = np.sum(p.AGr ** 2 / p.Nwvec)

            else:

                # G(G^TNG)^{-1}G^T = N^{-1} -
                # N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0 / p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiU = ((1.0 / p.Nvec) * p.Umat.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiU = np.dot(NiGc.T, p.Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(
                        np.log(p.Nvec)) + 2 * np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                # F^TG(G^TNG)^{-1}G^T\delta t
                if ct == 0:
                    d = np.dot(p.Umat.T, Nir - NiGcNiGcr)
                else:
                    d = np.append(d, np.dot(p.Umat.T, Nir - NiGcNiGcr))

                # triple product in likelihood function
                rGGNGGr = np.dot(p.detresiduals, Nir) - np.dot(GcNir, GcNiGcr)

                # compute F^TG(G^TNG)^{-1}G^TF
                UGGNGGU.append(
                    np.dot(NiU.T, p.Umat) - np.dot(GcNiU.T, GcNiGcU))

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rGGNGGr)

            # keep track of jitter terms needed later
            if self.npsr > 1 and incCorrelations:
                if ct == 0:
                    Jinv = 1 / p.Qamp
                else:
                    Jinv = np.append(Jinv, 1 / p.Qamp)

                FJ.append(p.UtFF.T * 1 / p.Qamp)
                FJF.append(np.dot(FJ[ct], p.UtFF))

        # if only using one pulsar
        if self.npsr == 1 or not(incCorrelations):
            logdetPhi = 0
            tmp = []
            for ct, p in enumerate(self.psr):
                Phi0 = np.diag(10 ** p.kappa_tot + self.gwamp)
                UPhiU = np.dot(p.UtFF, np.dot(Phi0, p.UtFF.T))
                Phi = UPhiU + np.diag(p.Qamp)

                try:
                    cf = sl.cho_factor(Phi)
                    logdetPhi += 2 * np.sum(np.log(np.diag(cf[0])))
                    tmp.append(sl.cho_solve(cf, np.identity(Phi.shape[0])))
                except np.linalg.LinAlgError:
                    # print 'Cholesky failed when inverting phi'
                    #U, s, Vh = sl.svd(Phi)
                    # if not np.all(s > 0):
                    return -np.inf
                    #raise ValueError("ERROR: Phi singular according to SVD")
                    #logdetPhi = np.sum(np.log(s))
                    #Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

            # block diagonal matrix
            Phiinv = sl.block_diag(*tmp)

        else:

            Phi0 = self.Phiinv + sl.block_diag(*FJF)
            logdet_J = np.sum(np.log(1 / Jinv))

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Phi0)
                logdet_Phi0 = 2 * np.sum(np.log(np.diag(cf[0])))
                PhiinvFJ = sl.cho_solve(cf, sl.block_diag(*FJ))
            except np.linalg.LinAlgError:
                print 'Cholesky Failed when inverting Phi0'
                return -np.inf
                #U, s, Vh = sl.svd(Phi0)
                # if not np.all(s > 0):
                #    return -np.inf
                #raise ValueError("ERROR: Sigma singular according to SVD")
                #logdet_Phi0 = np.sum(np.log(s))
                #PhiinvFJ = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, sl.block_diag(*FJ))))

            # get new Phiinv
            Phiinv = -np.dot(sl.block_diag(*FJ).T, PhiinvFJ)
            di = np.diag_indices(len(Jinv))
            Phiinv[di] += Jinv
            logdetPhi = self.logdetPhi + logdet_Phi0 + logdet_J

        # compute the red noise, DMV and GWB terms in the log likelihood

        # compute sigma
        Sigma = sl.block_diag(*UGGNGGU) + Phiinv

        # cholesky decomp for second term in exponential
        try:
            cf = sl.cho_factor(Sigma)
            logdet_Sigma = 2 * np.sum(np.log(np.diag(cf[0])))
            expval2 = sl.cho_solve(cf, d)
        except np.linalg.LinAlgError:
            # print 'Cholesky Failed when inverting Sigma'
            return -np.inf
            # return -np.inf
            #U, s, Vh = sl.svd(Sigma)
            # if not np.all(s > 0):
            # return -np.inf
            #    raise ValueError("ERROR: Sigma singular according to SVD")
            #logdet_Sigma = np.sum(np.log(s))
            #expval2 = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, d)))

        loglike += -0.5 * (logdetPhi + logdet_Sigma) + \
            0.5 * (np.dot(d, expval2))

        return loglike

    """
    mark 2 fixed noise likelihood (test)

    EFAC + EQUAD + Jitter +Red noise + DMV + GWs

    No frequency lines

    Uses Woodbury lemma

    """

    def mark2LogLikelihood_fixedNoise(
            self, parameters, incCorrelations=True, setup=False):

        loglike = 0

        # set pulsar white noise parameters
        if setup:
            self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        if setup:
            self.UGGNGGU = []
            self.logdet_N = []
            self.rGGNGGr = []
            self.logdet_N = []
            FJ = []
            FJF = []
            for ct, p in enumerate(self.psr):

                if p.twoComponentNoise:

                    # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two
                    # component basis
                    if ct == 0:
                        self.d = np.dot(p.AGU.T, p.AGr / p.Nwvec)
                    else:
                        self.d = np.append(
                            self.d, np.dot(p.AGU.T, p.AGr / p.Nwvec))

                    # compute F^TG(G^TNG)^{-1}G^TF
                    right = ((1 / p.Nwvec) * p.AGU.T).T
                    self.UGGNGGU.append(np.dot(p.AGU.T, right))

                    # log determinant of G^TNG
                    self.logdet_N.append(np.sum(np.log(p.Nwvec)))

                    # triple product in likelihood function
                    self.rGGNGGr.append(np.sum(p.AGr ** 2 / p.Nwvec))

                else:

                    # G(G^TNG)^{-1}G^T = N^{-1} -
                    # N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                    Nir = p.detresiduals / p.Nvec
                    NiGc = ((1.0 / p.Nvec) * p.Hcmat.T).T
                    GcNiGc = np.dot(p.Hcmat.T, NiGc)
                    NiU = ((1.0 / p.Nvec) * p.Umat.T).T
                    GcNir = np.dot(NiGc.T, p.detresiduals)
                    GcNiU = np.dot(NiGc.T, p.Umat)

                    try:
                        cf = sl.cho_factor(GcNiGc)
                        self.logdet_N.append(np.sum(np.log(p.Nvec)) +
                                             2 * np.sum(np.log(np.diag(cf[0]))))
                        GcNiGcr = sl.cho_solve(cf, GcNir)
                        GcNiGcU = sl.cho_solve(cf, GcNiU)
                        NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                    except np.linalg.LinAlgError:
                        print "MAJOR ERROR"

                    # F^TG(G^TNG)^{-1}G^T\delta t
                    if ct == 0:
                        self.d = np.dot(p.Umat.T, Nir - NiGcNiGcr)
                    else:
                        self.d = np.append(
                            self.d, np.dot(p.Umat.T, Nir - NiGcNiGcr))

                    # triple product in likelihood function
                    self.rGGNGGr.append(
                        np.dot(p.detresiduals, Nir) - np.dot(GcNir, GcNiGcr))

                    # compute F^TG(G^TNG)^{-1}G^TF
                    self.UGGNGGU.append(
                        np.dot(NiU.T, p.Umat) - np.dot(GcNiU.T, GcNiGcU))

                # first component of likelihood function
                loglike += -0.5 * (self.logdet_N[ct] + self.rGGNGGr[ct])

                # keep track of jitter terms needed later
                if self.npsr > 1 and incCorrelations:
                    if ct == 0:
                        Jinv = 1 / p.Qamp
                    else:
                        Jinv = np.append(Jinv, 1 / p.Qamp)

                    FJ.append(p.UtF.T * 1 / p.Qamp)
                    FJF.append(np.dot(FJ[ct], p.UtF))

        else:
            for ct, p in enumerate(self.psr):
                loglike += -0.5 * (self.logdet_N[ct] + self.rGGNGGr[ct])

        # if only using one pulsar
        if self.npsr == 1 or not(incCorrelations):
            logdetPhi = 0
            tmp = []
            for ct, p in enumerate(self.psr):
                Phi0 = np.diag(10 ** p.kappa_tot + self.gwamp)
                UPhiU = np.dot(p.UtF, np.dot(Phi0, p.UtF.T))
                Phi = UPhiU + np.diag(p.Qamp)

                try:
                    cf = sl.cho_factor(Phi)
                    logdetPhi += 2 * np.sum(np.log(np.diag(cf[0])))
                    tmp.append(sl.cho_solve(cf, np.identity(Phi.shape[0])))
                except np.linalg.LinAlgError:
                    # print 'Cholesky failed when inverting phi'
                    #U, s, Vh = sl.svd(Phi)
                    # if not np.all(s > 0):
                    return -np.inf
                    #raise ValueError("ERROR: Phi singular according to SVD")
                    #logdetPhi = np.sum(np.log(s))
                    #Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

            # block diagonal matrix
            Phiinv = sl.block_diag(*tmp)

        else:

            Phi0 = self.Phiinv + sl.block_diag(*FJF)
            logdet_J = np.sum(np.log(1 / Jinv))

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Phi0)
                logdet_Phi0 = 2 * np.sum(np.log(np.diag(cf[0])))
                PhiinvFJ = sl.cho_solve(cf, sl.block_diag(*FJ))
            except np.linalg.LinAlgError:
                print 'Cholesky Failed when inverting Phi0'
                return -np.inf
                #U, s, Vh = sl.svd(Phi0)
                # if not np.all(s > 0):
                #    return -np.inf
                #raise ValueError("ERROR: Sigma singular according to SVD")
                #logdet_Phi0 = np.sum(np.log(s))
                #PhiinvFJ = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, sl.block_diag(*FJ))))

            # get new Phiinv
            Phiinv = -np.dot(sl.block_diag(*FJ).T, PhiinvFJ)
            di = np.diag_indices(len(Jinv))
            Phiinv[di] += Jinv
            logdetPhi = self.logdetPhi + logdet_Phi0 + logdet_J

        # compute the red noise, DMV and GWB terms in the log likelihood

        # compute sigma
        Sigma = sl.block_diag(*self.UGGNGGU) + Phiinv

        # cholesky decomp for second term in exponential
        try:
            cf = sl.cho_factor(Sigma)
            logdet_Sigma = 2 * np.sum(np.log(np.diag(cf[0])))
            expval2 = sl.cho_solve(cf, self.d)
        except np.linalg.LinAlgError:
            # print 'Cholesky Failed when inverting Sigma'
            return -np.inf
            # return -np.inf
            #U, s, Vh = sl.svd(Sigma)
            # if not np.all(s > 0):
            # return -np.inf
            #    raise ValueError("ERROR: Sigma singular according to SVD")
            #logdet_Sigma = np.sum(np.log(s))
            #expval2 = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, d)))

        loglike += -0.5 * (logdetPhi + logdet_Sigma) + \
            0.5 * (np.dot(self.d, expval2))

        return loglike

    """
    mark 3 log likelihood. Note that this is not the same as mark3 in piccard

    Single pulsar test of daily average likelihood function

    Under Construction

    """

    def mark3LogLikelihood(self, parameters):

        tstart = time.time()
        tstart_tot = tstart

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, constructPhi=True)
        Phi = np.diag(self.Phi)

        # print 'Setting noise = {0} s'.format(time.time()-tstart)

        tstart = time.time()

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # construct covariance matrix
        #red = np.dot(self.psr[0].Ftot, (np.diag(self.Phi)*self.psr[0].Ftot).T)
        #cov = red + np.diag(self.psr[0].Nvec)

        QCQ = np.dot(self.psr[0].QRF, (Phi * self.psr[0].QRF).T)
        QCQ += np.dot(self.psr[0].QR, (self.psr[0].Nvec * self.psr[0].QR).T)

        # print 'Constructing Covariance = {0} s'.format(time.time()-tstart)

        tstart = time.time()

        # svd
        u, s, v = sl.svd(QCQ)
        ind = s / s[0] < 1e-15 * len(s)
        sinv = 1 / s
        sinv[ind] = 0.0

        logdetCov = np.sum(np.log(s[~ind]))
        invCov = np.dot(v.T, np.dot(np.diag(sinv), u.T))

        # print 'Computing Inverse = {0} s'.format(time.time()-tstart)

        tstart = time.time()

        loglike = -0.5 * \
            (logdetCov +
             np.dot(self.psr[0].QRr, np.dot(invCov, self.psr[0].QRr)))

        # print 'Matrix vector = {0} s'.format(time.time()-tstart)

        # print 'Total time = {0} s\n'.format(time.time()-tstart_tot)

        return loglike

    """
    mark 4 log likelihood. Full timing model with Fourier mode marginalization

    EFAC + EQUAD + Red noise + DMV + TM

    No jitter or frequency lines, only works with single pulsar

    """

    def mark4LogLikelihood(self, parameters, incCorrelations=True):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        FNF = []
        for ct, p in enumerate(self.psr):

            if ct == 0:
                d = np.dot(p.Ftot.T, p.detresiduals / p.Nvec)
            else:
                d = np.append(d, np.dot(p.Ftot.T, p.detresiduals / p.Nvec))

            # log determinant of N
            logdet_N = np.sum(np.log(p.Nvec))

            # triple product
            rNr = np.dot(p.detresiduals, p.detresiduals / p.Nvec)

            # compute F^TN^{-1}F
            right = ((1 / p.Nvec) * p.Ftot.T).T
            FNF.append(np.dot(p.Ftot.T, right))

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rNr)

        # compute the red noise, DMV and GWB terms in the log likelihood

        # compute sigma
        Sigma = sl.block_diag(*FNF) + self.Phiinv

        # cholesky decomp for second term in exponential
        try:
            cf = sl.cho_factor(Sigma)
            expval2 = sl.cho_solve(cf, d)
            logdet_Sigma = np.sum(2 * np.log(np.diag(cf[0])))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            logdet_Sigma = np.sum(np.log(s))
            expval2 = np.dot(Vh.T, np.dot(np.diag(1.0 / s), np.dot(U.T, d)))

        loglike += -0.5 * (self.logdetPhi + logdet_Sigma) + \
            0.5 * (np.dot(d, expval2))

        return loglike

    """
    Mark 5 Log Likelihood

    Uses non-gaussian framework of Lentati et al, 2014 (arXiv:1405.2460)

    Only includes EFAC and EQUAD for now, also only works with 1 pulsar

    """

    def mark5LogLikelihood(self, parameters, incCorrelations=True):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # get non-gaussian amplitudes
        if self.haveNonGaussian:
            self.getNonGaussianComponents(parameters)

        # shorthand
        if self.npsr > 1:
            raise ValueError(
                'ERROR: Mark 5 likelihood only valid with 1 pulsar')
        p = self.psr[0]

        # get gaussian part
        loglike += -0.5 * np.dot(p.detresiduals, p.detresiduals / p.Nvec)

        # non-gaussian part
        hermargs = p.detresiduals / np.sqrt(2 * p.Nvec)

        hermcoeff = []
        if np.abs(np.sum(p.alphacoeff[1:])) > 1:
            return -np.inf
        for ii in range(p.nalpha):
            hermcoeff.append(p.alphacoeff[ii] / np.sqrt(2 ** ii * ss.gamma(ii + 1)
                                                        * np.sqrt(2 * np.pi * p.Nvec)))

        # evaluate hermite polynomial sums
        hval = hermval(hermargs, np.array(hermcoeff))[0]
        #tmp = np.sum(hval, axis=0)
        loglike += 2 * np.sum(np.log(np.abs(hval)))

        return loglike

    """
    mark 6 log likelihood. Note that this is not the same as mark6 in piccard

    EFAC + EQUAD + Red noise + DMV + GWs

    No jitter or frequency lines

    Uses Woodbury lemma and "T" matrix formalism

    """

    def mark6LogLikelihood(self, parameters, incCorrelations=False,
                           incJitter=False, varyNoise=True, fixWhite=False):

        loglike = 0

        if varyNoise:

            # set pulsar white noise parameters
            if not fixWhite:
                self.setPsrNoise(parameters, incJitter=False)

            self.updateTmatrix(parameters)

            # set red noise, DM and GW parameters
            try:
                self.constructPhiMatrix(parameters, incCorrelations=incCorrelations,
                                        incTM=True, incJitter=incJitter)
            except np.linalg.LinAlgError:
                return -np.inf


        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        nfref = 0
        if varyNoise:
            self.logdet_Sigma = 0
        for ct, p in enumerate(self.psr):

            nf = p.Ttmat.shape[1]

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(
                    np.isinf(p.detresiduals)):
                return -np.inf

            # equivalent to T^T N^{-1} \delta t
            if ct == 0:
                d = np.dot(p.Ttmat.T, p.detresiduals / p.Nvec)
            else:
                d = np.append(d, np.dot(p.Ttmat.T, p.detresiduals / p.Nvec))

            if varyNoise:
                # compute T^T N^{-1} T
                if not fixWhite:
                    right = ((1 / p.Nvec) * p.Ttmat.T).T
                    self.TNT[nfref:(nfref+nf), nfref:(nfref+nf)] = \
                            np.dot(p.Ttmat.T, right)

            # log determinant of G^TNG
            logdet_N = np.sum(np.log(p.Nvec))

            # triple product in likelihood function
            rNr = np.sum(p.detresiduals ** 2 / p.Nvec)

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rNr)


            # calculate red noise piece
            if not incCorrelations:

                # compute sigma
                dd = d[nfref:(nfref + nf)]

                if varyNoise:
                    self.Sigma[nfref:(nfref + nf), nfref:(nfref + nf)] = \
                        self.TNT[nfref:(nfref + nf), nfref:(nfref + nf)] + \
                        self.Phiinv[nfref:(nfref + nf), nfref:(nfref + nf)]

                # cholesky decomp for maximum likelihood fourier components
                try:
                    cf = sl.cho_factor(
                        self.Sigma[nfref:(nfref + nf), nfref:(nfref + nf)])
                    expval2 = sl.cho_solve(cf, dd)
                    if varyNoise:
                        self.logdet_Sigma += np.sum(2 * np.log(np.diag(cf[0])))
                except np.linalg.LinAlgError:
                    #raise ValueError("ERROR: Sigma singular according to SVD")
                    return -np.inf

                loglike += 0.5 * (np.dot(dd, expval2))

                # increment frequency counter
                nfref += nf

        if not incCorrelations:
            loglike += -0.5 * (self.logdetPhi + self.logdet_Sigma)

        # compute the red noise, DMV and GWB terms in the log likelihood
        if incCorrelations:

            if varyNoise:
                # compute sigma
                self.Sigma = self.TNT + self.Phiinv

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(self.Sigma)
                expval2 = sl.cho_solve(cf, d)
                if varyNoise:
                    self.logdet_Sigma = np.sum(2 * np.log(np.diag(cf[0])))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(self.Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                expval2 = np.dot(
                    Vh.T, np.dot(np.diag(1.0 / s), np.dot(U.T, d)))
                if varyNoise:
                    self.logdet_Sigma = np.sum(np.log(s))

            loglike += -0.5 * \
                (self.logdetPhi + self.logdet_Sigma) + \
                0.5 * (np.dot(d, expval2))

        return loglike

    """
    mark 7 log likelihood. Full unmarginalized likelihood

    EFAC + EQUAD + Red noise + DMV + TM + GWs

    No jitter yet or frequency lines yet

    """

    def mark7LogLikelihood(self, parameters, incCorrelations=True):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations)

        # frequency lines
        # self.updateSpectralLines(parameters)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        rNr = []
        for ct, p in enumerate(self.psr):

            # log determinant of N
            logdet_N = np.sum(np.log(p.Nvec))

            # triple product
            rNr = np.dot(p.detresiduals, p.detresiduals / p.Nvec)

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rNr)

        # Fourier amplitude prior
        a = self.setFourierAmplitudes(parameters)
        if incCorrelations:
            aPhia = np.dot(a.T, np.dot(self.Phiinv, a))
        else:
            aPhia = np.dot(a.T, (np.diag(self.Phiinv) * a))

        loglike += -0.5 * (self.logdetPhi + aPhia)

        return loglike

    def mark8LogLikelihood(
            self, parameters, incCorrelations=False, incJitter=False):
        """
        Test likelihood only including timing model and no noise
        """

        loglike = 0
        print self.mark8Gradient(parameters)

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        self.updateTmatrix(parameters)

        # compute the white noise terms in the log likelihood
        for ct, p in enumerate(self.psr):

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(
                    np.isinf(p.detresiduals)):
                return -np.inf

            d = np.dot(p.Tmat.T, p.detresiduals / p.Nvec)

            loglike += -np.dot(p.detresiduals ** 2, 1 / p.Nvec)
            loglike += np.dot(d, np.dot(p.TNTinv, d))

        return loglike

    """
    mark 9 log likelihood. Note that this is not the same as mark6 in piccard

    EFAC + EQUAD + Red noise + DMV + GWs

    Puts jitter parameter into noise vector

    Uses Woodbury lemma and "T" matrix formalism

    """

    def mark9LogLikelihood(self, parameters, incCorrelations=False, varyNoise=True,
                           fixWhite=False):

        loglike = 0

        if varyNoise:

            # set pulsar white noise parameters
            if not fixWhite:
                self.setPsrNoise(parameters, incJitter=False)

            self.updateTmatrix(parameters)

            # set red noise, DM and GW parameters
            self.constructPhiMatrix(parameters, incCorrelations=incCorrelations,
                                    incTM=True, incJitter=False)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        TNT = []
        nfref = 0
        if varyNoise:
            self.logdet_Sigma = 0
        for ct, p in enumerate(self.psr):

            nf = p.Ttmat.shape[1]

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(
                    np.isinf(p.detresiduals)):
                return -np.inf

            # equivalent to T^T N^{-1} \delta t
            if ct == 0:
                d = np.dot(p.Ttmat.T, PALutils.python_block_shermor_0D(
                    p.detresiduals, p.Nvec, p.Qamp, p.Uinds))
            else:
                d = np.append(d, np.dot(p.Ttmat.T, PALutils.python_block_shermor_0D(
                    p.detresiduals, p.Nvec, p.Qamp, p.Uinds)))

            if varyNoise:
                # compute T^T N^{-1} T
                if not fixWhite:
                    self.TNT[nfref:(nfref + nf), nfref:(nfref + nf)] = \
                        PALutils.python_block_shermor_2D(
                            p.Ttmat, p.Nvec, p.Qamp, p.Uinds)
                    # TNT.append( \
                    # PALutils.python_block_shermor_2D(p.Ttmat, p.Nvec, p.Qamp,
                    # p.Uinds))

            # triple product in likelihood function
            logdet_N, rNr = PALutils.python_block_shermor_1D(p.detresiduals,
                                                             p.Nvec, p.Qamp, p.Uinds)

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rNr) - 0.5 * \
                len(p.toas) * np.log(2 * np.pi)


            # calculate red noise piece
            if not incCorrelations:

                # compute sigma
                dd = d[nfref:(nfref + nf)]

                if varyNoise:
                    self.Sigma[nfref:(nfref + nf), nfref:(nfref + nf)] = \
                        self.TNT[nfref:(nfref + nf), nfref:(nfref + nf)] +\
                        self.Phiinv[nfref:(nfref + nf), nfref:(nfref + nf)]
                    # self.Sigma[nfref:(nfref+nf), nfref:(nfref+nf)] = \
                    #        TNT[ct] + \
                    #        self.Phiinv[nfref:(nfref+nf), nfref:(nfref+nf)]

                # cholesky decomp for maximum likelihood fourier components
                try:
                    cf = sl.cho_factor(
                        self.Sigma[nfref:(nfref + nf), nfref:(nfref + nf)])
                    expval2 = sl.cho_solve(cf, dd)
                    if varyNoise:
                        self.logdet_Sigma += np.sum(2 * np.log(np.diag(cf[0])))
                except np.linalg.LinAlgError:
                    return -np.inf
                    #raise ValueError("ERROR: Sigma singular according to SVD")

                loglike += 0.5 * (np.dot(dd, expval2))

                # increment frequency counter
                nfref += nf

        if not incCorrelations:
            loglike += -0.5 * (self.logdetPhi + self.logdet_Sigma)

        # compute the red noise, DMV and GWB terms in the log likelihood
        if incCorrelations:

            if varyNoise:
                # compute sigma
                self.Sigma = self.TNT + self.Phiinv

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(self.Sigma)
                expval2 = sl.cho_solve(cf, d)
                if varyNoise:
                    self.logdet_Sigma = np.sum(2 * np.log(np.diag(cf[0])))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(self.Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                expval2 = np.dot(
                    Vh.T, np.dot(np.diag(1.0 / s), np.dot(U.T, d)))
                if varyNoise:
                    self.logdet_Sigma = np.sum(np.log(s))

            loglike += -0.5 * \
                (self.logdetPhi + self.logdet_Sigma) + \
                0.5 * (np.dot(d, expval2))
        return loglike
    
    """
    mark 10 log likelihood. Note that this is not the same as mark6 in piccard

    EFAC + EQUAD + Red noise + DMV + GWs


    Uses Woodbury lemma and "T" matrix formalism with coarse-grained covariance
    matrix for red noise and jitter

    """

    def mark10LogLikelihood(self, parameters, incJitter=False, incCorrelations=False):

        loglike = 0


        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)
        
        # construct dense coarse grained covariance matrix
        try:
            self.construct_dense_cov_matrix(parameters, incJitter=incJitter)
        except np.linalg.LinAlgError:
            return -np.inf

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        TNT = []
        nfref = 0
        for ct, p in enumerate(self.psr):

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(
                    np.isinf(p.detresiduals)):
                return -np.inf

            # equivalent to T^T N^{-1} \delta t
            if ct == 0:
                d = np.dot(p.Tmat.T, p.detresiduals / p.Nvec)
            else:
                d = np.append(d, np.dot(p.Tmat.T, p.detresiduals / p.Nvec))

            # compute T^T N^{-1} T
            right = ((1 / p.Nvec) * p.Tmat.T).T
            TNT.append(np.dot(p.Tmat.T, right))

            # log determinant of G^TNG
            logdet_N = np.sum(np.log(p.Nvec))

            # triple product in likelihood function
            rNr = np.sum(p.detresiduals ** 2 / p.Nvec)

            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rNr)

            # calculate red noise piece
            if not incCorrelations:

                # compute sigma
                nf = p.Tmat.shape[1]
                dd = d[nfref:(nfref + nf)]

                self.Sigma[nfref:(nfref + nf), nfref:(nfref + nf)] = \
                    TNT[ct] + \
                    self.Phiinv[nfref:(nfref + nf), nfref:(nfref + nf)]

                # cholesky decomp for maximum likelihood fourier components
                try:
                    cf = sl.cho_factor(
                        self.Sigma[nfref:(nfref + nf), nfref:(nfref + nf)])
                    expval2 = sl.cho_solve(cf, dd)
                    self.logdet_Sigma = np.sum(2 * np.log(np.diag(cf[0])))
                except np.linalg.LinAlgError:
                    return -np.inf
                    raise ValueError("ERROR: Sigma singular according to SVD")

                loglike += 0.5 * (np.dot(dd, expval2))

                # increment frequency counter
                nfref += nf

        if not incCorrelations:
            loglike += -0.5 * (self.logdetPhi + self.logdet_Sigma)


        return loglike

    # compute F_p statistic
    def fpStat(self, f0):
        """
        Computes the Fp-statistic as defined in Ellis, Siemens, Creighton (2012)
        Assumes that noise values have already been set

        @param psr: List of pulsar object instances
        @param f0: Gravitational wave frequency

        @return: Value of the Fp statistic evaluated at f0

        """

        fstat = 0

        # define N vectors from Ellis et al, 2012 N_i=(x|A_i) for each pulsar
        N = np.zeros(2)
        M = np.zeros((2, 2))
        nfref = 0
        TNA = []
        for ii, p in enumerate(self.psr):
            nf = p.Ttmat.shape[1]
            ntoa = len(p.toas)

            # Define A vector
            A = np.zeros((2, ntoa))
            A[0, :] = 1 / f0 ** (1 / 3) * np.sin(2 * np.pi * f0 * p.toas)
            A[1, :] = 1 / f0 ** (1 / 3) * np.cos(2 * np.pi * f0 * p.toas)

            Sigma = self.Sigma[nfref:(nfref + nf), nfref:(nfref + nf)]
            ip1 = PALutils.innerProduct_rr(A[0, :], p.residuals,
                                           p.Nvec, p.Tmat, Sigma)
            ip2 = PALutils.innerProduct_rr(A[1, :], p.residuals,
                                           p.Nvec, p.Tmat, Sigma)
            N = np.array([ip1, ip2])

            # define M matrix M_ij=(A_i|A_j)
            for jj in range(2):
                for kk in range(2):
                    M[jj, kk] = PALutils.innerProduct_rr(A[jj, :], A[kk, :],
                                                         p.Nvec, p.Tmat, Sigma)

            # take inverse of M
            Minv = np.linalg.inv(M)
            fstat += 0.5 * np.dot(N, np.dot(Minv, N))

            nfref += nf

        # return F-statistic
        return fstat

    def mark8Gradient(self, parameters):

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        self.updateTmatrix(parameters)

        for ct, p in enumerate(self.psr):

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(
                    np.isinf(p.detresiduals)):
                return -np.inf

            d = np.dot(p.Tmat.T, p.detresiduals / p.Nvec)

            M = p.Mmat / p.norm
            term1 = np.dot(M.T, p.detresiduals / p.Nvec)
            TNTinvd = np.dot(p.TNTinv, d)
            term2 = np.dot(M.T / p.Nvec, np.dot(p.Tmat, TNTinvd))

        return (term1 - term2) * p.norm

    """
    Zero log likelihood for prior testing purposes
    """

    def zeroLogLikelihood(self, parameters, kwargs=None):

        return 0

    """
    Reconstruct ML signal from Tmatrix

    EFAC + EQUAD + Red noise + DMV + GWs

    No frequency lines

    Uses Woodbury lemma and "T" matrix formalism

    """

    def reconstructML_old(self, parameters, incCorrelations=False, incJitter=False):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations,
                                incTM=True, incJitter=incJitter)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        self.updateTmatrix(parameters)

        # compute the white noise terms in the log likelihood
        TNT = []
        nfref = 0
        ml_vals, ml_errs, chisq = [], [], []
        for ct, p in enumerate(self.psr):

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(
                    np.isinf(p.detresiduals)):
                return -np.inf

            # equivalent to T^T N^{-1} \delta t
            if ct == 0:
                d = np.dot(p.Ttmat.T, p.residuals / p.Nvec)
            else:
                d = np.append(d, np.dot(p.Ttmat.T, p.residuals / p.Nvec))

            # compute T^T N^{-1} T
            right = ((1 / p.Nvec) * p.Ttmat.T).T
            TNT.append(np.dot(p.Ttmat.T, right))

            # calculate red noise piece
            if not incCorrelations:

                # compute sigma
                nf = p.Ttmat.shape[1]
                Sigma = TNT[ct] + \
                    self.Phiinv[nfref:(nfref + nf), nfref:(nfref + nf)]
                dd = d[nfref:(nfref + nf)]

                # cholesky decomp for maximum likelihood fourier components
                try:
                    cf = sl.cho_factor(Sigma)
                    ml_vals.append(sl.cho_solve(cf, dd))
                    sigma_inv = sl.cho_solve(cf, np.eye(len(dd)))
                    # ml_errs.append(1/np.diag(Sigma))
                    ml_errs.append(sigma_inv)
                except np.linalg.LinAlgError:
                    return -np.inf

                #detresiduals = p.detresiduals - np.dot(p.Ttmat, ml_vals[-1])
                #d = np.dot(p.Ttmat.T, detresiduals/p.Nvec)

                #expval2 = sl.cho_solve(cf, d)

                # triple product
                #rNr = np.dot(detresiduals, detresiduals/p.Nvec)

                #chisq.append(rNr - np.dot(d, expval2))

                # increment frequency counter
                nfref += nf

        return ml_vals, ml_errs

    """
    Reconstruct ML signal from Tmatrix

    EFAC + EQUAD + Red noise + DMV + GWs

    No frequency lines

    Uses Woodbury lemma and "T" matrix formalism

    """

    def reconstructML(
            self, parameters, incCorrelations=False, incJitter=False):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations,
                                incTM=True, incJitter=incJitter)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        self.updateTmatrix(parameters)

        # compute the white noise terms in the log likelihood
        TNT = []
        nfref = 0
        ml_vals, ml_errs, chisq = [], [], []
        for ct, p in enumerate(self.psr):

            # get design matrix
            Mmat = p.Ttmat[:, :len(p.ptmdescription)]

            # scale design matrix
            norm = np.sqrt(np.sum(Mmat ** 2, axis=0))
            Mmat /= norm

            # matrix containing everything else
            Tmat = p.Ttmat[:, len(p.ptmdescription):]
            Phiinv = np.diag(np.diag(self.Phiinv)[len(p.ptmdescription):])

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(
                    np.isinf(p.detresiduals)):
                return -np.inf

            # equivalent to T^T N^{-1} \delta t
            if ct == 0:
                d = np.dot(Tmat.T, p.detresiduals / p.Nvec)
            else:
                d = np.append(d, np.dot(Tmat.T, p.detresiduals / p.Nvec))

            # compute T^T N^{-1} T
            right = ((1 / p.Nvec) * Tmat.T).T
            TNT.append(np.dot(Tmat.T, right))

            # compute sigma
            nf = Tmat.shape[1]
            Sigma = TNT[ct] + Phiinv[nfref:(nfref + nf), nfref:(nfref + nf)]
            dd = d[nfref:(nfref + nf)]

            # cholesky decomp for maximum likelihood fourier components
            cf = sl.cho_factor(Sigma)
            Sigmainvd = sl.cho_solve(cf, dd)
            Xdt = p.detresiduals / p.Nvec - np.dot(Tmat, Sigmainvd) / p.Nvec
            MXdt = np.dot(Mmat.T, Xdt)

            # construct covariance matrix
            right = ((1 / p.Nvec) * Mmat.T).T
            term1 = np.dot(Mmat.T, right)

            # second term
            TNM = np.dot(Tmat.T / p.Nvec, Mmat)
            right = sl.cho_solve(cf, TNM)
            term2 = np.dot(TNM.T, right)

            # get ML values
            try:
                cf = sl.cho_factor(term1 - term2)
                ml_vals.append(sl.cho_solve(cf, MXdt) / norm)
                sigma_inv = sl.cho_solve(cf, np.eye(len(MXdt)))
                ml_errs.append(((sigma_inv / norm).T * 1 / norm).T)
                expval2 = sl.cho_solve(cf, MXdt)
            except np.linalg.LinAlgError:
                print 'Cholesky decomposition failed'
                raise("ValueError")
                #u, s, v = np.linalg.svd(term1-term2+Phiinvb)
                #sinv = 1/s
                #sinv[s[0]/s>1e16] = 0
                #ml_vals.append(np.dot(u, np.dot((sinv*u).T, TbarXdt)))
                #ml_errs.append(np.dot(u, (sinv*u).T))
                #expval2 = np.dot(u, np.dot((sinv*u).T, TbarXdt))

            # increment frequency counter
            nfref += nf

        return ml_vals, ml_errs

    """
    Very simple uniform prior on all parameters

    """

    def mark1LogPrior(self, parameters):

        prior = 0
        if np.all(parameters >= self.pmin) and np.all(parameters <= self.pmax):
            prior += -np.sum(np.log(self.pmax - self.pmin))

        else:
            prior += -np.inf

        return prior

    """
    Very simple uniform prior on all parameters except flag in GW amplitude

    """

    def mark2LogPrior(self, parameters):

        prior = 0
        if np.all(parameters >= self.pmin) and np.all(parameters <= self.pmax):
            prior += -np.sum(np.log(self.pmax - self.pmin))

        else:
            prior += -np.inf

        # TODO:find better way of finding the amplitude
        for ss, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind + npars]

            if sig['corr'] == 'gr':
                prior += np.log(10 ** sparameters[0])

        return prior

    """
    Very simple uniform prior on all amplitudes, can also include flat in
    Amplitudes of red noise
    """

    def mark3LogPrior(self, parameters):
        prior = 0
        #for ct, pp in enumerate(parameters):
        #   print pp, self.pmin[ct], pp >= self.pmin[ct]
        #   print pp, self.pmin[ct], pp <= self.pmax[ct]
        if np.all(parameters >= self.pmin) and np.all(parameters <= self.pmax):
            prior += -np.sum(np.log(self.pmax - self.pmin))

        else:
            prior += -np.inf

        # TODO:find better way of finding the amplitude
        for ss, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind + npars]

            if sig['corr'] == 'gr' and sig['stype'] == 'powerlaw':
                if sig['prior'][0] == 'uniform':
                    prior += np.log(10 ** sparameters[0])
                elif sig['prior'][0] == 'sesana':
                    m = -15
                    s = 0.22
                    logA = sparameters[0]
                    prior += -0.5 * \
                        (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
                elif sig['prior'][0] == 'mcwilliams':
                    m = np.log10(4.1e-15)
                    s = 0.26
                    logA = sparameters[0]
                    prior += -0.5 * \
                        (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)

            if sig['corr'] == 'gr' and sig['stype'] == 'turnover':
                if sig['prior'][0] == 'sesana':
                    m = -15
                    s = 0.22
                    logA = sparameters[0]
                    prior += -0.5 * \
                        (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
                elif sig['prior'][0] == 'mcwilliams':
                    m = np.log10(4.1e-15)
                    s = 0.26
                    logA = sparameters[0]
                    prior += -0.5 * \
                        (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
                elif sig['prior'][0] == 'uniform':
                    prior += np.log(10 ** sparameters[0])

            if sig['corr'] == 'gr' and sig['stype'] == 'spectrum':
                if np.any(np.array(sig['prior']) == 'uniform'):
                    idx = np.array(sig['prior']) == 'uniform'
                    prior += np.sum(np.log(10 ** sparameters[idx]))

            if sig['stype'] == 'spectrum' and sig['corr'] == 'single':
                if np.any(np.array(sig['prior']) == 'uniform'):
                    idx = np.array(sig['prior']) == 'uniform'
                    prior += np.sum(np.log(10 ** sparameters[idx]))

                # cheater prior
                sig_data = self.psr[psrind].sig_data * 10
                sig_red = np.sqrt(np.sum(10 ** sparameters))
                # print sig_red, sig_data
                #if sig_red > sig_data:
                #    prior += -np.inf

            if sig['corr'] == 'gr' and sig['stype'] == 'spectrum':
                if np.any(np.array(sig['prior']) == 'sqrt'):
                    idx = np.array(sig['prior']) == 'sqrt'
                    prior += np.sum(np.log(10 ** (sparameters[idx] / 2)))

            if sig['stype'] == 'spectrum' and sig['corr'] == 'single':
                if np.any(np.array(sig['prior']) == 'sqrt'):
                    idx = np.array(sig['prior']) == 'sqrt'
                    prior += np.sum(np.log(10 ** (sparameters[idx] / 2)))

                # cheater prior
                sig_data = self.psr[psrind].sig_data * 10
                sig_red = np.sqrt(np.sum(10 ** sparameters))
                # print sig_red, sig_data
                #if sig_red > sig_data:
                #    prior += -np.inf

            if sig['stype'] in ['powerlaw'] and sig['corr'] == 'single':
                if sig['bvary'][0]:
                    if sig['prior'][0] == 'uniform':
                        prior += np.log(10 ** sparameters[0])

                # cheater prior
                Amp = 10 ** sparameters[0]
                gam = sparameters[1]
                sig_data = self.psr[psrind].sig_data * 10
                if gam > 1:
                    sig_red = 2.05e-9 / np.sqrt(gam - 1) * (Amp / 1e-15) *\
                        (self.psr[psrind].Tmax / 3.16e7) ** ((gam - 1) / 2)
                else:
                    sig_red = 0
                #if sig_red > sig_data:
                #    prior += -np.inf

            if sig['corr'] in ['gr_sph']:
                if sig['stype'] == 'powerlaw':
                    clms = np.append(2 * np.sqrt(np.pi), sparameters[3:])
                elif sig['stype'] == 'spectrum':
                    clms = np.append(
                        2 * np.sqrt(np.pi), sparameters[int(self.npf[psrind] / 2):])
                prior += PALutils.PhysPrior(clms, self.harm_sky_vals)
                if sig['prior'][0] == 'uniform' and sig['stype'] == 'powerlaw':
                    prior += np.log(10 ** sparameters[0])

            # prior on ECC, EDOT and T0
            if sig['stype'] == 'lineartimingmodel' or sig[
                    'stype'] == 'nonlineartimingmodel':

                ecc, edot, t0, sini, a1, m2, pb, kin, stig, h3 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                pindex = 0
                for jj in range(sig['ntotpars']):
                    if sig['bvary'][jj]:
                        # get ECC
                        if sig['parid'][jj] == 'ECC':
                            ecc = sparameters[pindex]
                        elif sig['parid'][jj] == 'T0':
                            t0 = sparameters[pindex]
                        elif sig['parid'][jj] == 'EDOT':
                            edot = sparameters[pindex]
                        elif sig['parid'][jj] == 'SINI':
                            sini = sparameters[pindex]
                        elif sig['parid'][jj] == 'A1':
                            a1 = sparameters[pindex]
                        elif sig['parid'][jj] == 'PB':
                            pb = sparameters[pindex]
                        elif sig['parid'][jj] == 'M2':
                            m2 = sparameters[pindex]
                        elif sig['parid'][jj] == 'KIN':
                            kin = sparameters[pindex]
                        elif sig['parid'][jj] == 'H3':
                            h3 = sparameters[pindex]
                        elif sig['parid'][jj] == 'STIG':
                            stig = sparameters[pindex]

                        pindex += 1

                # check if including all three
                if ecc and edot and t0:
                    tt0 = (self.psr[psrind].toas - t0 * 86400)
                    check = ecc + edot * tt0

                    if np.any(check > 1) or np.any(check < 0):
                        prior += -np.inf

                # flat in cosi prior
                if sini:
                    prior += np.log(sini / np.sqrt(1 - sini ** 2))
                    #prior += 0

                if kin:
                    prior += np.log(np.abs(np.sin(kin * np.pi / 180)))

                # prior on pulsar mass [0,3]
                if sini and pb and m2 and a1:
                    Pb = pb*86400
                    X = a1*299.79e6/3e8
                    M2 = m2*4.9e-6
                    mp = ((sini*(Pb/2/np.pi)**(2./3)*M2/X)**(3./2) - M2)/4.9e-6
                    #extra = ((Pb/2/np.pi)**(2./3)*M2/X)**(3./2) * 1.5 * np.abs(sini)**(1/2)
                    #prior += np.log(extra)
                    ms = 1.4 * np.sqrt(2)
                    if mp < 0.5 or mp > 3:
                        prior += -1e10
                    #prior += np.log(mp/ms) - (mp/ms)**2

                if stig and pb and h3 and a1:
                    m2 = h3/stig**3 / 4.9e-6
                    sini = 2 * stig / (1+stig**2)
                    Pb = pb*86400
                    X = a1*299.79e6/3e8
                    M2 = m2*4.9e-6
                    mp = ((sini*(Pb/2/np.pi)**(2./3)*M2/X)**(3./2) - M2)/4.9e-6
                    #extra = ((Pb/2/np.pi)**(2./3)*M2/X)**(3./2) * 1.5 * np.abs(sini)**(1/2)
                    #prior += np.log(extra)
                    ms = 1.4 * np.sqrt(2)
                    if mp < 0.5 or mp > 3:
                        prior += -1e10
                    #prior += np.log(mp/ms) - (mp/ms)**2

            # CW parameters
            if sig['stype'] == 'cw':
                pindex = 0
                for jj in range(sig['ntotpars']):
                    if sig['bvary'][jj]:

                        if sig['prior'][jj] == 'cos':
                            prior += np.log(
                                np.abs(np.sin(sparameters[pindex])))
                        if sig['prior'][jj] == 'uniform' and \
                            (sig['parid'][jj] == 'logh' or \
                            sig['parid'][jj] == 'logq'):
                            #print 'In uniform {0}'.format(sig['parid'][jj])
                            prior += np.log(10 ** sparameters[pindex])

                        if sig['prior'][jj] == 'gaussian' and \
                                sig['mu'][jj] is not None and \
                                sig['sigma'][jj] is not None:
                            #print 'In gaussian {0}'.format(sig['parid'][jj])
                            mu = sig['mu'][jj]
                            sigma = sig['sigma'][jj]
                            #print 'In gaussian {0} {1} {2} {3}'.format(sig['parid'][jj],
                            #        mu, sigma, sparameters[pindex])
                            prior += -0.5 * (np.log(2 * np.pi * sigma**2) +
                                             (sparameters[pindex] - mu)**2 /
                                             sigma**2)

                    pindex += 1

            # pulsar distance prior
            if sig['stype'] == 'pulsardistance':
                pdist = sparameters
                m = self.psr[psrind].pdist
                s = self.psr[psrind].pdistErr
                prior += -0.5 * \
                    (np.log(2 * np.pi * s ** 2) + (m - pdist) ** 2 / s ** 2)

            # pulsar frequency prior
            if sig['stype'] == 'pulsarTerm':

                # get GW frequency
                signum = self.getSignalNumbersFromDict(
                    self.ptasignals, stype='cw', corr='gr')
                sig0 = self.ptasignals[signum]
                parind0 = sig0['parindex']
                npars0 = sig0['npars']
                sparameters2 = sig0['pstart'].copy()
                sparameters2[sig0['bvary']] = parameters[parind0:parind0 + npars0]
                lfgw = sparameters2[3]
                if sparameters[1] > lfgw:
                    prior += -np.inf


        return prior

    ##########################################################################

    # MCMC jump proposals

    # TODO: make one single proposal that can take stype as argument, will
    # have to change MCMC code...

    # red noise draws
    def drawFromRedNoisePrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='powerlaw',
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='powerlaw',
                                               corr='single')

        # which parameters to jump
        #ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary'][0]:

                # log prior
                if sig['prior'][0] == 'log':
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'][0] == 'uniform':
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0

                    #q[parind] = np.log10(np.random.uniform(10 ** self.pmin[parind],
                    #                                       10 ** self.pmax[parind]))
                    #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]

            # jump in spectral index if varying
            if sig['bvary'][1]:

                if sig['prior'][1] == 'uniform':
                    q[parind +
                        1] = np.random.uniform(self.pmin[parind + 1], self.pmax[parind + 1])
                    qxy += 0

                else:
                    q[parind + 1] = parameters[parind + 1]

        return q, qxy

    # DM noise draws
    def drawFromDMPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='dmpowerlaw',
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='dmpowerlaw',
                                               corr='single')

        # which parameters to jump
        #ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary'][0]:

                # log prior
                if sig['prior'][0] == 'log':
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'][0] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10 ** self.pmin[parind],
                                                           10 ** self.pmax[parind]))
                    qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]

            # jump in spectral index if varying
            if sig['bvary'][1]:

                if sig['prior'][1] == 'uniform':
                    q[parind +
                        1] = np.random.uniform(self.pmin[parind + 1], self.pmax[parind + 1])
                    qxy += 0

                else:
                    q[parind + 1] = parameters[parind + 1]

        return q, qxy

    # red noise draws
    def drawFromRedNoiseBandPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='powerlaw_band',
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='powerlaw_band',
                                               corr='single')

        # which parameters to jump
        #ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary'][0]:

                # log prior
                if sig['prior'][0] == 'log':
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'][0] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10 ** self.pmin[parind],
                                                           10 ** self.pmax[parind]))
                    qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]

            # jump in spectral index if varying
            if sig['bvary'][1]:

                if sig['prior'][1] == 'uniform':
                    q[parind +
                        1] = np.random.uniform(self.pmin[parind + 1], self.pmax[parind + 1])
                    qxy += 0

                else:
                    q[parind + 1] = parameters[parind + 1]

        return q, qxy

    # dm noise draws
    def drawFromDMNoiseBandPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='dmpowerlaw_band',
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='dmpowerlaw_band',
                                               corr='single')

        # which parameters to jump
        #ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary'][0]:

                # log prior
                if sig['prior'][0] == 'log':
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'][0] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10 ** self.pmin[parind],
                                                           10 ** self.pmax[parind]))
                    qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]

            # jump in spectral index if varying
            if sig['bvary'][1]:

                if sig['prior'][1] == 'uniform':
                    q[parind +
                        1] = np.random.uniform(self.pmin[parind + 1], self.pmax[parind + 1])
                    qxy += 0

                else:
                    q[parind + 1] = parameters[parind + 1]

        return q, qxy

        # red noise sepctrum draws
    def drawFromRedNoiseSpectrumPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='spectrum',
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='spectrum',
                                               corr='single')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            #for jj in range(npars):
            jj = np.random.randint(0, np.sum(sig['bvary']))
            if sig['bvary'][jj]:

                # log prior
                if sig['prior'][jj] == 'log':
                    q[parind + jj] = np.random.uniform(self.pmin[parind + jj],
                                                       self.pmax[parind + jj])
                    qxy += 0

                elif sig['prior'][jj] == 'uniform':
                    q[parind + jj] = np.log10(np.random.uniform(
                        10 ** self.pmin[parind + jj], 
                        10 ** self.pmax[parind + jj]))
                    qxy += np.log(10 ** parameters[parind + jj] 
                                  / 10 ** q[parind + jj])

                elif sig['prior'][jj] == 'sqrt':
                    q[parind + jj] = np.log10(np.random.uniform(
                        10 ** (self.pmin[parind + jj] / 2), 
                        10 ** (self.pmax[parind + jj] / 2)) ** 2)
                    qxy += np.log(10 ** (parameters[parind + jj] / 2) 
                                  / 10 ** (q[parind + jj] / 2))

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind + jj] = parameters[parind + jj]

        return q, qxy


    # red noise sepctrum draws
    def drawFromRedNoiseExtSpectrumPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals,
                                                       stype='ext_spectrum',
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals,
                                               stype='ext_spectrum',
                                               corr='single')


        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            #for jj in range(npars):
            jj = np.random.randint(0, np.sum(sig['bvary']))
            if sig['bvary'][jj]:

                # log prior
                if sig['prior'][jj] == 'log':
                    q[parind + jj] = np.random.uniform(self.pmin[parind + jj],
                                                       self.pmax[parind + jj])
                    qxy += 0

                elif sig['prior'][jj] == 'uniform':
                    q[parind + jj] = np.log10(np.random.uniform(
                        10 ** self.pmin[parind + jj], 
                        10 ** self.pmax[parind + jj]))
                    qxy += np.log(10 ** parameters[parind + jj] 
                                  / 10 ** q[parind + jj])

                elif sig['prior'][jj] == 'sqrt':
                    q[parind + jj] = np.log10(np.random.uniform(
                        10 ** (self.pmin[parind + jj] / 2), 
                        10 ** (self.pmax[parind + jj] / 2)) ** 2)
                    qxy += np.log(10 ** (parameters[parind + jj] / 2) 
                                  / 10 ** (q[parind + jj] / 2))

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind + jj] = parameters[parind + jj]

        return q, qxy

        # red noise sepctrum draws
    def drawFromGWBSpectrumPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = 1
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='spectrum',
                                               corr='gr')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            for jj in range(npars):
                if sig['bvary'][jj]:

                    # log prior
                    if sig['prior'][jj] == 'log':
                        q[parind + jj] = np.random.uniform(self.pmin[parind + jj],
                                                           self.pmax[parind + jj])
                        qxy += 0

                    elif sig['prior'][jj] == 'uniform':
                        q[parind + jj] = np.log10(np.random.uniform(10 ** self.pmin[parind + jj],
                                                                    10 ** self.pmax[parind + jj]))
                        qxy += np.log(10 **
                                      parameters[parind + jj] / 10 ** q[parind + jj])

                    elif sig['prior'][jj] == 'sqrt':
                        q[parind + jj] = np.log10(np.random.uniform(10 ** (self.pmin[parind + jj] / 2),
                                                                    10 ** (self.pmax[parind + jj] / 2)) ** 2)
                        qxy += np.log(10 **
                                      (parameters[parind + jj] / 2) / 10 ** (q[parind + jj] / 2))

                    else:
                        print 'Prior type not recognized for parameter'
                        q[parind + jj] = parameters[parind + jj]

        return q, qxy

    # GWB draws draws
    def drawFromGWBPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = 1
        signum = self.getSignalNumbersFromDict(
            self.ptasignals, stype='powerlaw', corr='gr')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary'][0]:

                # log prior
                if sig['prior'][0] == 'log':
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'][0] == 'uniform':
                    
                    # draw from log-uniform prior
                    # to get more samples in right range
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0
                    #q[parind] = np.log10(np.random.uniform(10 ** self.pmin[parind],
                    #                                       10 ** self.pmax[parind]))
                    #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

                elif sig['prior'][0] == 'sesana':
                    m = -15
                    s = 0.22
                    q[parind] = m + np.random.randn() * s
                    qxy -= (m - parameters[parind]) ** 2 / 2 / \
                        s ** 2 - (m - q[parind]) ** 2 / 2 / s ** 2
                elif sig['prior'][0] == 'mcwilliams':
                    m = np.log10(4.1e-15)
                    s = 0.26
                    q[parind] = m + np.random.randn() * s
                    qxy -= (m - parameters[parind]) ** 2 / 2 / \
                        s ** 2 - (m - q[parind]) ** 2 / 2 / s ** 2

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]

            # jump in spectral index if varying
            if sig['bvary'][1]:

                if sig['prior'][1] == 'uniform':
                    q[parind +
                        1] = np.random.uniform(self.pmin[parind + 1], self.pmax[parind + 1])
                    qxy += 0

                else:
                    q[parind + 1] = parameters[parind + 1]

        return q, qxy

    # aGWB draws draws
    def drawFromaGWBPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = 1
        signum = self.getSignalNumbersFromDict(
            self.ptasignals, stype='powerlaw', corr='gr_sph')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary'][0]:

                # log prior
                if sig['prior'][0] == 'log':
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'][0] == 'uniform':

                    # draw from log-uniform prior
                    # to get more samples in right range
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0
                    #q[parind] = np.log10(np.random.uniform(10 ** self.pmin[parind],
                    #                                       10 ** self.pmax[parind]))
                    #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

                elif sig['prior'][0] == 'sesana':
                    m = -15
                    s = 0.22
                    q[parind] = m + np.random.randn() * s
                    qxy -= (m - parameters[parind]) ** 2 / 2 / \
                        s ** 2 - (m - q[parind]) ** 2 / 2 / s ** 2
                elif sig['prior'][0] == 'mcwilliams':
                    m = np.log10(4.1e-15)
                    s = 0.26
                    q[parind] = m + np.random.randn() * s
                    qxy -= (m - parameters[parind]) ** 2 / 2 / \
                        s ** 2 - (m - q[parind]) ** 2 / 2 / s ** 2

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]

            # jump in spectral index if varying
            if sig['bvary'][1]:

                if sig['prior'][1] == 'uniform':
                    q[parind +
                        1] = np.random.uniform(self.pmin[parind + 1], self.pmax[parind + 1])
                    qxy += 0

                else:
                    q[parind + 1] = parameters[parind + 1]

            jj = np.random.randint(2, np.sum(sig['bvary']))
            if sig['bvary'][jj]:
                    q[parind + jj] = np.random.uniform(self.pmin[parind + jj],
                                                       self.pmax[parind + jj])
                    qxy += 0

        return q, qxy

    # GWB draws draws
    def drawFromGWBTurnoverPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = 1
        signum = self.getSignalNumbersFromDict(
            self.ptasignals, stype='turnover', corr='gr')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary'][0]:

                # amplitude priors
                if sig['prior'][0] == 'sesana':
                    m = -15
                    s = 0.22
                    q[parind] = m + np.random.randn() * s
                    qxy -= (m - parameters[parind]) ** 2 / 2 / \
                        s ** 2 - (m - q[parind]) ** 2 / 2 / s ** 2
                elif sig['prior'][0] == 'mcwilliams':
                    m = np.log10(4.1e-15)
                    s = 0.26
                    q[parind] = m + np.random.randn() * s
                    qxy -= (m - parameters[parind]) ** 2 / 2 / \
                        s ** 2 - (m - q[parind]) ** 2 / 2 / s ** 2
                elif sig['prior'][0] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10 ** self.pmin[parind],
                                                           10 ** self.pmax[parind]))
                    qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])
                elif sig['prior'][0] == 'log':
                    q[parind] = np.random.uniform(self.pmin[parind],
                                                  self.pmax[parind])

            # turnover frequency
            if sig['bvary'][2]:

                if sig['prior'][2] == 'uniform':
                    q[parind +
                        1] = np.random.uniform(self.pmin[parind + 1], self.pmax[parind + 1])
                    qxy += 0

                else:
                    q[parind + 1] = parameters[parind + 1]

            # kappa
            if sig['bvary'][3]:

                if sig['prior'][3] == 'uniform':
                    q[parind +
                        2] = np.random.uniform(self.pmin[parind + 2], self.pmax[parind + 2])
                    qxy += 0
                else:
                    q[parind + 3] = parameters[parind + 3]

            # beta
            if sig['bvary'][4]:

                if sig['prior'][4] == 'uniform':
                    q[parind +
                        3] = np.random.uniform(self.pmin[parind + 3], self.pmax[parind + 3])
                    qxy += 0

                else:
                    q[parind + 3] = parameters[parind + 3]

        return q, qxy

    def drawFromORFPrior(self, parameters, iter, beta):
        
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = 1
        signum = self.getSignalNumbersFromDict(
            self.ptasignals, stype='ORF', corr='gr')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        
        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            for jj in range(npars):
                if sig['bvary'][jj]:
                    q[parind + jj] = np.random.uniform(self.pmin[parind + jj],
                                                       self.pmax[parind + jj])
                    qxy += 0

        return q, qxy


    # CW draws from prior
    def drawFromCWPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        for ct, sig in enumerate(self.ptasignals):

            # CW parameters
            if sig['stype'] == 'cw':
                pindex = 0
                for jj in range(sig['ntotpars']):
                    if sig['bvary'][jj]:

                        parind = sig['parindex']

                        if sig['prior'][jj] == 'cos':
                            q[parind +
                                pindex] = np.arccos(np.random.uniform(-1, 1))
                            qxy += \
                                np.log(
                                    np.sin(parameters[parind + pindex]) /
                                    np.sin(q[parind + pindex]))
                        elif sig['prior'][jj] == 'uniform' and \
                        (sig['parid'][jj] == 'logh' or \
                         sig['parid'][jj] == 'logq'):
                            q[parind + pindex] = \
                                np.log10(np.random.uniform(10 ** self.pmin[parind + pindex],
                                                           10 ** self.pmax[parind + pindex]))
                            qxy += np.log(10 ** parameters[parind + pindex] / \
                                          10 ** q[parind + pindex])

                        elif sig['prior'][jj] == 'gaussian':
                            m = sig['mu'][jj]
                            s = sig['sigma'][jj]
                            q[parind+pindex] = m + np.random.randn() * s
                            qxy -= (m - parameters[parind+pindex]) ** 2 / 2 / \
                                s ** 2 - (m - q[parind+pindex]) ** 2 / 2 / s ** 2

                        else:
                            q[parind + pindex] = np.random.uniform(self.pmin[parind + pindex],
                                                                   self.pmax[parind + pindex])

                        pindex += 1

        return q, qxy

    # pulsar distance jump
    def pulsarDistanceJump(self, parameters, iter, beta):

        q = parameters.copy()
        qxy = 0

        # get pulsar distances
        L0 = []
        signum = self.getSignalNumbersFromDict(self.ptasignals,
                                               stype='pulsardistance', corr='single')

        # check to make sure we have all distances
        if len(signum) != self.npsr:
            raise ValueError(
                'ERROR: Number of pulsar distances != number of pulsars!')

        # get distances
        pdist_inds, pdisterr = [], []
        for signum0 in signum:
            sig0 = self.ptasignals[signum0]
            pdist_inds.append(sig0['parindex'])
            L0.append(self.psr[sig0['pulsarind']].pdist)
            pdisterr.append(self.psr[sig0['pulsarind']].pdistErr)

        # make indices array
        pdist_inds = np.array(pdist_inds)
        L0 = np.array(L0)
        pdisterr = np.array(pdisterr)

        #L_new = np.random.multivariate_normal(L0, np.diag(pdisterr ** 2))
        ind = np.random.randint(0, len(pdist_inds), len(pdist_inds))
        L_new = L0[ind] + np.random.randn(len(ind)) * pdisterr[ind]
        q[pdist_inds[ind]] = L_new
        qxy -= np.sum((L0 - parameters[pdist_inds]) ** 2 / 2 / pdisterr ** 2 -
                      (L0 - q[pdist_inds]) ** 2 / 2 / pdisterr ** 2)

        return q, qxy

    # pulsar phase prior draw
    def pulsarPhaseJump(self, parameters, iter, beta):

        q = parameters.copy()
        qxy = 0

        # loop over signals
        for ct, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind + npars]

            if sig['stype'] == 'pulsarTerm':
                if sig['bvary'][0]:
                    q[parind] = np.random.uniform(0, 2*np.pi)

            return q, qxy

    # phase wrapping fix
    def fix_cyclic_pars(self, prepar, postpar, iter, beta):
        """
        Wrap cyclic parameters into 0 -> 2 pi range
        """

        pre = prepar.copy()
        post = postpar.copy()

        # now get other relevant parameters
        for ct, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']

            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = post[parind:parind + npars]

            if sig['stype'] == 'cw':

                ind = sig['parid'].index('phi')
                if sig['bvary'][ind]:
                    pind = np.sum(sig['bvary'][:ind])
                    post[parind+pind] = np.mod(sparameters[ind], 2*np.pi)

                ind = sig['parid'].index('phase')
                if sig['bvary'][ind]:
                    pind = np.sum(sig['bvary'][:ind])
                    post[parind+pind] = np.mod(sparameters[ind], 2*np.pi)

            if sig['stype'] == 'pulsarTerm':
                if sig['bvary'][0]:
                    post[parind] = np.mod(sparameters[0], 2*np.pi)


        return post, 0

    # pulsar mode jump
    def pulsarPhaseFix(self, prepar, postpar, iter, beta):

        pre = prepar.copy()
        post = postpar.copy()

        # get pulsar distances
        L0, L1 = [], []
        signum = self.getSignalNumbersFromDict(
            self.ptasignals,stype='pulsardistance',
            corr='single')

        # check to make sure we have all distances
        if len(signum) != self.npsr:
            raise ValueError(
                'ERROR: Number of pulsar distances != number of pulsars!')

        # get distances
        pdist_inds = []
        for signum0 in signum:
            sig0 = self.ptasignals[signum0]
            pdist_inds.append(sig0['parindex'])
            L0.append(pre[sig0['parindex']])
            L1.append(post[sig0['parindex']])

        # make indices array
        pdist_inds = np.array(pdist_inds)
        L0 = np.array(L0)
        L1 = np.array(L1)

        # now get other relevant parameters
        for ct, sig in enumerate(self.ptasignals):
            
            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']

            if sig['stype'] == 'cw':

                # parameters for this signal
                par0 = sig['pstart'].copy()
                par1 = sig['pstart'].copy()

                # which ones are varying
                par0[sig['bvary']] = pre[parind:parind + npars]
                par1[sig['bvary']] = post[parind:parind + npars]

                #par0 = pre[sig['parindex']:(sig['parindex'] + sig['ntotpars'])]
                #par1 = post[sig['parindex']:(sig['parindex'] + sig['ntotpars'])]

                theta0, phi0, omega0 = par0[0], par0[1], \
                    10 ** par0[4] * np.pi
                theta1, phi1, omega1 = par1[0], par1[1], \
                    10 ** par1[4] * np.pi

                if sig['model'] == 'mass_ratio':
                    q = 10**par0[-1]
                    Mtot = 10**par0[2]
                    mc0 = Mtot * (q / (1+q)**2)**(3/5)

                    q = 10**par1[-1]
                    Mtot = 10**par1[2]
                    mc1 = Mtot * (q / (1+q)**2)**(3/5)
                else:
                    mc0 = 10**par0[2]
                    mc1 = 10**par1[2]


        # get angular separations
        cosMu0 = np.array([p.cosMu(theta0, phi0) for p in self.psr]).flatten()
        cosMu1 = np.array([p.cosMu(theta1, phi1) for p in self.psr]).flatten()

        # get in correct units
        L0 *= 1.0267e11
        L1 *= 1.0267e11
        mc0 *= 4.9e-6
        mc1 *= 4.9e-6

        # pulsar frequency
        omegap0 = omega0 * (1 + 256 / 5 * mc0 ** (5 / 3)
                            * omega0 ** (8 / 3) * L0 * (1 - cosMu0)) ** (-3 / 8)
        omegap1 = omega1 * (1 + 256 / 5 * mc1 ** (5 / 3)
                            * omega1 ** (8 / 3) * L1 * (1 - cosMu1)) ** (-3 / 8)

        # pulsar phase
        phase0 = 1 / 32 / \
            mc0 ** (5 / 3) * (omega0 ** (-5 / 3) - omegap0 ** (-5 / 3))
        phase1 = 1 / 32 / \
            mc1 ** (5 / 3) * (omega1 ** (-5 / 3) - omegap1 ** (-5 / 3))

        # size of phase jump
        prob = np.random.rand()
        if prob > 0.8:
            sigma = 0.5 * np.random.randn(self.npsr)

        elif prob > 0.6:
            sigma = 0.1 * np.random.randn(self.npsr)

        elif prob > 0.4:
            sigma = 0.02 * np.random.randn(self.npsr)

        else:
            sigma = 0.05 * np.random.randn(self.npsr)

        deltaL = (np.mod(phase1, 2 * np.pi) - np.mod(phase0, 2 * np.pi)
                  + sigma * np.sqrt(1 / beta)) / (omegap1 * (1 - cosMu1))
        L_new = (L1 + deltaL) / 1.0267e11
        post[pdist_inds] = L_new

        #L1 = L_new * 1.0267e11
        #omegap1 = omega1 * (1+256/5*mc1**(5/3)*omega1**(8/3)*L1*(1-cosMu1))**(-3/8)

        ## pulsar phase
        #phase1 = 1/32/mc1**(5/3) * (omega1**(-5/3) - omegap1**(-5/3))

        #for p1, p2, l1, l2, cm in zip(phase0, phase1, L0/1.0267e11, L_new, cosMu1):
        #    print np.mod(p1, 2*np.pi), np.mod(p2, 2*np.pi), \
        #        np.abs(l1-l2) / ((l1+l2)/2), 1-cm

        #print '\n'

        return post, 0

    # mass distance correlation jump
    def massDistanceJump(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # now get other relevant parameters
        for ct, sig in enumerate(self.ptasignals):
            if sig['stype'] == 'cw':

                par0 = parameters[
                    sig['parindex']:(sig['parindex'] + sig['npars'])]
                mc0, dist0 = 10 ** par0[2] * 4.9e-6, 10 ** par0[3] * 1.0267e14

                # draw distance uniformly from prior
                dist1 = 10 ** np.random.uniform(sig['pmin'][3], sig['pmax'][3])
                dist1 *= 1.0267e14

                # find chirp mass that keeps M^{5/3}/dL constant
                mc1 = (dist1 * mc0 ** (5 / 3) / dist0) ** (3 / 5)

                # put values in return array
                q[sig['parindex'] + 2] = np.log10(mc1 / 4.9e-6)
                q[sig['parindex'] + 3] = np.log10(dist1 / 1.0267e14)

        return q, qxy

    # phase and polarization reversal
    def phaseAndPolarizationReverseJump(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # now get other relevant parameters
        for ct, sig in enumerate(self.ptasignals):
            if sig['stype'] == 'cw':

                # short hand
                psrind = sig['pulsarind']
                parind = sig['parindex']
                npars = sig['npars']

                # parameters for this signal
                sparameters = sig['pstart'].copy()

                # which ones are varying
                sparameters[sig['bvary']] = parameters[parind:parind + npars]

                # parameter indices
                ind1, ind2 = sig['parid'].index('phase'), sig['parid'].index('pol')

                if sig['bvary'][ind1] and sig['bvary'][ind2]:
                    pind1 = np.sum(sig['bvary'][:ind1])
                    pind2 = np.sum(sig['bvary'][:ind2])
                    phase0, phi0 = sparameters[ind1], sparameters[ind2]
                    q[parind+pind1] = np.mod(phase0+np.pi, 2*np.pi)
                    q[parind+pind2] = np.mod(phi0+np.pi/2, np.pi)

        return q, qxy



    # draws from equad prior
    def drawFromEquadPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='equad',
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='equad',
                                               corr='single')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary']:

                # log prior
                if sig['prior'] == 'log':
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10 ** self.pmin[parind],
                                                           10 ** self.pmax[parind]))
                    qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]

        return q, qxy

    # draws from jitter equad prior
    def drawFromJitterEpochPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals,
                                                       stype='jitter_epoch', corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='jitter_epoch',
                                               corr='single')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            for jj in range(npars):
                if sig['bvary'][jj]:

                    # log prior
                    if sig['prior'][jj] == 'log':
                        q[parind + jj] = np.random.uniform(self.pmin[parind + jj],
                                                           self.pmax[parind + jj])
                        qxy += 0

                    elif sig['prior'][jj] == 'uniform':
                        q[parind + jj] = np.log10(np.random.uniform(10 ** self.pmin[parind + jj],
                                                                    10 ** self.pmax[parind + jj]))
                        qxy += np.log(10 **
                                      parameters[parind + jj] / 10 ** q[parind + jj])

                    else:
                        print 'Prior type not recognized for parameter'
                        q[parind + jj] = parameters[parind + jj]

        return q, qxy

    # draws from jitter equad prior
    def drawFromJitterEquadPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals,
                                                       stype='jitter_equad', corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='jitter_equad',
                                               corr='single')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary']:

                # log prior
                if sig['prior'] == 'log':
                    q[parind] = np.random.uniform(self.pmin[parind],
                                                  self.pmax[parind])
                    qxy += 0

                elif sig['prior'][jj] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10 ** self.pmin[parind],
                                                           10 ** self.pmax[parind]))
                    qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]

        return q, qxy

    # draws from efac prior
    def drawFromEfacPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='efac',
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='efac',
                                               corr='single')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, nsigs))
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            if sig['bvary']:

                # uniform prior
                if sig['prior'] == 'uniform':
                    q[parind] = np.random.uniform(
                        self.pmin[parind], self.pmax[parind])
                    qxy += 0

                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]

        return q, qxy

    def updateFisher(self, parameters, incJitter=False):

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals,
                                                       stype='nonlineartimingmodel', corr='single'))
        if np.any(nsigs):
            Mmat = self.psr[0].t2psr.designmatrix(fixunits=True)
        else:
            Mmat = self.psr[0].Mmat.copy()

        # normalize
        self.norm = np.sqrt(np.sum(Mmat ** 2, axis=0))
        Mmat /= self.norm

       # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=False,
                                incTM=True, incJitter=incJitter)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        self.updateTmatrix(parameters)

        # number of parameters
        nptmpars = self.psr[0].Mmat_reduced.shape[1]
        nf = len(self.psr[0].Ffreqs)
        nfdm = len(self.psr[0].Fdmfreqs)
        try:
            necorr = len(self.psr[0].avetoas)
        except TypeError:
            necorr = 0

        Tmat = self.psr[0].Ttmat[:, nptmpars:nptmpars + nf + nfdm + necorr]

        mat1 = np.dot(Mmat.T / self.psr[0].Nvec, Mmat)
        TNT = np.dot(Tmat.T / self.psr[0].Nvec, Tmat)
        TNM = np.dot(Tmat.T / self.psr[0].Nvec, Mmat)
        Phiinv = self.Phiinv[nptmpars:nptmpars + nf + nfdm + necorr,
                             nptmpars:nptmpars + nf + nfdm + necorr]

        Sigma = TNT + Phiinv

        cf = sl.cho_factor(Sigma)
        SigmaTNM = sl.cho_solve(cf, TNM)

        fisherind = mat1 - np.dot(TNM.T, SigmaTNM)

        U, s, Vh = sl.svd(fisherind)
        if not np.all(s > 0):
            raise ValueError("Sigi singular according to SVD")
        fisher = np.dot(Vh.T, np.dot(np.diag(1.0 / s), U.T))

        # set fisher matrix
        self.fisher = fisher
        self.fisherU = U
        self.fisherS = s

    # draws from timing model parameter prior
    def drawFromTMPrior(self, parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals,
                                                       stype='nonlineartimingmodel', corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='nonlineartimingmodel',
                                               corr='single')

        # which parameters to jump
        ind = np.unique(np.random.randint(0, nsigs, 1))

        # draw params from prior
        for ii in ind:

            # get signal
            sig = self.ptasignals[signum[ii]]
            parind = sig['parindex']
            npars = sig['npars']

            # jump in amplitude if varying
            for jj in range(npars):
                if sig['bvary'][jj]:

                    # prior
                    q[parind + jj] = np.random.uniform(self.pmin[parind + jj],
                                                       self.pmax[parind + jj])

        return q, qxy

    # draws from timing model fisher matrix
    def drawFromTMfisherMatrix(self, parameters, iter, beta):

        # if (iter-1) % 5000 == 0 and (iter-1) != 0 and self.likfunc != 'mark8':
        #    self.updateFisher(parameters)

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals,
                                                       stype='nonlineartimingmodel', corr='single'))
        if np.any(nsigs):
            signum = self.getSignalNumbersFromDict(self.ptasignals,
                                                   stype='nonlineartimingmodel', corr='single')
        else:
            signum = self.getSignalNumbersFromDict(self.ptasignals,
                                                   stype='lineartimingmodel', corr='single')

        # current timing model parameters
        sig = self.ptasignals[signum]
        x = parameters[sig['parindex']:(sig['parindex'] + sig['ntotpars'])]

        # get parmeters in new diagonalized basis
        #y = np.dot(self.psr[0].fisherU.T, x*self.psr[0].norm)

        # get scale of jump
        #scale = np.random.uniform(1, 50)
        #alpha = np.random.rand()
        #scale = 1
        # if alpha < 0.1:
        #    scale = 50
        # if alpha < 0.5:
        #    scale = 25
        # if alpha < 0.25:
        #    scale = 10
        scale = np.random.uniform(1, 50)

        # make correlated componentwise adaptive jump
        ind = np.unique(np.random.randint(0, len(x), 1))
        #ind = np.arange(0, len(x))
        neff = len(ind)
        sd = 2.4 / np.sqrt(2 * neff) * scale / np.sqrt(beta)

        q[sig['parindex']:(sig['parindex'] + sig['ntotpars'])] += \
            np.random.randn() / np.sqrt(self.psr[0].fisherS[ind]) \
            * self.psr[0].fisherU[:, ind].flatten() * scale / self.psr[0].norm

        #y[ind] = y[ind] + np.random.randn(neff) * sd * np.sqrt(1/self.psr[0].fisherS[ind])
        # q[sig['parindex']:(sig['parindex']+sig['ntotpars'])] = \
        #        np.dot(self.psr[0].fisherU, y)/self.psr[0].norm

        return q, qxy

    """
    Reconstruct maximum likelihood and uncertainty of fourier coefficients

    EFAC + EQUAD + Red noise + DMV + GWs

    No jitter or frequency lines

    Uses Woodbury lemma

    """

    def reconstructFourier(self, parameters, incCorrelations=True):

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # compute the white noise terms in the log likelihood
        FGGNGGF = []
        ahat, sigma_ahat = [], []
        nfref = 0
        for ct, p in enumerate(self.psr):

            if p.twoComponentNoise:

                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component basis
                # if ct == 0:
                d = np.dot(p.AGF.T, p.AGr / p.Nwvec)
                # else:
                #    d = np.append(d, np.dot(p.AGF.T, p.AGr/p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1 / p.Nwvec) * p.AGF.T).T
                FGGNGGF.append(np.dot(p.AGF.T, right))

            else:

                # G(G^TNG)^{-1}G^T = N^{-1} -
                # N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0 / p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiF = ((1.0 / p.Nvec) * p.Ftot.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiF = np.dot(NiGc.T, p.Ftot)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                # F^TG(G^TNG)^{-1}G^T\delta t
                # if ct == 0:
                d = np.dot(p.Ftot.T, Nir - NiGcNiGcr)
                # else:
                #    d = np.append(d, np.dot(p.Ftot.T, Nir - NiGcNiGcr))

                # compute F^TG(G^TNG)^{-1}G^TF
                FGGNGGF.append(
                    np.dot(NiF.T, p.Ftot) - np.dot(GcNiF.T, GcNiGcF))

            # compute the red noise, DMV and GWB terms in the log likelihood

            # compute sigma
            nf = self.npftot[ct]
            print nf
            Sigma = FGGNGGF[ct] + \
                self.Phiinv[nfref:(nfref + nf), nfref:(nfref + nf)]
            Phi = np.diag(self.Phi[nfref:(nfref + nf)])

            # cholesky decomp for maximum likelihood fourier components
            try:
                cf = sl.cho_factor(Sigma)
                ahat.append(sl.cho_solve(cf, d))
                SigmaInv = sl.cho_solve(cf, np.eye(Sigma.shape[0]))
            except np.linalg.LinAlgError:
                raise ValueError("ERROR: Sigma singular according to SVD")

            # calculate uncertainty in max-likelihood fourier coefficients
            sigma_ahat.append(np.diag(SigmaInv))

            # increment frequency counter
            nfref += nf

        return ahat, sigma_ahat

    def maxLikePTpars(self, parameters, ptmpars=None, incJitter=False):

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # loop over all pulsars and contruct ML estimates
        cov, eps, chisq = [], [], []
        nfref = 0
        for ct, p in enumerate(self.psr):

            # determine which parameters were analytically marginalized
            if ptmpars is not None:
                tmpardel = p.getNewTimingModelParameterList(
                    keep=False, tmpars=pttmpars)
                Mmat, newptmpars, newptmdescription = self.delFromDesignMatrix(
                    tmpardel)
            else:
                Mmat = p.Mmat

            # compute F^TN^{-1}F
            right = ((1 / p.Nvec) * p.Ftot.T).T
            FNF = np.dot(p.Ftot.T, right)

            if incJitter:
                # compute F^TN^{-1}F
                right = ((1 / p.Nvec) * p.Umat.T).T
                UNU = np.dot(p.Umat.T, right)

            # compute sigma
            nf = self.npftot[ct]
            Sigma = FNF + self.Phiinv[nfref:(nfref + nf), nfref:(nfref + nf)]

            if incJitter:
                Phi0 = np.diag(self.Phi[nfref:(nfref + nf)])
                UPhiU = np.dot(p.UtF, np.dot(Phi0, p.UtF.T))
                Phi = UPhiU + np.diag(p.Qamp)

                try:
                    cf = sl.cho_factor(Phi)
                    Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
                except np.linalg.LinAlgError:
                    raise ValueError("ERROR: Phi0 singular according to SVD")

                Sigma = Phiinv + UNU

            # cholesky decomp for ML estimators
            try:
                cf = sl.cho_factor(Sigma)
                SigmaInv = sl.cho_solve(cf, np.eye(Sigma.shape[0]))
            except np.linalg.LinAlgError:
                raise ValueError("ERROR: Sigma singular according to SVD")

            # ML estimate for parameter offset
            if incJitter:
                tmp = np.dot(
                    SigmaInv, np.dot(p.Umat.T, (1 / p.Nvec * Mmat.T).T))
                tmp2 = np.dot(
                    SigmaInv, np.dot(p.Umat.T, p.detresiduals / p.Nvec))
                g = np.dot(Mmat.T, p.detresiduals / p.Nvec) - \
                    np.dot(Mmat.T, np.dot((1 / p.Nvec * p.Umat.T).T, tmp2))
                Gamma = np.dot(Mmat.T, (1 / p.Nvec * Mmat.T).T) - \
                    np.dot(Mmat.T, np.dot((1 / p.Nvec * p.Umat.T).T, tmp))
                GammaInv = np.linalg.inv(Gamma)
                cov.append(GammaInv)
                eps.append(np.dot(GammaInv, g))

            else:
                tmp = np.dot(
                    SigmaInv, np.dot(p.Ftot.T, (1 / p.Nvec * Mmat.T).T))
                tmp2 = np.dot(
                    SigmaInv, np.dot(p.Ftot.T, p.detresiduals / p.Nvec))
                g = np.dot(Mmat.T, p.detresiduals / p.Nvec) - \
                    np.dot(Mmat.T, np.dot((1 / p.Nvec * p.Ftot.T).T, tmp2))
                Gamma = np.dot(Mmat.T, (1 / p.Nvec * Mmat.T).T) - \
                    np.dot(Mmat.T, np.dot((1 / p.Nvec * p.Ftot.T).T, tmp))
                GammaInv = np.linalg.inv(Gamma)
                cov.append(GammaInv)
                eps.append(np.dot(GammaInv, g))

            p.detresiduals = p.residuals - np.dot(Mmat, eps[-1])
            if incJitter:
                d = np.dot(p.Umat.T, p.detresiduals / p.Nvec)
            else:
                d = np.dot(p.Ftot.T, p.detresiduals / p.Nvec)

            expval2 = sl.cho_solve(cf, d)

            # triple product
            rNr = np.dot(p.detresiduals, p.detresiduals / p.Nvec)

            chisq.append(rNr - np.dot(d, expval2))

            # increment frequency counter
            nfref += nf

        return eps, cov, chisq

    def create_realization(self, pars, incJitter=True, signal='red'):
        """
        Very simple function to return ML realization
        of linear signal.

        """

        ml_vals, ml_cov = self.reconstructML_old(pars, incJitter=incJitter)
        mlreal, mlerr = [], []
        for ct, p in enumerate(self.psr):

            nfred = len(p.Ffreqs)
            nfdm = len(p.Fdmfreqs)
            ntmpars = len(p.ptmdescription)
            if incJitter:
                njitter = len(p.avetoas)
            else:
                njitter = 0

            # index dictionary
            ind = {}
            ind['tm'] = (0, ntmpars)
            ind['red'] = (ntmpars, ntmpars+nfred)
            ind['dm'] = (ntmpars+nfred, ntmpars+nfred+nfdm)
            ind['jitter'] = (ntmpars+nfred+nfdm, ntmpars+nfred+nfdm+njitter)

            # ML realization
            mlpars = ml_vals[ct][ind[signal][0]:ind[signal][1]]
            mlcov = ml_cov[ct][ind[signal][0]:ind[signal][1],
                               ind[signal][0]:ind[signal][1]]
            Tmat = p.Ttmat[:,ind[signal][0]:ind[signal][1]]

            mlreal.append(np.dot(Tmat, mlpars))
            tmp = np.dot(Tmat, np.dot(mlcov, Tmat.T))
            mlerr.append(np.sqrt(np.diag(tmp)))

        return mlreal, mlerr

