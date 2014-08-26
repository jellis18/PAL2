#!/usr/bin/env python

from __future__ import division

import numpy as np
import h5py as h5
import os, sys, time
import json
import tempfile
import scipy.linalg as sl
import scipy.special as ss
from numpy.polynomial.hermite import hermval

import PALutils
import PALdatafile
import PALpsr

# In order to keep the dictionary in order
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


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

    def __init__(self, h5filename=None, jsonfilename=None, pulsars='all', auxFromFile=True):
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
        if pulsars=='all':
            readpsrs = psrnames
        else:
            # Check if all provided pulsars are indeed in the HDF5 file
            if np.all(np.array([pulsars[ii] in psrnames for ii in range(len(pulsars))]) == True):
                readpsrs = pulsars
            elif pulsars in destpsrnames:
                pulsars = [pulsars]
                readpsrs = pulsars
            else:
                raise ValueError("ERROR: Not all provided pulsars in HDF5 file")

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
    def makeModelDict(self,  nfreqs=20, ndmfreqs=None, \
            incRedNoise=False, noiseModel='powerlaw', fc=None, logf=False,\
            incDM=False, dmModel='powerlaw', \
            incScattering=False, scatteringModel='powerlaw', \
            incGWB=False, gwbModel='powerlaw', \
            incBWM=False, \
            incCW=False, incPulsarDistance=False, \
            varyEfac=True, separateEfacs=False, separateEfacsByFreq=False, \
            incEquad=False,separateEquads=False, separateEquadsByFreq=False, \
            incJitter=False, separateJitter=False, separateJitterByFreq=False, \
            incTimingModel=False, nonLinear=False, fulltimingmodel=False,\
            incJitterEpoch=False, nepoch=None, \
            incNonGaussian=False, nnongaussian=3, \
            incJitterEquad=False, separateJitterEquad=False, separateJitterEquadByFreq=False, \
            efacPrior='uniform', equadPrior='log', jitterPrior='uniform', \
            jitterEquadPrior='log', \
            redAmpPrior='log', redSiPrior='uniform', GWAmpPrior='log', GWSiPrior='uniform', \
            DMAmpPrior='log', DMSiPrior='uniform', redSpectrumPrior='log', \
            DMSpectrumPrior='log', \
            GWspectrumPrior='log', \
            incSingleFreqNoise=False, numSingleFreqLines=1, \
            incSingleFreqDMNoise=False, numSingleFreqDMLines=1, \
            singlePulsarMultipleFreqNoise=None, \
            multiplePulsarMultipleFreqNoise=None, \
            dmFrequencyLines=None, \
            orderFrequencyLines=False, \
            compression = 'None', \
            targetAmp = 1e-14, \
            evalCompressionComplement = False, \
            likfunc='mark1'):
        
        signals = []

        # start loop over pulsars 
        for ii, p in enumerate(self.psr):

            # how many frequencies
            if incDM:
                if ndmfreqs is None or ndmfreqs=="None":
                    ndmfreqs = nfreqs
            else:
                ndmfreqs = 0

            if separateEfacs or separateEfacsByFreq:
                if separateEfacs and ~separateEfacsByFreq:
                    pass

                # if both set, default to fflags
                else:
                    p.flags = p.fflags  # TODO: make this more elegant

                uflagvals = list(set(p.flags))  # Unique flags
                for flagval in uflagvals:
                    newsignal = OrderedDict({
                        "stype":"efac",
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"efacequad",
                        "flagvalue":flagval,
                        "bvary":[varyEfac],
                        "pmin":[0.001],
                        "pmax":[50.0],
                        "pwidth":[0.1],
                        "pstart":[1.0],
                        "prior":efacPrior
                        })
                    signals.append(newsignal)
            else:
                newsignal = OrderedDict({
                    "stype":"efac",
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":p.name,
                    "bvary":[varyEfac],
                    "pmin":[0.001],
                    "pmax":[50.0],
                    "pwidth":[0.1],
                    "pstart":[1.0], 
                    "prior":efacPrior
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
                            "stype":"jitter",
                            "corr":"single",
                            "pulsarind":ii,
                            "flagname":"jitter",
                            "flagvalue":flagval,
                            "bvary":[True],
                            "pmin":[0],
                            "pmax":[5],
                            "pwidth":[0.1],
                            "pstart":[0.333],
                            "prior":jitterPrior
                            })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype":"jitter",
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":p.name,
                        "bvary":[True],
                        "pmin":[0],
                        "pmax":[5],
                        "pwidth":[0.1],
                        "pstart":[0.333],
                        "prior":jitterPrior
                        })
                    signals.append(newsignal)
            
            if incJitterEquad:
                if separateJitterEquad or separateJitterEquadByFreq:
                    if separateJitterEquad and ~separateJitterEquadByFreq:
                        pass

                    # if both set, default to fflags
                    else:
                        p.flags = p.fflags

                    uflagvals = list(set(p.flags))  # Unique flags
                    for flagval in uflagvals:
                        newsignal = OrderedDict({
                            "stype":"jitter_equad",
                            "corr":"single",
                            "pulsarind":ii,
                            "flagname":"jitter_equad",
                            "flagvalue":flagval,
                            "bvary":[True],
                            "pmin":[-8.5],
                            "pmax":[-4.0],
                            "pwidth":[0.1],
                            "pstart":[-8.0],
                            "prior":jitterEquadPrior
                            })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype":"jitter_equad",
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":p.name,
                        "bvary":[True],
                        "pmin":[-8.5],
                        "pmax":[-4.0],
                        "pwidth":[0.1],
                        "pstart":[-8.0],
                        "prior":jitterEquadPrior
                        })
                    signals.append(newsignal)


            if incJitterEpoch:
                    bvary = [True]*nepoch[ii]
                    pmin = [-10.0]*nepoch[ii]
                    pmax = [-4.0]*nepoch[ii]
                    pstart = [-9.0]*nepoch[ii]
                    pwidth = [0.5]*nepoch[ii]
                    prior = [jitterEquadPrior]*nepoch[ii]

                    newsignal = OrderedDict({
                        "stype":"jitter_epoch",
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":p.name,
                        "bvary":bvary,
                        "pmin":pmin,
                        "pmax":pmax,
                        "pwidth":pwidth,
                        "pstart":pstart,
                        "prior":prior
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
                            "stype":"equad",
                            "corr":"single",
                            "pulsarind":ii,
                            "flagname":"equad",
                            "flagvalue":flagval,
                            "bvary":[True],
                            "pmin":[-10.0],
                            "pmax":[-4.0],
                            "pwidth":[0.1],
                            "pstart":[-8.0],
                            "prior":equadPrior
                            })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype":"equad",
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":p.name,
                        "bvary":[True],
                        "pmin":[-10.0],
                        "pmax":[-4.0],
                        "pwidth":[0.1],
                        "pstart":[-8.0],
                        "prior":equadPrior
                        })
                    signals.append(newsignal)


            if incRedNoise:
                if noiseModel=='spectrum':
                    #nfreqs = numNoiseFreqs[ii]
                    bvary = [True]*nfreqs
                    pmin = [-18.0]*nfreqs
                    pmax = [-7.0]*nfreqs
                    pstart = [-18.0]*nfreqs
                    pwidth = [0.1]*nfreqs
                    prior = [redSpectrumPrior]*nfreqs
                elif noiseModel=='powerlaw':
                    bvary = [True, True, False]
                    pmin = [-20.0, 1.02, 1.0e-11]
                    pmax = [-11.0, 6.98, 3.0e-9]
                    pstart = [-19.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                    prior = [redAmpPrior, redSiPrior, 'log']
                elif noiseModel=='spectralModel':
                    bvary = [True, True, True]
                    pmin = [-28.0, 0.0, -4.0]
                    pmax = [-14.0, 12.0, 2.0]
                    pstart = [-22.0, 2.0, -1.0]
                    pwidth = [-0.2, 0.1, 0.1]
                    prior = [redAmpPrior, redSiPrior, 'uniform']

                newsignal = OrderedDict({
                    "stype":noiseModel,
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":p.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "prior":prior
                    })
                signals.append(newsignal)

            if incDM:
                if dmModel=='spectrum':
                    #nfreqs = ndmfreqs
                    bvary = [True]*ndmfreqs
                    pmin = [-14.0]*ndmfreqs
                    pmax = [-3.0]*ndmfreqs
                    pstart = [-7.0]*ndmfreqs
                    pwidth = [0.1]*ndmfreqs
                    prior = [DMSpectrumPrior]*nfreqs
                    DMModel = 'dmspectrum'
                elif dmModel=='powerlaw':
                    bvary = [True, True, False]
                    pmin = [-14.0, 1.02, 1.0e-11]
                    pmax = [-6.5, 6.98, 3.0e-9]
                    pstart = [-13.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                    prior = [DMAmpPrior, DMSiPrior, 'log']
                    DMModel = 'dmpowerlaw'

                newsignal = OrderedDict({
                    "stype":DMModel,
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":p.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "prior":prior
                    })
                signals.append(newsignal)
            
            if incScattering:
                if scatteringModel=='spectrum':
                    #nfreqs = ndmfreqs
                    bvary = [True]*ndmfreqs
                    pmin = [-14.0]*ndmfreqs
                    pmax = [-3.0]*ndmfreqs
                    pstart = [-7.0]*ndmfreqs
                    pwidth = [0.1]*ndmfreqs
                    prior = [DMSpectrumPrior]*nfreqs
                    ScatteringModel = 'scatspectrum'
                elif scatteringModel=='powerlaw':
                    bvary = [True, True, False]
                    pmin = [-14.0, 1.02, 1.0e-11]
                    pmax = [-6.5, 6.98, 3.0e-9]
                    pstart = [-13.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                    prior = [DMAmpPrior, DMSiPrior, 'log']
                    ScatteringModel = 'scatpowerlaw'

                newsignal = OrderedDict({
                    "stype":ScatteringModel,
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":p.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "prior":prior
                    })
                signals.append(newsignal)

            if incSingleFreqNoise:

                for jj in range(numSingleFreqLines):
                    newsignal = OrderedDict({
                        "stype":'frequencyline',
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":p.name,
                        "bvary":[True, True],
                        "pmin":[-9.0, -18.0],
                        "pmax":[-5.0, -9.0],
                        "pwidth":[-0.1, -0.1],
                        "pstart":[-7.0, -10.0],
                        "prior":['log', 'log']
                        })
                    signals.append(newsignal)

            if incSingleFreqDMNoise:

                for jj in range(numSingleFreqDMLines):
                    newsignal = OrderedDict({
                        "stype":'dmfrequencyline',
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":p.name,
                        "bvary":[True, True],
                        "pmin":[-9.0, -14.0],
                        "pmax":[-5.0, -3.0],
                        "pwidth":[-0.1, -0.1],
                        "pstart":[-7.0, -10.0],
                        "prior":['log', 'log']
                        })
                    signals.append(newsignal)

            
            if incTimingModel:
                if nonLinear:
                    # Get the parameter errors from libstempo. Initialise the
                    # libstempo object
                    p.initLibsTempoObject()

                    #errs = []
                    #est = []
                    #for t2par in p.t2psr.pars:
                    #    errs += [np.longdouble(p.t2psr[t2par].err)]
                    #    est += [np.longdouble(p.t2psr[t2par].val)]
                    #tmperrs = np.array([0.0] + errs)
                    #tmpest = np.array([0.0] + est)
                #else:

                # Just do the timing-model fit ourselves here, in order to set
                # the prior.
                w = 1.0 / p.toaerrs**2
                Sigi = np.dot(p.Mmat.T, (w * p.Mmat.T).T)
                try:
                    cf = sl.cho_factor(Sigi)
                    Sigma = sl.cho_solve(cf, np.eye(Sigi.shape[0]))
                except np.linalg.LinAlgError:
                    U, s, Vh = sl.svd(Sigi)
                    if not np.all(s > 0):
                        raise ValueError("Sigi singular according to SVD")
                    Sigma = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))
                
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
                    newptmdescription = p.getNewTimingModelParameterList(keep=True, \
                            tmpars=[])
                else:
                    newptmdescription = p.getNewTimingModelParameterList(keep=True, \
                        tmpars=['Offset', 'F0', 'F1', 'RAJ', 'DECJ', 'LAMBDA', \
                                'BETA' ,'DM', 'DM1', 'DM2'] + jumps + dmx + fds)


                # Select the numerical parameters. These are the ones not
                # present in the quantities that getModifiedDesignMatrix
                # returned
                parids=[]
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
                        if parid == 'SINI':
                            pmin += [-1.0]
                            pmax += [1.0]
                            pwidth += [tmperrs[jj]]
                            if tmpest[jj] < -1 or tmpest[jj] > 1:
                                pstart += [0.99]
                            else:
                                pstart += [tmpest[jj]]
                        elif parid == 'ECC':
                            pmin += [0.0]
                            pmax += [1.0]
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
                        elif parid == 'EDOT':
                            pmin += [-500.0 * tmperrs[jj] + tmpest[jj]]
                            pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                            pwidth += [tmperrs[jj]]
                            pstart += [0]
                        
                        # make parameter be the 'parameter offset'
                        elif parid == 'Offset':
                            pmin += [-500]
                            pmax += [500]
                            pwidth += [0.1]
                            pstart += [0]

                        elif parid == 'F0':
                            pmin += [-500]
                            pmax += [500]
                            pwidth += [0.1]
                            pstart += [0]
                        
                        elif parid == 'F1':
                            pmin += [-500]
                            pmax += [500]
                            pwidth += [0.1]
                            pstart += [0]

                        else:
                            pmin += [-500.0 * tmperrs[jj] + tmpest[jj]]
                            pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                            pwidth += [tmperrs[jj]]
                            pstart += [tmpest[jj]]
                        #print parid, pmin[-1], pmax[-1], pwidth[-1], pstart[-1]

                if nonLinear:
                    stype = 'nonlineartimingmodel'
                else:
                    stype = 'lineartimingmodel'

                newsignal = OrderedDict({
                    "stype":stype,
                    "corr":"single",
                    "pulsarind":ii,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "parid":parids,
                    "prior":'flat'
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
                    "stype":'nongausscoeff',
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":p.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "prior":prior
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
                    "stype":'pulsardistance',
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":p.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "prior":prior
                })
                signals.append(newsignal)


        if incCW:
            bvary = [True] * 8
            pmin = [0, 0, 7, 0.1, -9, 0, 0, 0]
            pmax = [np.pi, 2*np.pi, 10, 3, -7, np.pi, np.pi, np.pi]
            pstart = [np.pi/2, np.pi/2, 8, 1, -8, np.pi, np.pi, np.pi/2]
            pwidth = [0.1, 0.1, 0.1, 0.1, 0.001, 0.1, 0.1, 0.1]
            prior = ['cos', 'uniform', 'log', 'log', 'log', 'uniform', 'uniform', 'cos']
            parids = ['theta', 'phi', 'logmc', 'logd', 'logf', 'phase', 'pol', 'inc']
            
            newsignal = OrderedDict({
                "stype":"cw",
                "corr":"gr",
                "pulsarind":-1,
                "bvary":bvary,
                "pmin":pmin,
                "pmax":pmax,
                "pwidth":pwidth,
                "pstart":pstart,
                "parid":parids,
                "prior":prior
                })
            signals.append(newsignal)

 
        if incGWB:
            if gwbModel=='spectrum':
                bvary = [True]*nfreqs
                pmin = [-18.0]*nfreqs
                pmax = [-8.0]*nfreqs
                pstart = [-10.0]*nfreqs
                pwidth = [0.1]*nfreqs
                prior = [GWspectrumPrior]*nfreqs
            elif gwbModel=='powerlaw':
                bvary = [True, True, False]
                pmin = [-17.0, 1.02, 1.0e-11]
                pmax = [-11.0, 6.98, 3.0e-9]
                pstart = [-15.0, 2.01, 1.0e-10]
                pwidth = [0.1, 0.1, 5.0e-11]
                prior = [GWAmpPrior, GWSiPrior, 'log']

            newsignal = OrderedDict({
                "stype":gwbModel,
                "corr":"gr",
                "pulsarind":-1,
                "bvary":bvary,
                "pmin":pmin,
                "pmax":pmax,
                "pwidth":pwidth,
                "pstart":pstart,
                "prior":prior
                })
            signals.append(newsignal)

        # The list of signals
        modeldict = OrderedDict({
            "file version":2014.02,
            "author":"PAL-makeModel",
            "numpulsars":len(self.psr),
            "pulsarnames":[self.psr[ii].name for ii in range(len(self.psr))],
            "numNoiseFreqs":[nfreqs for ii in range(len(self.psr))],
            "numDMFreqs":[ndmfreqs for ii in range(len(self.psr))],
            "compression":compression,
            "logfrequencies":logf,
            "targetAmp":targetAmp,
            "orderFrequencyLines":orderFrequencyLines,
            "evalCompressionComplement":evalCompressionComplement,
            "likfunc":likfunc,
            "signals":signals
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
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))

        # Determine the time baseline of the array of pulsars
        if not 'Tmax' in signal:
            Tstart = np.min(self.psr[0].toas)
            Tfinish = np.max(self.psr[0].toas)
            for p in self.psr:
                Tstart = np.min([np.min(p.toas), Tstart])
                Tfinish = np.max([np.max(p.toas), Tfinish])
            Tmax = Tfinish - Tstart

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
        if signal['stype']=='efac':
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

        elif signal['stype'] in ['powerlaw', 'spectrum', 'spectralModel']:
            # Any time-correlated signal
            self.addSignalTimeCorrelated(signal)
            self.haveStochSources = True

        elif signal['stype'] in ['dmpowerlaw', 'dmspectrum']:
            # A DM variation signal
            self.addSignalDMV(signal)
            self.haveStochSources = True

        elif signal['stype'] == 'cw':
            # a continuous GW signal
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
            psrSingleFreqs = self.getNumberOfSignals(stype='frequencyline', \
                    corr='single')
            signal['npsrfreqindex'] = psrSingleFreqs[signal['pulsarind']]
            self.addSignalFrequencyLine(signal)
            self.haveFrequencyLines = True

        elif signal['stype'] == 'dmfrequencyline':
            # Single free-floating frequency line
            psrSingleFreqs = self.getNumberOfSignals(stype='dmfrequencyline', \
                    corr='single')
            signal['npsrfreqindex'] = psrSingleFreqs[signal['pulsarind']]
            self.addSignalFrequencyLine(signal)
            self.haveFrequencyLines = True

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
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in efac signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))

        signal['Nvec'] = self.psr[signal['pulsarind']].toaerrs**2

        if signal['flagname'] != 'pulsarname':
            # This efac only applies to some TOAs, not all of 'm
            ind = np.array(self.psr[signal['pulsarind']].flags) != signal['flagvalue']
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
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in jitter signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))
        
        # eq 5 from cordes and shannon measurement model paper
        Wims = 0.1 * self.psr[signal['pulsarind']].period*1e3
        mI = 1
        if self.likfunc == 'mark2':
            N6 = self.psr[signal['pulsarind']].avetobs/self.psr[signal['pulsarind']].period/1e6
            sigmaJ = 0.28e-6 * Wims * 1/np.sqrt(N6) * np.sqrt((1+mI**2)/2) 
            signal['Jvec'] = sigmaJ**2
        else:
            N6 = self.psr[signal['pulsarind']].tobsflags/self.psr[signal['pulsarind']].period/1e6
            sigmaJ = 0.28e-6 * Wims * 1/np.sqrt(N6) * np.sqrt((1+mI**2)/2) 
            signal['Nvec'] = sigmaJ**2

        if signal['flagname'] != 'pulsarname':
            # This jitter only applies to some average TOAs, not all of 'm
            if self.likfunc in ['mark2', 'mark6']:
                ind = np.array(self.psr[signal['pulsarind']].aveflags) != signal['flagvalue']
                signal['Jvec'][ind] = 0.0
            else:
                ind = np.array(self.psr[signal['pulsarind']].flags) != signal['flagvalue']
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
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary', \
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
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in jitter equad signal. \
                             Keys: {0}. Required: {1}".format(signal.keys(), keys))
        
        # This is the 'jitter' that we have used before
        signal['Jvec'] = np.ones(len(self.psr[signal['pulsarind']].avetoas))

        if signal['flagname'] != 'pulsarname':
            # This jitter only applies to some average TOAs, not all of 'm
            ind = np.array(self.psr[signal['pulsarind']].aveflags) != signal['flagvalue']
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
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in equad signal. Keys: {0}. \
                             Required: {1}".format(signal.keys(), keys))

        signal['Nvec'] = np.ones(len(self.psr[signal['pulsarind']].toaerrs))

        if signal['flagname'] != 'pulsarname':
            # This equad only applies to some TOAs, not all of 'm
            ind = np.array(self.psr[signal['pulsarind']].flags) != signal['flagvalue']
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
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
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
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
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
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex', 'Tmax']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in signal. Keys: {0}. \
                             Required: {1}".format(signal.keys(), keys))

        if signal['corr'] == 'gr':
            # Correlated with the Hellings \& Downs matrix
            signal['corrmat'] = PALutils.computeORFMatrix(self.psr)/2
        elif signal['corr'] == 'uniform':
            # Uniformly correlated (Clock signal)
            signal['corrmat'] = np.ones((len(self.psr), len(self.psr)))

        if signal['corr'] != 'single':
            # Also fill the Ffreqs array, since we are dealing with correlations
            numfreqs = np.array([len(self.psr[ii].Ffreqs) \
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
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
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
        keys = ['pulsarind', 'stype', 'corr', 'bvary', 'parid', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in TimingModel signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))

        # Assert that this signal applies to a pulsar
        if signal['pulsarind'] < 0 or signal['pulsarind'] >= len(self.psr):
            raise ValueError("ERROR: timingmodel signal applied to non-pulsar ({0})".format(signal['pulsarind']))

        # Check that the parameters included here are also present in the design
        # matrix
        for ii, parid in enumerate(signal['parid']):
            if not parid in self.psr[signal['pulsarind']].ptmdescription:
                raise ValueError("ERROR: timingmodel signal contains non-valid parameter id")

        # If this is a non-linear signal, make sure to initialise the libstempo
        # object
        if linear == False:
            self.psr[signal['pulsarind']].initLibsTempoObject()

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
    def getNumberOfSignalsFromDict(self, signals, stype='powerlaw', corr='single'):
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
    def getSignalNumbersFromDict(self, signals, stype='powerlaw', \
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
    def initModel(self, fullmodel, fromFile=False, write=True, \
                  verbose=False, memsave=True):
        numNoiseFreqs = fullmodel['numNoiseFreqs']
        numDMFreqs = fullmodel['numDMFreqs']
        compression = fullmodel['compression']
        evalCompressionComplement = fullmodel['evalCompressionComplement']
        orderFrequencyLines = fullmodel['orderFrequencyLines']
        likfunc = fullmodel['likfunc']
        signals = fullmodel['signals']
        targetAmp = fullmodel['targetAmp']

        if len(self.psr) < 1:
            raise IOError, "No pulsars loaded"

        if fullmodel['numpulsars'] != len(self.psr):
            raise IOError, "Model does not have the right number of pulsars"

        #if not self.checkSignalDictionary(signals):
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
        self.Tmax = Tmax
        self.Tref = Tstart

        print 'WARNING: Using seperate Tmax for each pulsar'
        for p in self.psr:
            p.Tmax = p.toas.max() - p.toas.min()
            #p.Tmax = self.Tmax

        # If the compressionComplement is defined, overwrite the default
        if evalCompressionComplement != 'None':
            self.evallikcomp = evalCompressionComplement
            self.compression = compression
        elif compression == 'None':
            self.evallikcomp = False
        else:
            self.evallikcomp = True
            self.compression = compression

        # Find out how many single-frequency modes there are
        numSingleFreqs = self.getNumberOfSignalsFromDict(signals, \
                stype='frequencyline', corr='single')
        numSingleDMFreqs = self.getNumberOfSignalsFromDict(signals, \
                stype='dmfrequencyline', corr='single')

        # Find out how many efac signals there are, and translate that to a
        # separateEfacs boolean array (for two-component noise analysis)
        numEfacs = self.getNumberOfSignalsFromDict(signals, \
                stype='efac', corr='single')
        numEquads = self.getNumberOfSignalsFromDict(signals, \
                stype='equad', corr='single')
        numJitter = self.getNumberOfSignalsFromDict(signals, \
                stype='jitter_equad', corr='single')
        incJitter = np.array(numJitter) > 0
        print numJitter, incJitter
        separateEfacs = np.logical_or(numEfacs > 1, numEquads > 1)
        

        # Modify design matrices, and create pulsar Auxiliary quantities
        for pindex, p in enumerate(self.psr):
            # If we model DM variations, we will need to include QSD
            # marginalisation for DM. Modify design matrix accordingly
            #if dmModel[pindex] != 'None':
            #if numDMFreqs[pindex] > 0:
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
                    raise StandardError('Requested to re-create the Auxiliaries')
                # Read Auxiliaries
                if verbose:
                    print "Reading Auxiliaries for {0}".format(p.name)
                p.readPulsarAuxiliaries(self.h5df, p.Tmax, numNoiseFreqs[pindex], \
                        numDMFreqs[pindex], ~separateEfacs[pindex], \
                                nSingleFreqs=numSingleFreqs[pindex], \
                                nSingleDMFreqs=numSingleDMFreqs[pindex], \
                                likfunc=likfunc, compression=compression, \
                                memsave=memsave)
            except (StandardError, ValueError, KeyError, IOError, RuntimeError) as err:
                # Create the Auxiliaries ourselves

                # For every pulsar, construct the auxiliary quantities like the Fourier
                # design matrix etc
                if verbose:
                    print str(err)
                    print "Creating Auxiliaries for {0}".format(p.name)
                p.createPulsarAuxiliaries(self.h5df, p.Tmax, numNoiseFreqs[pindex], \
                        numDMFreqs[pindex], ~separateEfacs[pindex], \
                                nSingleFreqs=numSingleFreqs[pindex], \
                                nSingleDMFreqs=numSingleDMFreqs[pindex], \
                                likfunc=likfunc, compression=compression, \
                                write=write, tmpars=tmpars, memsave=memsave, \
                                         incJitter=incJitter[pindex])

        # Initialise the ptasignal objects
        self.ptasignals = []
        index = 0
        for ii, signal in enumerate(signals):
            self.addSignal(signal, index, p.Tmax)
            index += self.ptasignals[-1]['npars']

        self.allocateLikAuxiliaries()
        self.initPrior()
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
                    flagvalue = 'efac_'+sig['flagvalue']

                elif sig['stype'] == 'equad':
                    flagname = sig['flagname']
                    flagvalue = 'equad_'+sig['flagvalue']

                elif sig['stype'] == 'jitter':
                    flagname = sig['flagname']
                    flagvalue = 'jitter_'+sig['flagvalue']
                
                elif sig['stype'] == 'jitter_equad':
                    flagname = sig['flagname']
                    flagvalue = 'jitter_q_'+sig['flagvalue']
                
                elif sig['stype'] == 'jitter_epoch':
                    flagname = sig['flagname']
                    flagvalue = 'jitter_p_' + str(jj)
                
                elif sig['stype'] == 'nongausscoeff':
                    flagname = sig['flagname']
                    flagvalue = 'alpha_' + str(jj+1)

                elif sig['stype'] == 'pulsardistance':
                    flagname = sig['flagname']
                    flagvalue = 'pdist_'+sig['flagvalue'] 

                elif sig['stype'] == 'spectrum':
                    flagname = 'frequency'
                    #flagvalue = 'rho' + str(jj)
                    if sig['corr'] == 'single':
                        flagvalue = 'red_' + str(self.psr[psrindex].Ffreqs[2*jj])
                    elif sig['corr'] == 'gr':
                        flagvalue = 'gwb_' + str(self.psr[psrindex].Ffreqs[2*jj])

                elif sig['stype'] == 'dmspectrum':
                    flagname = 'dmfrequency'
                    flagvalue = 'dm_' +str(self.psr[psrindex].Fdmfreqs[2*jj])

                elif sig['stype'] == 'cw':
                    flagname = 'cw'
                    flagvalue = sig['parid'][jj]

                elif sig['stype'] == 'powerlaw':
                    flagname = 'powerlaw'

                    if sig['corr'] == 'gr':
                        flagvalue = ['GWB-Amplitude', 'GWB-spectral-index', 'low-frequency-cutoff'][jj]
                    elif sig['corr'] == 'uniform':
                        flagvalue = ['CLK-Amplitude', 'CLK-spectral-index', 'low-frequency-cutoff'][jj]
                    elif sig['corr'] == 'dipole':
                        flagvalue = ['DIP-Amplitude', 'DIP-spectral-index', 'low-frequency-cutoff'][jj]
                    else:
                        flagvalue = ['RN-Amplitude', 'RN-spectral-index', 'low-frequency-cutoff'][jj]

                elif sig['stype'] == 'dmpowerlaw':
                    flagname = 'dmpowerlaw'
                    flagvalue = ['DM-Amplitude', 'DM-spectral-index', 'low-frequency-cutoff'][jj]

                elif sig['stype'] == 'spectralModel':
                    flagname = 'spectralModel'
                    flagvalue = ['SM-Amplitude', 'SM-spectral-index', 'SM-corner-frequency'][jj]

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
                
                #TODO: add BWM and continuous Wave
                #elif sig['stype'] == 'bwm':
                #    flagname = 'BurstWithMemory'
                #    flagvalue = ['burst-arrival', 'amplitude', 'raj', 'decj', 'polarisation'][jj]

                else:
                    flagname = 'none'
                    flagvalue = 'none'

                pardes.append(\
                        {'index': index, 'pulsar': psrindex, 'sigindex': ii, \
                            'sigtype': sig['stype'], 'correlation': sig['corr'], \
                            'name': flagname, 'id': flagvalue})

        return pardes


    """
    Determine intial parameters drawn from prior ranges

    """
    def initParameters(self, startEfacAtOne=True, fixpstart=False):
        
        p0 = []
        for ct, sig in enumerate(self.ptasignals):
            if np.any(sig['bvary']):
                for min, max, pstart, pwidth in zip(sig['pmin'][sig['bvary']], \
                    sig['pmax'][sig['bvary']], sig['pstart'][sig['bvary']],\
                    sig['pwidth'][sig['bvary']]):
                    if startEfacAtOne and sig['stype'] == 'efac':
                        p0.append(1)
                    else:
                        if fixpstart:
                            p0.append(np.double(pstart))
                        else:
                            p0.append(min + np.random.rand() * (max-min))
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
                    cov_diag.append((step)**2)

        cov = np.diag(cov_diag)

        # if we have timing model parameters use fisher matrix
        #for ct, sig in enumerate(self.ptasignals):
        #    if 'timingmodel' in sig['stype']:           
        #        parinds = np.arange(sig['parindex'], sig['parindex']+sig['npars'])
        #        pulsarind = sig['pulsarind']
        #        w = 1/self.psr[pulsarind].toaerrs**2
        #        fishinv = np.dot(self.psr[pulsarind].Mmat.T, \
        #                        (w * self.psr[pulsarind].Mmat.T).T)
        #        fisher = np.sqrt(np.linalg.inv(fishinv))
        #        print np.diag(fisher), sig['pwidth']
        #        for ct1, ii in enumerate(parinds):
        #            for ct2, jj in enumerate(parinds):
        #                cov[ii,jj] = fisher[ct1,ct2]

                    
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
        self.ntmpars = 0
        nphiTmat = 0
        
        for ii, p in enumerate(self.psr):

            # number of red and DM frequencies
            self.npf[ii] = len(p.Ffreqs)
            self.npfdm[ii] = len(p.Fdmfreqs)
            self.npftot[ii] = self.npf[ii] + self.npfdm[ii] + \
                    p.nSingleFreqs*2 + p.nSingleDMFreqs*2
            self.ntmpars += len(p.ptmdescription)

            if self.likfunc == 'mark6':
                nphiTmat += p.Tmat.shape[1] + p.nSingleFreqs*2 + p.nSingleDMFreqs*2

            # noise vectors
            p.Nvec = np.zeros(len(p.toas))
            p.Nwvec = np.zeros(p.nbasis)
            p.Nwovec = np.zeros(p.nobasis)

        # number of GW frequencies
        self.ngwf = np.max(self.npf)
        self.gwfreqs = self.psr[np.argmax(self.npf)].Ffreqs
        nftot = np.max(self.npftot)
        if self.likfunc == 'mark6':
            self.Phiinv = np.zeros((nphiTmat, nphiTmat))
            self.Phi = np.zeros((nphiTmat, nphiTmat))
        else:
            self.Phiinv = np.zeros((nftot*self.npsr, nftot*self.npsr))
            self.Phi = np.zeros((nftot*self.npsr, nftot*self.npsr))

        #for ii in range(self.npsr):
        #    if not self.likfunc in ['mark2']:
        #        self.npf[ii] = len(self.psr[ii].Ffreqs)
        #        self.npff[ii] = self.npf[ii]

        #    if self.likfunc in ['mark4ln', 'mark9', 'mark10']:
        #        self.npff[ii] += len(self.psr[ii].SFfreqs)

        #    if self.likfunc in ['mark4', 'mark4ln']:
        #        self.npu[ii] = len(self.psr[ii].avetoas)

        #    if self.likfunc in ['mark1', 'mark4', 'mark4ln', 'mark6', 'mark6fa', 'mark8', 'mark10']:
        #        self.npfdm[ii] = len(self.psr[ii].Fdmfreqs)
        #        self.npffdm[ii] = len(self.psr[ii].Fdmfreqs)

        #    if self.likfunc in ['mark10']:
        #        self.npffdm[ii] += len(self.psr[ii].SFdmfreqs)

        #    self.npobs[ii] = len(self.psr[ii].toas)
        #    self.npgs[ii] = self.psr[ii].Hmat.shape[1]
        #    self.npgos[ii] = self.psr[ii].Homat.shape[1]
        #    self.psr[ii].Nvec = np.zeros(len(self.psr[ii].toas))
        #    self.psr[ii].Nwvec = np.zeros(self.psr[ii].Hmat.shape[1])
        #    self.psr[ii].Nwovec = np.zeros(self.psr[ii].Homat.shape[1])

        #self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
        #self.Thetavec = np.zeros(np.sum(self.npfdm))

        #if self.likfunc == 'mark1':
        #    self.GNGldet = np.zeros(self.npsr)
        #    self.rGr = np.zeros(self.npsr)

        #    self.Gr = np.zeros(np.sum(self.npgs))
        #    self.GCG = np.zeros((np.sum(self.npgs), np.sum(self.npgs)))
        #elif self.likfunc == 'mark2':
        #    self.GNGldet = np.zeros(self.npsr)
        #    self.rGr = np.zeros(self.npsr)
        #elif self.likfunc == 'mark3' or self.likfunc == 'mark7' \
        #        or self.likfunc == 'mark3fa':
        #    self.Sigma = np.zeros((np.sum(self.npf), np.sum(self.npf)))
        #    self.GNGldet = np.zeros(self.npsr)
        #    self.rGr = np.zeros(self.npsr)
        #    self.rGF = np.zeros(np.sum(self.npf))
        #    self.FGGNGGF = np.zeros((np.sum(self.npf), np.sum(self.npf)))
        #elif self.likfunc == 'mark4':
        #    self.Sigma = np.zeros((np.sum(self.npu), np.sum(self.npu)))
        #    self.GNGldet = np.zeros(self.npsr)
        #    self.rGr = np.zeros(self.npsr)
        #    self.rGU = np.zeros(np.sum(self.npu))
        #    self.UGGNGGU = np.zeros((np.sum(self.npu), np.sum(self.npu)))
        #elif self.likfunc == 'mark4ln':
        #    self.Sigma = np.zeros((np.sum(self.npu), np.sum(self.npu)))
        #    self.GNGldet = np.zeros(self.npsr)
        #    self.rGr = np.zeros(self.npsr)
        #    self.rGU = np.zeros(np.sum(self.npu))
        #    self.UGGNGGU = np.zeros((np.sum(self.npu), np.sum(self.npu)))
        #elif self.likfunc == 'mark6' or self.likfunc == 'mark8' \
        #        or self.likfunc == 'mark6fa':
        #    self.Sigma = np.zeros((np.sum(self.npf)+np.sum(self.npfdm), \
        #            np.sum(self.npf)+np.sum(self.npfdm)))
        #    self.Thetavec = np.zeros(np.sum(self.npfdm))
        #    self.GNGldet = np.zeros(self.npsr)
        #    self.rGr = np.zeros(self.npsr)
        #    self.rGE = np.zeros(np.sum(self.npf)+np.sum(self.npfdm))
        #    self.EGGNGGE = np.zeros((np.sum(self.npf)+np.sum(self.npfdm), np.sum(self.npf)+np.sum(self.npfdm)))
        #elif self.likfunc == 'mark9':
        #    self.Sigma = np.zeros((np.sum(self.npff), np.sum(self.npff)))
        #    self.GNGldet = np.zeros(self.npsr)
        #    self.rGr = np.zeros(self.npsr)
        #    self.rGF = np.zeros(np.sum(self.npff))
        #    self.FGGNGGF = np.zeros((np.sum(self.npff), np.sum(self.npff)))
        #elif self.likfunc == 'mark10':
        #    self.Sigma = np.zeros((np.sum(self.npff)+np.sum(self.npffdm), \
        #            np.sum(self.npff)+np.sum(self.npffdm)))
        #    self.GNGldet = np.zeros(self.npsr)
        #    self.rGr = np.zeros(self.npsr)
        #    self.rGE = np.zeros(np.sum(self.npff)+np.sum(self.npffdm))
        #    self.EGGNGGE = np.zeros((np.sum(self.npff)+np.sum(self.npffdm), \
        #            np.sum(self.npff)+np.sum(self.npffdm)))


   
    """
    Loop over all signals, and fill the diagonal pulsar noise covariance matrix
    (based on efac/equad)
    For two-component noise model, fill the total weights vector

    @param parameters:  The total parameters vector
    @param selection:   Boolean array, indicating which parameters to include
    @param incJitter:   Whether or not to include Jitter in the noise vector,
                        Should only be included if using 'average' compression

    """
    def setPsrNoise(self, parameters, incJitter=True, twoComponent=True):

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
                    self.psr[psrind].Nwvec += self.psr[psrind].Wvec * pefac**2
                    self.psr[psrind].Nwovec += self.psr[psrind].Wovec * pefac**2
                    self.psr[psrind].Nvec += sig['Nvec'] * pefac**2

                else:   # use Nvec stored in dictionary
                    self.psr[psrind].Nvec += sig['Nvec'] * pefac**2
            
            # equad signal
            elif sig['stype'] == 'equad':

                # is this parameter being varied
                if sig['npars'] == 1:
                    pequadsqr = 10**(2*parameters[parind])
                
                # if not use reference value
                else:
                    pequadsqr = 10**(2*sig['pstart'][0])

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
                    pequadsqr = parameters[parind]**2

                # if not use reference value
                else:
                    pequadsqr = sig['pstart'][0]**2
                
                if self.likfunc in ['mark2', 'mark6']:
                    self.psr[psrind].Qamp += sig['Jvec'] * pequadsqr
                else:
                    self.psr[psrind].Nvec += sig['Nvec'] * pequadsqr
            
            # jitter equad signal
            elif sig['stype'] == 'jitter_equad':

                # is this parameter being varied
                if sig['npars'] == 1:
                    pequadsqr = 10**(2*parameters[parind])

                # if not use reference value
                else:
                    pequadsqr = 10**(2*sig['pstart'][0])

                self.psr[psrind].Qamp += sig['Jvec'] * pequadsqr

            # jitter by epoch signal
            elif sig['stype'] == 'jitter_epoch':

                # short hand
                npars = sig['npars']
            
                # parameters for this signal
                sparameters = sig['pstart'].copy()

                # which ones are varying
                sparameters[sig['bvary']] = parameters[parind:parind+npars]

                self.psr[psrind].Qamp += 10**(2*sparameters)


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
            sparameters[sig['bvary']] = parameters[parind:parind+npars]
            
            if sig['stype'] == 'frequencyline':
                findex = sig['npsrfreqindex']
                self.psr[psrind].SFfreqs[findex] = 10**sparameters[0]
                addedSingle = True

            if sig['stype'] == 'dmfrequencyline':
                findex = sig['npsrfreqindex']
                self.psr[psrind].DMSFfreqs[findex] = 10**sparameters[0]
                addedDMSingle = True

        # loop over all pulsars and re-construct F matrices and needed auxiliaries
        for ct, p in enumerate(self.psr):

            # if using average likelihood
            if self.likfunc == 'mark2': 
                Ftemp = p.UtF.copy()

                # added frequency independent line
                if addedSingle:
                    SFmat = PALutils.singlefourierdesignmatrix(p.avetoas, p.SFfreqs)
                    Ftemp = np.append(Ftemp, SFmat, axis=1)

                # added DM line
                if addedDMSingle:
                    SdmFmat = PALutils.singlefourierdesignmatrix(p.avetoas, p.DMSFfreqs)
                    Dmat = 4.15e3 / (p.avefreqs**2)
                    SDMFmat = (Dmat * SdmFmat.T).T
                    Ftemp = np.append(Ftemp, SDMFmat, axis=1)

                # final product
                p.UtFF = Ftemp

            else:
                Ftemp = p.Ftot.copy()

                # added frequency independent line
                if addedSingle:
                    SFmat = PALutils.singlefourierdesignmatrix(p.toas, p.SFfreqs)
                    Ftemp = np.append(Ftemp, SFmat, axis=1)
                # added DM line
                if addedDMSingle:
                    SdmFmat = PALutils.singlefourierdesignmatrix(p.toas, p.DMFSfreqs)
                    Dmat = 4.15e3 / (p.freqs**2)
                    SDMFmat = (Dmat * SdmFmat.T).T
                    Ftemp = np.append(Ftemp, SDMFmat, axis=1)

                # two component noise stuff
                if p.twoComponentNoise:
                    GtF = np.dot(p.Hmat.T, Ftemp)
                    p.AGFF = np.dot(p.Amat.T, GtF)
                else:
                    p.FFtot = Ftemp






                



    """
    Construct complete Phi inverse matrix, including DM corrections if needed

    TODO: this code only works if all pulsars have the same number of frequencies
    want to make this more flexible
    """
    def constructPhiMatrix(self, parameters, constructPhi=False, \
                           incCorrelations=True, incTM=False, incJitter=False):

        
        # Loop over all signals and determine rho (GW signals) and kappa (red + DM signals)
        rho = None
        for ss, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']
            
            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind+npars]

            # spectrum
            if sig['stype'] == 'spectrum':

                # pulsar independent red noise spectrum
                if sig['corr'] == 'single':

                    # doubled amplitudes 
                    pcdoubled = np.array([sparameters, sparameters]).T.flatten()

                    # fill in kappa
                    self.psr[psrind].kappa = pcdoubled

                # correlated signals
                if sig['corr'] in ['gr', 'uniform', 'dipole']:
                    
                    # correlation matrix
                    self.corrmat = sig['corrmat']

                    # define rho
                    rho = np.array([sparameters, sparameters]).T.flatten()

            # spectral Model
            if sig['stype'] == 'spectralModel':

                # pulsar independent 
                if sig['corr'] == 'single':
                    PAL_spy = 3.16e7
                    Amp = 10**sparameters[0]
                    alpha = sparameters[1]
                    fc = 10**sparameters[2] / PAL_spy
                    freqpy = self.psr[psrind].Ffreqs

                    pcdoubled = np.log10((Amp * PAL_spy**3 / sig['Tmax']) * \
                            ((1 + (freqpy/fc)**2)**(-0.5*alpha)))
                    
                    # fill in kappa
                    self.psr[psrind].kappa = pcdoubled
                

            # powerlaw spectrum
            if sig['stype'] == 'powerlaw':

                # pulsar independend red noise powerlaw
                if sig['corr'] == 'single':

                    # get Amplitude and spectral index
                    Amp = 10**sparameters[0]
                    gamma = sparameters[1]
                    
                    freqpy = self.psr[psrind].Ffreqs
                    f1yr = 1/3.16e7
                    #f1yr = 1/self.psr[psrind].Tmax 
                    pcdoubled = np.log10(Amp**2/12/np.pi**2 * f1yr**(gamma-3) * \
                                         freqpy**(-gamma)/sig['Tmax'])

                    # fill in kappa
                    self.psr[psrind].kappa = pcdoubled

                # correlated signals
                if sig['corr'] in ['gr', 'uniform', 'dipole']:

                    # correlation matrix
                    self.corrmat = sig['corrmat']
                    
                    # number of GW frequencies is the max from all pulsars
                    fgw = self.gwfreqs

                    # get Amplitude and spectral index
                    Amp = 10**sparameters[0]
                    gamma = sparameters[1]
                    
                    f1yr = 1/3.16e7
                    rho = np.log10(Amp**2/12/np.pi**2 * f1yr**(gamma-3) * \
                                         fgw**(-gamma)/sig['Tmax'])


          # DM spectrum
            if sig['stype'] == 'dmspectrum':

                # pulsar independent DM noise spectrum
                if sig['corr'] == 'single':

                    # doubled amplitudes 
                    pcdoubled = np.array([sparameters, sparameters]).T.flatten()

                    # fill in kappa
                    self.psr[psrind].kappadm = pcdoubled


            # powerlaw DM spectrum
            if sig['stype'] == 'dmpowerlaw':

                # pulsar independend red noise powerlaw
                if sig['corr'] == 'single':

                    # get Amplitude and spectral index
                    Amp = 10**sparameters[0]
                    gamma = sparameters[1]
                    
                    freqpy = self.psr[psrind].Fdmfreqs
                    f1yr = 1/3.16e7
                    pcdoubled = np.log10(Amp**2/12/np.pi**2 * f1yr**(gamma-3) * \
                                         freqpy**(-gamma)/sig['Tmax'])

                    # fill in kappa
                    self.psr[psrind].kappadm = pcdoubled

            # frequency line
            if sig['stype'] == 'frequencyline':
                    
                # doubled amplitudes 
                pcdoubled = np.array([sparameters[1], sparameters[1]]).T.flatten()

                # fill in kappa
                self.psr[psrind].kappasingle = pcdoubled

            # dm frequency line
            if sig['stype'] == 'dmfrequencyline':
                    
                # doubled amplitudes 
                pcdoubled = np.array([sparameters[1], sparameters[1]]).T.flatten()

                # fill in kappa
                self.psr[psrind].kappadmsingle = pcdoubled


        # now that we have obtained rho and kappa, we can construct Phiinv
        sigdiag = []
        sigoffdiag = []
        self.gwamp = 0

        # no correlated signals (easy)
        if rho is None:

            # loop over all pulsars
            for ii, p in enumerate(self.psr):

                # have both red noise and DM variations
                if p.incRed and p.incDM:
                #if np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = np.concatenate((p.kappa, p.kappadm))
                
                # red noise but no dm
                elif p.incRed and not(p.incDM):
                #elif np.any(p.kappa) and ~np.any(p.kappadm):
                    p.kappa_tot = p.kappa.copy()
                
                # dm but no red noise
                elif not(p.incRed) and p.incDM:
                #elif ~np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = p.kappadm.copy()
                
                # neither
                else:
                    p.kappa_tot = np.ones(p.kappa.shape) * -40

                if p.nSingleFreqs > 0:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappasingle))
                
                if p.nSingleDMFreqs > 0:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappadmsingle))

                # append to signal diagonal
                if incTM:
                    p.kappa_tot = np.concatenate((np.ones(p.Mmat.shape[1])*80, \
                                                 p.kappa_tot))
                    if incJitter:
                        p.kappa_tot = np.concatenate((p.kappa_tot, np.log10(p.Qamp)))

                sigdiag.append(10**p.kappa_tot)

            # convert to array and flatten
            self.Phi = np.array(sigdiag).flatten()
            np.fill_diagonal(self.Phiinv, 1/self.Phi)
            #self.logdetPhi = np.sum(np.log(self.Phi[-self.npftot[ii]:]))
            self.logdetPhi = np.sum(np.log(self.Phi[:]))

        # Do not include correlations but include GWB in red noise
        if rho is not None and not(incCorrelations):

            # loop over all pulsars
            for ii, p in enumerate(self.psr):

                # have both red noise and DM variations
                if p.incRed and p.incDM:
                #if np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = np.concatenate((p.kappa, p.kappadm))
                
                # red noise but no dm
                elif p.incRed and not(p.incDM):
                #elif np.any(p.kappa) and ~np.any(p.kappadm):
                    p.kappa_tot = p.kappa.copy()
                
                # dm but no red noise
                elif not(p.incRed) and p.incDM:
                #elif ~np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = p.kappadm.copy()
                
                # neither
                else:
                    p.kappa_tot = np.ones(p.kappa.shape) * -40
                
                if p.nSingleFreqs > 0:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappasingle))
                
                if p.nSingleDMFreqs > 0:
                    p.kappa_tot = np.concatenate((p.kappa_tot, p.kappadmsingle))

                # get number of DM freqs (not included in GW spectrum)
                ndmfreq = np.sum(p.kappadm != 0)

                # append to rho
                if ndmfreq > 0:
                    self.gwamp = np.concatenate((10**rho, np.zeros(ndmfreq)))
                else:
                    self.gwamp = 10**rho
                
                # append to signal diagonal
                if incTM:
                    p.kappa_tot = np.concatenate((np.ones(p.Mmat.shape[1])*80, \
                                                 p.kappa_tot))
                    if incJitter:
                        p.kappa_tot = np.concatenate((p.kappa_tot, np.log10(p.Qamp)))

                # append to signal diagonal
                sigdiag.append(10**p.kappa_tot+self.gwamp)

            # convert to array and flatten
            self.Phi = np.array(sigdiag).flatten()
            np.fill_diagonal(self.Phiinv, 1/self.Phi)
            self.logdetPhi = np.sum(np.log(self.Phi[-self.npftot[ii]:]))
            self.logdetPhi = np.sum(np.log(self.Phi))


        # correlated signals (not as easy)
        if rho is not None and incCorrelations:
        
            for ii, p in enumerate(self.psr):

                # have both red noise and DM variations
                if p.incRed and p.incDM:
                #if np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = np.concatenate((p.kappa, p.kappadm))
                
                # red noise but no dm
                elif p.incRed and not(p.incDM):
                #elif np.any(p.kappa) and ~np.any(p.kappadm):
                    p.kappa_tot = p.kappa.copy()
                
                # dm but no red noise
                elif not(p.incRed) and p.incDM:
                #elif ~np.any(p.kappa) and np.any(p.kappadm):
                    p.kappa_tot = p.kappadm.copy()
                
                # for now, assume that GW freqs is the same as 
                # number of freqs per pulsar

                # get number of DM freqs (not included in GW spectrum)
                ndmfreq = np.sum(p.kappadm != 0)

                # append to rho
                if ndmfreq > 0:
                    self.gwamp = np.concatenate((10**rho, np.zeros(ndmfreq)))
                else:
                    self.gwamp = 10**rho

                # append to diagonal elements
                if len(p.kappa_tot) > 0:
                    sigdiag.append(10**p.kappa_tot + self.gwamp)
                else:
                    sigdiag.append(self.gwamp)
                
                # append to off diagonal elements
                sigoffdiag.append(self.gwamp)


            # compute Phi inverse from Lindley's code
            nftot = self.ngwf + np.max(self.npfdm)
            smallMatrix = np.zeros((nftot, self.npsr, self.npsr))
            for ii in range(self.npsr):
                #smallMatrix[:,ii,ii] = self.corrmat[ii,ii] * sigdiag[ii]
                #if ii < self.npsr -1:
                #    jj = np.arange(ii+1, self.npsr) 
                #    print type(jj)
                #    smallMatrix[:,ii,jj] = self.corrmat[ii,jj] * sigoffdiag[ii]
                #    smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]

                for jj in range(ii, self.npsr):
                    if ii == jj:
                        smallMatrix[:,ii,jj] = self.corrmat[ii,jj] * sigdiag[jj]
                    else:
                        smallMatrix[:,ii,jj] = self.corrmat[ii,jj] * sigoffdiag[jj]
                        smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]

            # invert them
            self.logdetPhi = 0
            for ii in range(nftot):
                L = sl.cho_factor(smallMatrix[ii,:,:])
                smallMatrix[ii,:,:] = sl.cho_solve(L, np.eye(self.npsr))
                self.logdetPhi += np.sum(2*np.log(np.diag(L[0])))

            
            # now fill in real covariance matrix
            ind2 = [np.arange(jj*nftot, jj*nftot+nftot) for jj in range(self.npsr)]
            for ii in range(self.npsr):
                ind1 = np.arange(ii*nftot, ii*nftot+nftot)
                for jj in range(0, self.npsr):
                    self.Phiinv[ind1,ind2[jj]] = smallMatrix[:,ii,jj]

    """
    Function to contruct non-gaussian signal
    coefficients.
    """

    def getNonGaussianComponents(self, parameters):

        for ss, sig in enumerate(self.ptasignals):

            # Create a parameters array for this particular signal
            sparameters = sig['pstart'].copy()
            sparameters[sig['bvary']] = \
                    parameters[sig['parindex']:sig['parindex']+sig['npars']]

            # shorthand
            psrind = sig['pulsarind']

            if sig['stype'] == 'nongausscoeff':
                self.psr[psrind].nalpha = sig['npars'] + 1
                self.psr[psrind].alphacoeff = sparameters

                # prepend 0th component
                val = np.sqrt(1-np.sum(self.psr[psrind].alphacoeff**2))
                self.psr[psrind].alphacoeff = np.insert(self.psr[psrind].alphacoeff, 0, val)


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
                    parameters[sig['parindex']:sig['parindex']+sig['npars']]

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
                        if sig['parid'][jj] == 'Offset':
                            offset += [sparameters[pindex]]
                        elif sig['parid'][jj] == 'F0':
                            offset += [sparameters[pindex]]
                        elif sig['parid'][jj] == 'F1':
                            offset += [sparameters[pindex]]
                        else:
                            offset += [sparameters[pindex]-sig['pstart'][jj]]

                        pindex += 1

                # residuals = M * pars
                psr.detresiduals -= np.dot(Mmat, np.array(offset))

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
                                    psr.ptmpars[jj] + psr.ptmparerrs[jj] * sparameters[pindex]
                        elif sig['parid'][jj] == 'Offset':
                            offset[:] = psr.ptmparerrs[jj] * sparameters[pindex]
                        else:
                            psr.t2psr[sig['parid'][jj]].val = np.longdouble(sparameters[pindex])
                        pindex += 1

                # Generate the new residuals
                psr.detresiduals = np.array(psr.t2psr.residuals(updatebats=True), 
                                            dtype=np.double) + offset

        
            # continuous wave signal
            elif sig['stype'] == 'cw':

                # get pulsar distances  
                nsigs = self.getNumberOfSignalsFromDict(self.ptasignals, \
                    stype='pulsardistance', corr='single')
                
                # including pulsar term
                pdist = []
                if np.any(nsigs):
                    incPterm = True
                    signum = self.getSignalNumbersFromDict(self.ptasignals, \
                        stype='pulsardistance', corr='single')

                    # check to make sure we have all distances
                    if len(signum) != self.npsr:
                        raise ValueError('ERROR: Number of pulsar distances != number of pulsars!')
            
                    # current timing model parameters
                    for signum0 in signum:
                        sig0 = self.ptasignals[signum0]
                        pdist.append(parameters[sig0['parindex']:(sig0['parindex']+ \
                                                                  sig0['npars'])])
                    pdist = np.array(pdist).copy()

                else:
                    incPterm = False
                    pdist = None

                # construct CW signal
                cwsig = PALutils.createResidualsFast(self.psr, sparameters[0], \
                            sparameters[1], 10**sparameters[2], 10**sparameters[3], \
                            10**sparameters[4], sparameters[5], sparameters[6], \
                            sparameters[7], pdist=pdist, psrTerm=incPterm, 
                            phase_approx=True, evolve=False, tref=self.Tref)

                
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
    def simData(self, parameters, setup=False, turnover=False, f0=1e-9, \
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
            gwbs = PALutils.createGWB(self.psr, 10**parameters[-2], parameters[-1], \
                                      DM=False, noCorr=False, seed=None,
                                      turnover=turnover, f0=f0, beta=beta, \
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
            white = n*w

            # jitter noise
            if p.Umat is not None:
                j = np.sqrt(p.Qamp)
                w = np.random.randn(len(p.avetoas))
                white += np.dot(p.Umat, j*w)

            # red noise
            phi = np.sqrt(10**p.kappa_tot)
            x = np.random.randn(npftot)
            red = np.dot(p.Ftot, phi*x)

            # gwb noise
            if np.any(self.gwamp):
                gwb = gwbs[ct]
                #gwphi = np.sqrt(self.gwamp)
                #phiy = gwphi*ypsr[ct,:] 
                #gwb = np.dot(p.Ftot, phiy)
            else:
                gwb = 0
            
            # add residuals
            res.append(white+red+gwb)

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
                
                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component basis
                X.append(np.dot(p.AGF.T, p.AGr/p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1/p.Nwvec) * p.AGF.T).T
                FGGNGGF.append(np.dot(p.AGF.T, right))

            else:   

                # G(G^TNG)^{-1}G^T = N^{-1} - N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0/p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiF = ((1.0/p.Nvec) * p.Ftot.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiF = np.dot(NiGc.T, p.Ftot)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(np.log(p.Nvec)) + 2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"
                
                #F^TG(G^TNG)^{-1}G^T\delta t 
                X.append(np.dot(p.Ftot.T, Nir - NiGcNiGcr))

                # compute F^TG(G^TNG)^{-1}G^TF
                FGGNGGF.append(np.dot(NiF.T, p.Ftot) - np.dot(GcNiF.T, GcNiGcF))

            
            # compute relevant quantities
            nf = len(p.Ffreqs) + len(p.Fdmfreqs)
            #phiinv = 1/self.Phi[ct*nf:(ct*nf+nf)]
            phiinv = np.diag(self.Phiinv)[ct*nf:(ct*nf+nf)]
            Sigma = np.diag(phiinv) + FGGNGGF[ct]
        
            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Sigma)
                right = sl.cho_solve(cf, FGGNGGF[ct])
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                right = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, FGGNGGF[ct])))

            Y.append(X[ct] - np.dot(X[ct], right))

            Z.append(FGGNGGF[ct] - np.dot(FGGNGGF[ct], right))


        # cross correlations
        top = 0
        bot = 0
        rho, sig, xi = [], [], []
        for ii in range(self.npsr):
            for jj in range(ii+1, self.npsr):

                fgw = self.psr[ii].Ffreqs

                # get Amplitude and spectral index
                Amp = 1
                gamma = 13/3
                
                f1yr = 1/3.16e7
                pcdoubled = Amp**2/12/np.pi**2 * f1yr**(gamma-3) * \
                                     fgw**(-gamma)/self.Tmax

                phiIJ =  0.5 * np.concatenate((pcdoubled, \
                                    np.zeros(len(self.psr[ii].Fdmfreqs))))


                top = np.dot(Y[ii], phiIJ * Y[jj])
                bot = np.trace(np.dot((Z[ii]*phiIJ.T).T, (Z[jj]*phiIJ.T).T))

                # cross correlation and uncertainty
                rho.append(top/bot)
                sig.append(1/np.sqrt(bot))
                xi.append(PALutils.angularSeparation(self.psr[ii].theta[0],\
                                                     self.psr[ii].phi[0],\
                                                    self.psr[jj].theta[0],\
                                                     self.psr[jj].phi[0]))

        # return Opt, sigma, snr

        #return top/bot, 1/np.sqrt(bot), top/np.sqrt(bot)
        return np.array(xi), np.array(rho), np.array(sig), \
                np.sum(np.array(rho)*ORF/np.array(sig)**2)/np.sum(ORF**2/np.array(sig)**2), \
                1/np.sqrt(np.sum(ORF**2/np.array(sig)**2))

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
                
                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component basis
                X.append(np.dot(p.AGU.T, p.AGr/p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1/p.Nwvec) * p.AGU.T).T
                UGGNGGU.append(np.dot(p.AGU.T, right))

            else:   

                # G(G^TNG)^{-1}G^T = N^{-1} - N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0/p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiU = ((1.0/p.Nvec) * p.Umat.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiU = np.dot(NiGc.T, p.Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(np.log(p.Nvec)) + 2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"
                
                #F^TG(G^TNG)^{-1}G^T\delta t 
                X.append(np.dot(p.Umat.T, Nir - NiGcNiGcr))

                # compute F^TG(G^TNG)^{-1}G^TF
                UGGNGGU.append(np.dot(NiU.T, p.Umat) - np.dot(GcNiU.T, GcNiGcU))

            # construct modified phi matrix
            nf = len(p.Ffreqs) + len(p.Fdmfreqs)
            Phi0 = np.diag(self.Phi[ct*nf:(ct*nf+nf)])
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
                phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))


            Sigma = phiinv + UGGNGGU[ct]
        
            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Sigma)
                right = sl.cho_solve(cf, UGGNGGU[ct])
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                right = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, UGGNGGU[ct])))

            Y.append(X[ct] - np.dot(X[ct], right))

            Z.append(UGGNGGU[ct] - np.dot(UGGNGGU[ct], right))


        # cross correlations
        top = 0
        bot = 0
        rho, sig, xi = [], [], []
        for ii in range(self.npsr):
            for jj in range(ii+1, self.npsr):

                fgw = self.psr[ii].Ffreqs
                nf = len(fgw)
                nfdm = len(self.psr[ii].Fdmfreqs)

                # get Amplitude and spectral index
                Amp = 1
                gamma = 13/3
                
                f1yr = 1/3.16e7
                pcdoubled = 0.5 * Amp**2/12/np.pi**2 * f1yr**(gamma-3) * \
                                     fgw**(-gamma)/self.Tmax
                    
                Phi = np.zeros(nf+nfdm)
                #di = np.diag_indices(nf)
                Phi[:nf] = pcdoubled
        
                phiIJ = np.dot(self.psr[ii].UtF, (Phi * self.psr[jj].UtF).T)
        
                top = np.dot(Y[ii], np.dot(phiIJ, Y[jj]))
                bot = np.trace(np.dot(Z[ii], np.dot(phiIJ, np.dot(Z[jj], phiIJ.T))))
            
                # cross correlation and uncertainty
                rho.append(top/bot)
                sig.append(1/np.sqrt(bot))
                xi.append(PALutils.angularSeparation(self.psr[ii].theta[0], self.psr[ii].phi[0] \
                                                  , self.psr[jj].theta[0], self.psr[jj].phi[0]))

        # return Opt, sigma, snr

        #return top/bot, 1/np.sqrt(bot), top/np.sqrt(bot)
        return np.array(xi), np.array(rho), np.array(sig), \
                np.sum(np.array(rho)*ORF/np.array(sig)**2)/np.sum(ORF**2/np.array(sig)**2), \
                1/np.sqrt(np.sum(ORF**2/np.array(sig)**2))

             

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
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        
        # compute the white noise terms in the log likelihood
        FGGNGGF = []
        nfref = 0
        for ct, p in enumerate(self.psr):

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(np.isinf(p.detresiduals)):
                return -np.inf
                        
            if p.twoComponentNoise:
                
                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component basis
                if ct == 0:
                    d = np.dot(p.AGFF.T, p.AGr/p.Nwvec)
                else:
                    d = np.append(d, np.dot(p.AGFF.T, p.AGr/p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1/p.Nwvec) * p.AGFF.T).T
                FGGNGGF.append(np.dot(p.AGFF.T, right))

                # log determinant of G^TNG
                logdet_N = np.sum(np.log(p.Nwvec))

                # triple product in likelihood function
                rGGNGGr = np.sum(p.AGr**2/p.Nwvec)
            
            else:   

                # G(G^TNG)^{-1}G^T = N^{-1} - N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0/p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiF = ((1.0/p.Nvec) * p.FFtot.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiF = np.dot(NiGc.T, p.FFtot)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(np.log(p.Nvec)) + 2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"
                
                #F^TG(G^TNG)^{-1}G^T\delta t 
                if ct == 0:
                    d = np.dot(p.FFtot.T, Nir - NiGcNiGcr)
                else:
                    d = np.append(d, np.dot(p.FFtot.T, Nir - NiGcNiGcr))

                # triple product in likelihood function
                rGGNGGr = np.dot(p.detresiduals, Nir) - np.dot(GcNir, GcNiGcr)

                # compute F^TG(G^TNG)^{-1}G^TF
                FGGNGGF.append(np.dot(NiF.T, p.FFtot) - np.dot(GcNiF.T, GcNiGcF))

                
            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rGGNGGr)
            
            # calculate red noise piece
            if not incCorrelations:

                # compute sigma
                logdet_Sigma = 0
                nf = self.npftot[ct]
                Sigma = FGGNGGF[ct] + self.Phiinv[nfref:(nfref+nf), nfref:(nfref+nf)]
                dd = d[nfref:(nfref+nf)]

                # cholesky decomp for maximum likelihood fourier components
                try:
                    cf = sl.cho_factor(Sigma)
                    expval2 = sl.cho_solve(cf, dd)
                    logdet_Sigma += np.sum(2*np.log(np.diag(cf[0])))
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
                logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                logdet_Sigma = np.sum(np.log(s))
                expval2 = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, d)))


            loglike += -0.5 * (self.logdetPhi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) 

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
                
                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component basis
                if ct == 0:
                    d = np.dot(p.AGU.T, p.AGr/p.Nwvec)
                else:
                    d = np.append(d, np.dot(p.AGU.T, p.AGr/p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1/p.Nwvec) * p.AGU.T).T
                UGGNGGU.append(np.dot(p.AGU.T, right))

                # log determinant of G^TNG
                logdet_N = np.sum(np.log(p.Nwvec))

                # triple product in likelihood function
                rGGNGGr = np.sum(p.AGr**2/p.Nwvec)
            
            else:   

                # G(G^TNG)^{-1}G^T = N^{-1} - N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0/p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiU = ((1.0/p.Nvec) * p.Umat.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiU = np.dot(NiGc.T, p.Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(np.log(p.Nvec)) + 2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"
                
                #F^TG(G^TNG)^{-1}G^T\delta t 
                if ct == 0:
                    d = np.dot(p.Umat.T, Nir - NiGcNiGcr)
                else:
                    d = np.append(d, np.dot(p.Umat.T, Nir - NiGcNiGcr))

                # triple product in likelihood function
                rGGNGGr = np.dot(p.detresiduals, Nir) - np.dot(GcNir, GcNiGcr)

                # compute F^TG(G^TNG)^{-1}G^TF
                UGGNGGU.append(np.dot(NiU.T, p.Umat) - np.dot(GcNiU.T, GcNiGcU))
                
                
            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rGGNGGr)

        # cheat for now
        #TODO: make this more general
        if self.npsr == 1:
            Phi0 = np.diag(1/np.diag(self.Phiinv))
            UPhiU = np.dot(self.psr[0].UtFF, np.dot(Phi0, self.psr[0].UtFF.T))
            Phi = UPhiU + np.diag(self.psr[0].Qamp) 
            
            try:
                cf = sl.cho_factor(Phi)
                self.logdetPhi = 2*np.sum(np.log(np.diag(cf[0])))
                self.Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                print 'ERROR: Cholesky Failed when inverting Phi0'
                print parameters
                #U, s, Vh = sl.svd(Phi)
                #if not np.all(s > 0):
                #    print "ERROR: Sigma singular according to SVD when inverting Phi0"
                return -np.inf
                    #raise ValueError("ERROR: Phi singular according to SVD")
                #self.logdetPhi = np.sum(np.log(s))
                #self.Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

        else:
            raise ValueError("ERROR: Have not yet implemented jitter for multiple pulsars")

        # compute the red noise, DMV and GWB terms in the log likelihood
        
        # compute sigma
        Sigma = sl.block_diag(*UGGNGGU) + self.Phiinv

        # cholesky decomp for second term in exponential
        try:
            cf = sl.cho_factor(Sigma)
            logdet_Sigma = 2*np.sum(np.log(np.diag(cf[0])))
            expval2 = sl.cho_solve(cf, d)
        except np.linalg.LinAlgError:
            print 'Cholesky failed when inverting Sigma'
            print parameters
            #U, s, Vh = sl.svd(Sigma)
            #if not np.all(s > 0):
                #print "ERROR: Sigma singular according to SVD when inverting Sigma"
            return -np.inf
                #raise ValueError("ERROR: Sigma singular according to SVD")
            logdet_Sigma = np.sum(np.log(s))
            expval2 = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, d)))


        loglike += -0.5 * (self.logdetPhi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) 

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
                
                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component basis
                if ct == 0:
                    d = np.dot(p.AGU.T, p.AGr/p.Nwvec)
                else:
                    d = np.append(d, np.dot(p.AGU.T, p.AGr/p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1/p.Nwvec) * p.AGU.T).T
                UGGNGGU.append(np.dot(p.AGU.T, right))

                # log determinant of G^TNG
                logdet_N = np.sum(np.log(p.Nwvec))

                # triple product in likelihood function
                rGGNGGr = np.sum(p.AGr**2/p.Nwvec)
            
            else:   

                # G(G^TNG)^{-1}G^T = N^{-1} - N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0/p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiU = ((1.0/p.Nvec) * p.Umat.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiU = np.dot(NiGc.T, p.Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    logdet_N = np.sum(np.log(p.Nvec)) + 2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"
                
                #F^TG(G^TNG)^{-1}G^T\delta t 
                if ct == 0:
                    d = np.dot(p.Umat.T, Nir - NiGcNiGcr)
                else:
                    d = np.append(d, np.dot(p.Umat.T, Nir - NiGcNiGcr))

                # triple product in likelihood function
                rGGNGGr = np.dot(p.detresiduals, Nir) - np.dot(GcNir, GcNiGcr)

                # compute F^TG(G^TNG)^{-1}G^TF
                UGGNGGU.append(np.dot(NiU.T, p.Umat) - np.dot(GcNiU.T, GcNiGcU))
                
                
            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rGGNGGr)

            # keep track of jitter terms needed later
            if self.npsr > 1 and incCorrelations:
                if ct == 0:
                    Jinv = 1/p.Qamp
                else:
                    Jinv = np.append(Jinv, 1/p.Qamp)

                FJ.append(p.UtFF.T * 1/p.Qamp)
                FJF.append(np.dot(FJ[ct], p.UtFF))


        # if only using one pulsar
        if self.npsr == 1 or not(incCorrelations):
            logdetPhi = 0
            tmp = []
            for ct, p in enumerate(self.psr):
                Phi0 = np.diag(10**p.kappa_tot+self.gwamp)
                UPhiU = np.dot(p.UtFF, np.dot(Phi0, p.UtFF.T))
                Phi = UPhiU + np.diag(p.Qamp) 

            
                try:
                    cf = sl.cho_factor(Phi)
                    logdetPhi += 2*np.sum(np.log(np.diag(cf[0])))
                    tmp.append(sl.cho_solve(cf, np.identity(Phi.shape[0])))
                except np.linalg.LinAlgError:
                    #print 'Cholesky failed when inverting phi'
                    #U, s, Vh = sl.svd(Phi)
                    #if not np.all(s > 0):
                    return -np.inf
                        #raise ValueError("ERROR: Phi singular according to SVD")
                    #logdetPhi = np.sum(np.log(s))
                    #Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))
            
            # block diagonal matrix
            Phiinv = sl.block_diag(*tmp)

        else:
            
            Phi0 = self.Phiinv + sl.block_diag(*FJF)
            logdet_J = np.sum(np.log(1/Jinv))

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Phi0)
                logdet_Phi0 = 2*np.sum(np.log(np.diag(cf[0])))
                PhiinvFJ = sl.cho_solve(cf, sl.block_diag(*FJ))
            except np.linalg.LinAlgError:
                print 'Cholesky Failed when inverting Phi0'
                return -np.inf
                #U, s, Vh = sl.svd(Phi0)
                #if not np.all(s > 0):
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
            logdet_Sigma = 2*np.sum(np.log(np.diag(cf[0])))
            expval2 = sl.cho_solve(cf, d)
        except np.linalg.LinAlgError:
            #print 'Cholesky Failed when inverting Sigma'
            return -np.inf
            #return -np.inf
            #U, s, Vh = sl.svd(Sigma)
            #if not np.all(s > 0):
                #return -np.inf
            #    raise ValueError("ERROR: Sigma singular according to SVD")
            #logdet_Sigma = np.sum(np.log(s))
            #expval2 = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, d)))


        loglike += -0.5 * (logdetPhi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) 

        return loglike
    
    """
    mark 2 fixed noise likelihood (test)

    EFAC + EQUAD + Jitter +Red noise + DMV + GWs

    No frequency lines

    Uses Woodbury lemma

    """

    def mark2LogLikelihood_fixedNoise(self, parameters, incCorrelations=True, setup=False):

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
                    
                    # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component basis
                    if ct == 0:
                        self.d = np.dot(p.AGU.T, p.AGr/p.Nwvec)
                    else:
                        self.d = np.append(self.d, np.dot(p.AGU.T, p.AGr/p.Nwvec))

                    # compute F^TG(G^TNG)^{-1}G^TF
                    right = ((1/p.Nwvec) * p.AGU.T).T
                    self.UGGNGGU.append(np.dot(p.AGU.T, right))

                    # log determinant of G^TNG
                    self.logdet_N.append(np.sum(np.log(p.Nwvec)))

                    # triple product in likelihood function
                    self.rGGNGGr.append(np.sum(p.AGr**2/p.Nwvec))
                
                else:   

                    # G(G^TNG)^{-1}G^T = N^{-1} - N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                    Nir = p.detresiduals / p.Nvec
                    NiGc = ((1.0/p.Nvec) * p.Hcmat.T).T
                    GcNiGc = np.dot(p.Hcmat.T, NiGc)
                    NiU = ((1.0/p.Nvec) * p.Umat.T).T
                    GcNir = np.dot(NiGc.T, p.detresiduals)
                    GcNiU = np.dot(NiGc.T, p.Umat)

                    try:
                        cf = sl.cho_factor(GcNiGc)
                        self.logdet_N.append(np.sum(np.log(p.Nvec)) + \
                                             2*np.sum(np.log(np.diag(cf[0]))))
                        GcNiGcr = sl.cho_solve(cf, GcNir)
                        GcNiGcU = sl.cho_solve(cf, GcNiU)
                        NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                    except np.linalg.LinAlgError:
                        print "MAJOR ERROR"
                    
                    #F^TG(G^TNG)^{-1}G^T\delta t 
                    if ct == 0:
                        self.d = np.dot(p.Umat.T, Nir - NiGcNiGcr)
                    else:
                        self.d = np.append(self.d, np.dot(p.Umat.T, Nir - NiGcNiGcr))

                    # triple product in likelihood function
                    self.rGGNGGr.append(np.dot(p.detresiduals, Nir) - np.dot(GcNir, GcNiGcr))

                    # compute F^TG(G^TNG)^{-1}G^TF
                    self.UGGNGGU.append(np.dot(NiU.T, p.Umat) - np.dot(GcNiU.T, GcNiGcU))
                    
                    
                # first component of likelihood function
                loglike += -0.5 * (self.logdet_N[ct] + self.rGGNGGr[ct])

                # keep track of jitter terms needed later
                if self.npsr > 1 and incCorrelations:
                    if ct == 0:
                        Jinv = 1/p.Qamp
                    else:
                        Jinv = np.append(Jinv, 1/p.Qamp)

                    FJ.append(p.UtF.T * 1/p.Qamp)
                    FJF.append(np.dot(FJ[ct], p.UtF))

        else:
            for ct, p in enumerate(self.psr):
                loglike += -0.5 * (self.logdet_N[ct] + self.rGGNGGr[ct])


        # if only using one pulsar
        if self.npsr == 1 or not(incCorrelations):
            logdetPhi = 0
            tmp = []
            for ct, p in enumerate(self.psr):
                Phi0 = np.diag(10**p.kappa_tot+self.gwamp)
                UPhiU = np.dot(p.UtF, np.dot(Phi0, p.UtF.T))
                Phi = UPhiU + np.diag(p.Qamp) 

            
                try:
                    cf = sl.cho_factor(Phi)
                    logdetPhi += 2*np.sum(np.log(np.diag(cf[0])))
                    tmp.append(sl.cho_solve(cf, np.identity(Phi.shape[0])))
                except np.linalg.LinAlgError:
                    #print 'Cholesky failed when inverting phi'
                    #U, s, Vh = sl.svd(Phi)
                    #if not np.all(s > 0):
                    return -np.inf
                        #raise ValueError("ERROR: Phi singular according to SVD")
                    #logdetPhi = np.sum(np.log(s))
                    #Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))
            
            # block diagonal matrix
            Phiinv = sl.block_diag(*tmp)

        else:
            
            Phi0 = self.Phiinv + sl.block_diag(*FJF)
            logdet_J = np.sum(np.log(1/Jinv))

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Phi0)
                logdet_Phi0 = 2*np.sum(np.log(np.diag(cf[0])))
                PhiinvFJ = sl.cho_solve(cf, sl.block_diag(*FJ))
            except np.linalg.LinAlgError:
                print 'Cholesky Failed when inverting Phi0'
                return -np.inf
                #U, s, Vh = sl.svd(Phi0)
                #if not np.all(s > 0):
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
            logdet_Sigma = 2*np.sum(np.log(np.diag(cf[0])))
            expval2 = sl.cho_solve(cf, self.d)
        except np.linalg.LinAlgError:
            #print 'Cholesky Failed when inverting Sigma'
            return -np.inf
            #return -np.inf
            #U, s, Vh = sl.svd(Sigma)
            #if not np.all(s > 0):
                #return -np.inf
            #    raise ValueError("ERROR: Sigma singular according to SVD")
            #logdet_Sigma = np.sum(np.log(s))
            #expval2 = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, d)))


        loglike += -0.5 * (logdetPhi + logdet_Sigma) + 0.5 * (np.dot(self.d, expval2)) 

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

        #print 'Setting noise = {0} s'.format(time.time()-tstart)

        tstart = time.time()

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # construct covariance matrix
        #red = np.dot(self.psr[0].Ftot, (np.diag(self.Phi)*self.psr[0].Ftot).T)
        #cov = red + np.diag(self.psr[0].Nvec)

        QCQ = np.dot(self.psr[0].QRF, (Phi * self.psr[0].QRF).T)
        QCQ += np.dot(self.psr[0].QR, (self.psr[0].Nvec * self.psr[0].QR).T)

        #print 'Constructing Covariance = {0} s'.format(time.time()-tstart)

        tstart = time.time()

        # svd
        u, s, v = sl.svd(QCQ)
        ind = s/s[0] < 1e-15*len(s)
        sinv = 1/s
        sinv[ind] = 0.0

        logdetCov = np.sum(np.log(s[~ind]))
        invCov = np.dot(v.T, np.dot(np.diag(sinv), u.T))
        
        #print 'Computing Inverse = {0} s'.format(time.time()-tstart)
        
        tstart = time.time()

        loglike = -0.5 * (logdetCov + np.dot(self.psr[0].QRr, np.dot(invCov, self.psr[0].QRr)))

        #print 'Matrix vector = {0} s'.format(time.time()-tstart)
        
        #print 'Total time = {0} s\n'.format(time.time()-tstart_tot)

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
                d = np.dot(p.Ftot.T, p.detresiduals/p.Nvec)
            else:
                d = np.append(d, np.dot(p.Ftot.T, p.detresiduals/p.Nvec))

            # log determinant of N
            logdet_N = np.sum(np.log(p.Nvec))

            # triple product
            rNr = np.dot(p.detresiduals, p.detresiduals/p.Nvec)

            # compute F^TN^{-1}F
            right = ((1/p.Nvec) * p.Ftot.T).T
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
            logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            logdet_Sigma = np.sum(np.log(s))
            expval2 = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, d)))


        loglike += -0.5 * (self.logdetPhi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) 

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
            raise ValueError('ERROR: Mark 5 likelihood only valid with 1 pulsar')
        p = self.psr[0]

        # get gaussian part
        loglike += -0.5 * np.dot(p.detresiduals, p.detresiduals/p.Nvec)

        # non-gaussian part
        hermargs = p.detresiduals/np.sqrt(2*p.Nvec)

        hermcoeff = []
        if np.abs(np.sum(p.alphacoeff[1:])) > 1:
            return -np.inf
        for ii in range(p.nalpha):
            hermcoeff.append(p.alphacoeff[ii] / np.sqrt(2**ii*ss.gamma(ii+1)*np.sqrt(2*np.pi*p.Nvec)))
        
        # evaluate hermite polynomial sums
        hval = hermval(hermargs, np.array(hermcoeff))[0]
        #tmp = np.sum(hval, axis=0)
        loglike += 2*np.sum(np.log(np.abs(hval)))

        return loglike

    """
    mark 6 log likelihood. Note that this is not the same as mark6 in piccard

    EFAC + EQUAD + Red noise + DMV + GWs

    No jitter or frequency lines

    Uses Woodbury lemma and "T" matrix formalism

    """

    def mark6LogLikelihood(self, parameters, incCorrelations=True, incJitter=False):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters, incCorrelations=incCorrelations, \
                                incTM=True, incJitter=incJitter)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        
        # compute the white noise terms in the log likelihood
        TNT = []
        nfref = 0
        for ct, p in enumerate(self.psr):

            # check for nans or infs
            if np.any(np.isnan(p.detresiduals)) or np.any(np.isinf(p.detresiduals)):
                return -np.inf
                        
                
            # equivalent to T^T N^{-1} \delta t
            if ct == 0:
                d = np.dot(p.Tmat.T, p.detresiduals/p.Nvec)
            else:
                d = np.append(d, np.dot(p.Tmat.T, p.detresiduals/p.Nvec))

            # compute T^T N^{-1} T
            right = ((1/p.Nvec) * p.Tmat.T).T
            TNT.append(np.dot(p.Tmat.T, right))

            # log determinant of G^TNG
            logdet_N = np.sum(np.log(p.Nvec))

            # triple product in likelihood function
            rNr = np.sum(p.detresiduals**2/p.Nvec)
            
            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rNr)
            
            # calculate red noise piece
            if not incCorrelations:

                # compute sigma
                logdet_Sigma = 0
                nf = p.Tmat.shape[1]
                Sigma = TNT[ct] + self.Phiinv[nfref:(nfref+nf), nfref:(nfref+nf)]
                dd = d[nfref:(nfref+nf)]

                # cholesky decomp for maximum likelihood fourier components
                try:
                    cf = sl.cho_factor(Sigma)
                    expval2 = sl.cho_solve(cf, dd)
                    logdet_Sigma += np.sum(2*np.log(np.diag(cf[0])))
                except np.linalg.LinAlgError:
                    return -np.inf
                    raise ValueError("ERROR: Sigma singular according to SVD")

                loglike += -0.5 * logdet_Sigma + 0.5 * (np.dot(dd, expval2))

                # increment frequency counter
                nfref += nf
        
        if not incCorrelations:
            loglike += -0.5 * self.logdetPhi

        # compute the red noise, DMV and GWB terms in the log likelihood
        if incCorrelations: 
            # compute sigma
            Sigma = sl.block_diag(*TNT) + self.Phiinv

            # cholesky decomp for second term in exponential
            try:
                cf = sl.cho_factor(Sigma)
                expval2 = sl.cho_solve(cf, d)
                logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                logdet_Sigma = np.sum(np.log(s))
                expval2 = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, d)))


            loglike += -0.5 * (self.logdetPhi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) 

        return loglike



    
    """
    Zero log likelihood for prior testing purposes
    """
    def zeroLogLikelihood(self, parameters):

        return 0

        
    """
    Very simple uniform prior on all parameters

    """

    def mark1LogPrior(self, parameters):

        prior = 0
        if np.all(parameters >= self.pmin) and np.all(parameters <= self.pmax):
            prior += -np.sum(np.log(self.pmax-self.pmin))

        else:
            prior += -np.inf

        return prior

    """
    Very simple uniform prior on all parameters except flag in GW amplitude

    """

    def mark2LogPrior(self, parameters):

        prior = 0
        if np.all(parameters >= self.pmin) and np.all(parameters <= self.pmax):
            prior += -np.sum(np.log(self.pmax-self.pmin))

        else:
            prior += -np.inf

        #TODO:find better way of finding the amplitude
        for ss, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']
            
            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind+npars]

            if sig['corr'] == 'gr':
                prior += np.log(10**sparameters[0])


        return prior
    
    """
    Very simple uniform prior on all amplitudes, can also include flat in 
    Amplitudes of red noise
    """

    def mark3LogPrior(self, parameters):

        prior = 0
        if np.all(parameters >= self.pmin) and np.all(parameters <= self.pmax):
            prior += -np.sum(np.log(self.pmax-self.pmin))

        else:
            prior += -np.inf

        #TODO:find better way of finding the amplitude
        for ss, sig in enumerate(self.ptasignals):

            # short hand
            psrind = sig['pulsarind']
            parind = sig['parindex']
            npars = sig['npars']
            
            # parameters for this signal
            sparameters = sig['pstart'].copy()

            # which ones are varying
            sparameters[sig['bvary']] = parameters[parind:parind+npars]

            if sig['corr'] == 'gr' and sig['stype'] == 'powerlaw':
                if sig['prior'][0] == 'uniform':
                    prior += np.log(10**sparameters[0])
            
            if sig['corr'] == 'gr' and sig['stype'] == 'spectrum':
                if np.any(np.array(sig['prior']) == 'uniform'):
                    idx = np.array(sig['prior']) == 'uniform'
                    prior += np.sum(np.log(10**sparameters[idx]))
            
            if sig['stype'] == 'spectrum' and sig['corr'] == 'single':
                if np.any(np.array(sig['prior']) == 'uniform'):
                    idx = np.array(sig['prior']) == 'uniform'
                    prior += np.sum(np.log(10**sparameters[idx]))
                
                # cheater prior
                sig_data = self.psr[psrind].sig_data * 10
                sig_red = np.sqrt(np.sum(10**sparameters))
                #print sig_red, sig_data
                if sig_red > sig_data:
                    prior += -np.inf
            
            if sig['corr'] == 'gr' and sig['stype'] == 'spectrum':
                if np.any(np.array(sig['prior']) == 'sqrt'):
                    idx = np.array(sig['prior']) == 'sqrt'
                    prior += np.sum(np.log(10**(sparameters[idx]/2)))
            
            if sig['stype'] == 'spectrum' and sig['corr'] == 'single':
                if np.any(np.array(sig['prior']) == 'sqrt'):
                    idx = np.array(sig['prior']) == 'sqrt'
                    prior += np.sum(np.log(10**(sparameters[idx]/2)))
                
                # cheater prior
                sig_data = self.psr[psrind].sig_data * 10
                sig_red = np.sqrt(np.sum(10**sparameters))
                #print sig_red, sig_data
                if sig_red > sig_data:
                    prior += -np.inf

            
            if sig['stype'] in ['powerlaw'] and sig['corr'] == 'single':
                if sig['bvary'][0]:
                    if sig['prior'][0] == 'uniform':
                        prior += np.log(10**sparameters[0])
                
                # cheater prior
                Amp = 10**sparameters[0]
                gam = sparameters[1]
                sig_data = self.psr[psrind].sig_data * 10
                if gam > 1:
                    sig_red = 2.05e-9 / np.sqrt(gam-1)*(Amp/1e-15)*\
                        (self.psr[psrind].Tmax/3.16e7)**((gam-1)/2) 
                else:
                    sig_red = 0
                if sig_red > sig_data:
                    prior += -np.inf

            # prior on ECC, EDOT and T0
            if sig['stype'] == 'lineartimingmodel' or sig['stype'] == 'nonlineartimingmodel':
                
                ecc, edot, t0, sini, a1, m2, pb = 0, 0, 0, 0, 0, 0, 0
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

                        pindex += 1

                # check if including all three
                if ecc and edot and t0:
                    tt0 = (self.psr[psrind].toas - t0*86400)
                    check = ecc + edot*tt0

                    if np.any(check > 1) or np.any(check<0):
                        prior += -np.inf

                # flat in cosi prior
                if sini:
                    prior += np.log(sini/np.sqrt(1-sini**2))

                # prior on pulsar mass [0,3]
                #if sini and pb and m2 and a1:
                #    Pb = pb*86400
                #    X = a1*299.79e6/3e8
                #    M2 = m2*4.9e-6
                #    mp = ((sini*(Pb/2/np.pi)**(2./3)*M2/X)**(3./2) - M2)/4.9e-6
                #    if mp <= 0 or mp >= 3:
                #        prior += -np.inf


            # CW parameters
            if sig['stype'] == 'cw':
                pindex = 0
                for jj in range(sig['ntotpars']):
                    if sig['bvary'][jj]:

                        if sig['prior'][jj] == 'cos':
                            prior += np.log(np.abs(np.sin(sparameters[pindex])))

                        pindex += 1

            # pulsar distance prior
            if sig['stype'] == 'pulsardistance':
                pdist = sparameters
                m = self.psr[psrind].pdist
                s = self.psr[psrind].pdistErr
                prior += -0.5 * (np.log(2*np.pi*s**2) + (m-pdist)**2/s**2)
                
        return prior


    #################################################################################
    
    # MCMC jump proposals
    
    # TODO: make one single proposal that can take stype as argument, will have to change MCMC code...

    # red noise draws
    def drawFromRedNoisePrior(self, parameters, iter, beta):
        
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='powerlaw', \
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='powerlaw', \
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
            if sig['bvary'][0]:

                # log prior
                if sig['prior'][0] == 'log':
                    q[parind] = np.random.uniform(self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'][0] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10**self.pmin[parind], \
                                                           10**self.pmax[parind]))
                    qxy += np.log(10**parameters[parind]/10**q[parind])
                    
                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]
        
            # jump in spectral index if varying
            if sig['bvary'][1]:

                if sig['prior'][1] == 'uniform':
                    q[parind+1] = np.random.uniform(self.pmin[parind+1], self.pmax[parind+1])
                    qxy += 0

                else:
                    q[parind+1] = parameters[parind+1]

        
        return q, qxy
    
    # red noise sepctrum draws
    def drawFromRedNoiseSpectrumPrior(self, parameters, iter, beta):
        
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='spectrum', \
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='spectrum', \
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
                        q[parind+jj] = np.random.uniform(self.pmin[parind+jj], \
                                                         self.pmax[parind+jj])
                        qxy += 0

                    elif sig['prior'][jj] == 'uniform':
                        q[parind+jj] = np.log10(np.random.uniform(10**self.pmin[parind+jj], \
                                                               10**self.pmax[parind+jj]))
                        qxy += np.log(10**parameters[parind+jj]/10**q[parind+jj])
                    
                    elif sig['prior'][jj] == 'sqrt':
                        q[parind+jj] = np.log10(np.random.uniform(10**(self.pmin[parind+jj]/2), \
                                            10**(self.pmax[parind+jj]/2))**2)
                        qxy += np.log(10**(parameters[parind+jj]/2)/10**(q[parind+jj]/2))
                        
                    else:
                        print 'Prior type not recognized for parameter'
                        q[parind+jj] = parameters[parind+jj]

        return q, qxy

        # red noise sepctrum draws
    def drawFromGWBSpectrumPrior(self, parameters, iter, beta):
        
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = 1
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='spectrum', \
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
                        q[parind+jj] = np.random.uniform(self.pmin[parind+jj], \
                                                         self.pmax[parind+jj])
                        qxy += 0

                    elif sig['prior'][jj] == 'uniform':
                        q[parind+jj] = np.log10(np.random.uniform(10**self.pmin[parind+jj], \
                                                               10**self.pmax[parind+jj]))
                        qxy += np.log(10**parameters[parind+jj]/10**q[parind+jj])
                    
                    elif sig['prior'][jj] == 'sqrt':
                        q[parind+jj] = np.log10(np.random.uniform(10**(self.pmin[parind+jj]/2), \
                                            10**(self.pmax[parind+jj]/2))**2)
                        qxy += np.log(10**(parameters[parind+jj]/2)/10**(q[parind+jj]/2))
                        
                    else:
                        print 'Prior type not recognized for parameter'
                        q[parind+jj] = parameters[parind+jj]

        return q, qxy



    # GWB draws draws
    def drawFromGWBPrior(self, parameters, iter, beta):
        
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = 1
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='powerlaw', corr='gr')

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
                    q[parind] = np.random.uniform(self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'][0] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10**self.pmin[parind], \
                                                           10**self.pmax[parind]))
                    qxy += np.log(10**parameters[parind]/10**q[parind])
                    
                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]
        
            # jump in spectral index if varying
            if sig['bvary'][1]:

                if sig['prior'][1] == 'uniform':
                    q[parind+1] = np.random.uniform(self.pmin[parind+1], self.pmax[parind+1])
                    qxy += 0

                else:
                    q[parind+1] = parameters[parind+1]

        
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
                            q[parind+jj] = np.arccos(np.random.uniform(-1, 1))
                            qxy += np.log(np.sin(parameters[parind+jj])/np.sin(q[parind+jj]))

                        else:
                            q[parind+jj] = np.random.uniform(self.pmin[parind+jj], \
                                                             self.pmax[parind+jj])

        
        return q, qxy

    # pulsar distance jump
    def pulsarDistanceJump(self, parameters, iter, beta):

        q = parameters.copy()
        qxy = 0

        # get pulsar distances
        L0 = []
        signum = self.getSignalNumbersFromDict(self.ptasignals, \
            stype='pulsardistance', corr='single')

        # check to make sure we have all distances
        if len(signum) != self.npsr:
            raise ValueError('ERROR: Number of pulsar distances != number of pulsars!')


        # get distances
        pdist_inds, pdisterr = [], []
        for signum0 in signum:
            sig0 = self.ptasignals[signum0]
            pdist_inds.append(sig0['parindex'])
            L0.append(q[sig0['parindex']])
            pdisterr.append(self.psr[sig0['pulsarind']].pdistErr)

        # make indices array
        pdist_inds = np.array(pdist_inds)
        L0 = np.array(L0) 
        pdisterr = np.array(pdisterr)

        L_new = np.random.multivariate_normal(L0, np.diag(pdisterr**2))
        q[pdist_inds] = L_new
        
        return q, qxy

    # pulsar mode jump
    def pulsarPhaseFix(self, prepar, postpar, iter, beta):

        pre = prepar.copy()
        post = postpar.copy()

        # get pulsar distances
        L0, L1 = [], []
        signum = self.getSignalNumbersFromDict(self.ptasignals, \
            stype='pulsardistance', corr='single')

        # check to make sure we have all distances
        if len(signum) != self.npsr:
            raise ValueError('ERROR: Number of pulsar distances != number of pulsars!')


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
            
            if sig['stype'] == 'cw':

                par0 = pre[sig['parindex']:(sig['parindex']+sig['npars'])]
                par1 = post[sig['parindex']:(sig['parindex']+sig['npars'])]

                theta0, phi0, mc0, omega0 = par0[0], par0[1], 10**par0[2], 10**par0[4]*np.pi
                theta1, phi1, mc1, omega1 = par1[0], par1[1], 10**par1[2], 10**par1[4]*np.pi
                
        # get angular separations
        cosMu0 = np.array([p.cosMu(theta0, phi0) for p in self.psr]).flatten()
        cosMu1 = np.array([p.cosMu(theta1, phi1) for p in self.psr]).flatten()

        # get in correct units
        L0 *= 1.0267e11
        L1 *= 1.0267e11
        mc0 *= 4.9e-6
        mc1 *= 4.9e-6

        # pulsar frequency
        omegap0 = omega0 * (1+256/5*mc0**(5/3)*omega0**(8/3)*L0*(1-cosMu0))**(-3/8)
        omegap1 = omega1 * (1+256/5*mc1**(5/3)*omega1**(8/3)*L1*(1-cosMu1))**(-3/8)

        # pulsar phase
        phase0 = 1/32/mc0**(5/3) * (omega0**(-5/3) - omegap0**(-5/3))
        phase1 = 1/32/mc1**(5/3) * (omega1**(-5/3) - omegap1**(-5/3))

        # size of phase jump
        prob = np.random.rand()
        if prob > 0.8:
            sigma = 0.1 * np.random.randn(self.npsr)

        elif prob > 0.6:
            sigma = 0.02 * np.random.randn(self.npsr)

        elif prob > 0.4:
            sigma = 0.002 * np.random.randn(self.npsr)

        else:
            sigma = 0.005 * np.random.randn(self.npsr)

        deltaL = (np.mod(phase1, 2*np.pi) - np.mod(phase0, 2*np.pi)
                     + sigma * np.sqrt(1/beta))/(omegap1*(1-cosMu1))

        L_new  = (L1 + deltaL) / 1.0267e11
        post[pdist_inds] = L_new
        #L1 = L_new * 1.0267e11
        #omegap1 = omega1 * (1+256/5*mc1**(5/3)*omega1**(8/3)*L1*(1-cosMu1))**(-3/8)

        ## pulsar phase
        #phase1 = 1/32/mc1**(5/3) * (omega1**(-5/3) - omegap1**(-5/3))

        #print np.mod(phase0, 2*np.pi), np.mod(phase1, 2*np.pi)


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

                par0 = parameters[sig['parindex']:(sig['parindex']+sig['npars'])]
                mc0, dist0  = 10**par0[2]*4.9e-6, 10**par0[3]*1.0267e14

                # draw distance uniformly from prior
                dist1 = 10**np.random.uniform(sig['pmin'][3], sig['pmax'][3])
                dist1 *= 1.0267e14

                # find chirp mass that keeps M^{5/3}/dL constant
                mc1 = (dist1 * mc0**(5/3)/dist0)**(3/5)

                # put values in return array
                q[sig['parindex']+2] = np.log10(mc1/4.9e-6)
                q[sig['parindex']+3] = np.log10(dist1/1.0267e14)

        return q, qxy



    # draws from equad prior
    def drawFromEquadPrior(self, parameters, iter, beta):
        
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='equad', \
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='equad', \
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
                    q[parind] = np.random.uniform(self.pmin[parind], self.pmax[parind])
                    qxy += 0

                elif sig['prior'] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10**self.pmin[parind], \
                                                           10**self.pmax[parind]))
                    qxy += np.log(10**parameters[parind]/10**q[parind])
                    
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
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, \
                                    stype='jitter_epoch', corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='jitter_epoch', \
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
                        q[parind+jj] = np.random.uniform(self.pmin[parind+jj], \
                                                         self.pmax[parind+jj])
                        qxy += 0

                    elif sig['prior'][jj] == 'uniform':
                        q[parind+jj] = np.log10(np.random.uniform(10**self.pmin[parind+jj], \
                                                               10**self.pmax[parind+jj]))
                        qxy += np.log(10**parameters[parind+jj]/10**q[parind+jj])
                        
                    else:
                        print 'Prior type not recognized for parameter'
                        q[parind+jj] = parameters[parind+jj]
        
        return q, qxy
    
    # draws from jitter equad prior
    def drawFromJitterEquadPrior(self, parameters, iter, beta):
        
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, \
                                    stype='jitter_equad', corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='jitter_equad', \
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
                    q[parind] = np.random.uniform(self.pmin[parind], \
                                                     self.pmax[parind])
                    qxy += 0

                elif sig['prior'][jj] == 'uniform':
                    q[parind] = np.log10(np.random.uniform(10**self.pmin[parind], \
                                                           10**self.pmax[parind]))
                    qxy += np.log(10**parameters[parind]/10**q[parind])
                    
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
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, stype='efac', \
                                                       corr='single'))
        signum = self.getSignalNumbersFromDict(self.ptasignals, stype='efac', \
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
                    q[parind] = np.random.uniform(self.pmin[parind], self.pmax[parind])
                    qxy += 0
                    
                else:
                    print 'Prior type not recognized for parameter'
                    q[parind] = parameters[parind]
        
        return q, qxy

    # draws from timing model fisher matrix
    def drawFromTMfisherMatrix(self, parameters, iter, beta):
        
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # find number of signals
        nsigs = np.sum(self.getNumberOfSignalsFromDict(self.ptasignals, \
                    stype='nonlineartimingmodel', corr='single'))
        if np.any(nsigs):
            signum = self.getSignalNumbersFromDict(self.ptasignals, 
                    stype='nonlineartimingmodel', corr='single')
        else:
            signum = self.getSignalNumbersFromDict(self.ptasignals, 
                    stype='lineartimingmodel', corr='single')
        
        # current timing model parameters
        sig = self.ptasignals[signum]
        x = parameters[sig['parindex']:(sig['parindex']+sig['ntotpars'])]
        
        # get parmeters in new diagonalized basis
        y = np.dot(self.psr[0].fisherU.T, x)
        
        # get scale of jump
        alpha = np.random.rand()
        scale = 1
        if alpha > 0.9:
            scale = 10

        # make correlated componentwise adaptive jump
        ind = np.unique(np.random.randint(0, len(x), len(x)))
        neff = len(ind)
        sd = 2.4  / np.sqrt(2*neff) * scale

        y[ind] = y[ind] + np.random.randn(neff) * sd * np.sqrt(1/self.psr[0].fisherS[ind])
        q[sig['parindex']:(sig['parindex']+sig['ntotpars'])] = \
                np.dot(self.psr[0].fisherU, y)

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
                #if ct == 0:
                d = np.dot(p.AGF.T, p.AGr/p.Nwvec)
                #else:
                #    d = np.append(d, np.dot(p.AGF.T, p.AGr/p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1/p.Nwvec) * p.AGF.T).T
                FGGNGGF.append(np.dot(p.AGF.T, right))

            
            else:   

                # G(G^TNG)^{-1}G^T = N^{-1} - N^{-1}G_c(G_c^TN^{-1}G_c)^{-1}N^{-1}
                Nir = p.detresiduals / p.Nvec
                NiGc = ((1.0/p.Nvec) * p.Hcmat.T).T
                GcNiGc = np.dot(p.Hcmat.T, NiGc)
                NiF = ((1.0/p.Nvec) * p.Ftot.T).T
                GcNir = np.dot(NiGc.T, p.detresiduals)
                GcNiF = np.dot(NiGc.T, p.Ftot)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                    NiGcNiGcr = np.dot(NiGc, GcNiGcr)

                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"
                
                #F^TG(G^TNG)^{-1}G^T\delta t 
                #if ct == 0:
                d = np.dot(p.Ftot.T, Nir - NiGcNiGcr)
                #else:
                #    d = np.append(d, np.dot(p.Ftot.T, Nir - NiGcNiGcr))

                # compute F^TG(G^TNG)^{-1}G^TF
                FGGNGGF.append(np.dot(NiF.T, p.Ftot) - np.dot(GcNiF.T, GcNiGcF))

                
            # compute the red noise, DMV and GWB terms in the log likelihood
            
            # compute sigma
            nf = self.npftot[ct]
            print nf
            Sigma = FGGNGGF[ct] + self.Phiinv[nfref:(nfref+nf), nfref:(nfref+nf)]
            Phi = np.diag(self.Phi[nfref:(nfref+nf)])

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
                tmpardel = p.getNewTimingModelParameterList(keep=False, tmpars=pttmpars)
                Mmat, newptmpars, newptmdescription = self.delFromDesignMatrix(tmpardel) 
            else:
                Mmat = p.Mmat


            # compute F^TN^{-1}F
            right = ((1/p.Nvec) * p.Ftot.T).T
            FNF = np.dot(p.Ftot.T, right)

            if incJitter:
                # compute F^TN^{-1}F
                right = ((1/p.Nvec) * p.Umat.T).T
                UNU = np.dot(p.Umat.T, right)


            # compute sigma
            nf = self.npftot[ct]
            Sigma = FNF + self.Phiinv[nfref:(nfref+nf), nfref:(nfref+nf)]
            
            if incJitter:
                Phi0 = np.diag(self.Phi[nfref:(nfref+nf)])
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
                tmp = np.dot(SigmaInv, np.dot(p.Umat.T, (1/p.Nvec * Mmat.T).T))
                tmp2 = np.dot(SigmaInv, np.dot(p.Umat.T, p.detresiduals/p.Nvec))
                g = np.dot(Mmat.T, p.detresiduals/p.Nvec) - \
                        np.dot(Mmat.T, np.dot((1/p.Nvec * p.Umat.T).T, tmp2))
                Gamma = np.dot(Mmat.T, (1/p.Nvec*Mmat.T).T) - \
                        np.dot(Mmat.T, np.dot((1/p.Nvec * p.Umat.T).T, tmp))
                GammaInv = np.linalg.inv(Gamma)
                cov.append(GammaInv)
                eps.append(np.dot(GammaInv, g))

            else:
                tmp = np.dot(SigmaInv, np.dot(p.Ftot.T, (1/p.Nvec * Mmat.T).T))
                tmp2 = np.dot(SigmaInv, np.dot(p.Ftot.T, p.detresiduals/p.Nvec))
                g = np.dot(Mmat.T, p.detresiduals/p.Nvec) - \
                        np.dot(Mmat.T, np.dot((1/p.Nvec * p.Ftot.T).T, tmp2))
                Gamma = np.dot(Mmat.T, (1/p.Nvec*Mmat.T).T) - \
                        np.dot(Mmat.T, np.dot((1/p.Nvec * p.Ftot.T).T, tmp))
                GammaInv = np.linalg.inv(Gamma)
                cov.append(GammaInv)
                eps.append(np.dot(GammaInv, g))
            
            p.detresiduals -= np.dot(Mmat, eps[-1])
            if incJitter:
                d = np.dot(p.Umat.T, p.detresiduals/p.Nvec)
            else:
                d = np.dot(p.Ftot.T, p.detresiduals/p.Nvec)

            expval2 = sl.cho_solve(cf, d)

            # triple product
            rNr = np.dot(p.detresiduals, p.detresiduals/p.Nvec)

            chisq.append(rNr - np.dot(d, expval2))

            # increment frequency counter
            nfref += nf

        return eps, cov, chisq
            

        


    

