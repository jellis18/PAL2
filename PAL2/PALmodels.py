#!/usr/bin/env python

from __future__ import division

import numpy as np
import h5py as h5
import os, sys, time
import json
import tempfile
import scipy.linalg as sl

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
            incRedNoise=False, noiseModel='powerlaw', fc=None, \
            incDM=False, dmModel='powerlaw', \
            incGWB=False, gwbModel='powerlaw', \
            incBWM=False, \
            incCW=False, \
            varyEfac=True, separateEfacs=False, separateEfacsByFreq=False, \
            incEquad=False,separateEquads=False, separateEquadsByFreq=False, \
            incCEquad=False, \
            incJitter=False, separateJitter=False, separateJitterByFreq=False, \
            incSingleFreqNoise=False, numSingleFreqLines=1, \
            incSingleFreqDMNoise=False, numSingleFreqDMLines=1, \
            singlePulsarMultipleFreqNoise=None, \
            multiplePulsarMultipleFreqNoise=None, \
            dmFrequencyLines=None, \
            orderFrequencyLines=False, \
            compression = 'None', \
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
                        "pstart":[1.0]
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
                    "pstart":[1.0]
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
                            "pmin":[-10.0],
                            "pmax":[-4.0],
                            "pwidth":[0.1],
                            "pstart":[-8.0]
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
                        "pmin":[-10.0],
                        "pmax":[-4.0],
                        "pwidth":[0.1],
                        "pstart":[-8.0]
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
                            "flagname":"jitter",
                            "flagvalue":flagval,
                            "bvary":[True],
                            "pmin":[-10.0],
                            "pmax":[-4.0],
                            "pwidth":[0.1],
                            "pstart":[-8.0]
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
                        "pstart":[-8.0]
                        })
                    signals.append(newsignal)


            if incRedNoise:
                if noiseModel=='spectrum':
                    #nfreqs = numNoiseFreqs[ii]
                    bvary = [True]*nfreqs
                    pmin = [-18.0]*nfreqs
                    pmax = [-7.0]*nfreqs
                    pstart = [-10.0]*nfreqs
                    pwidth = [0.1]*nfreqs
                elif noiseModel=='powerlaw':
                    bvary = [True, True, False]
                    pmin = [-20.0, 0.02, 1.0e-11]
                    pmax = [-10.0, 6.98, 3.0e-9]
                    pstart = [-14.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                elif noiseModel=='spectralModel':
                    bvary = [True, True, True]
                    pmin = [-28.0, 0.0, -4.0]
                    pmax = [-14.0, 12.0, 2.0]
                    pstart = [-22.0, 2.0, -1.0]
                    pwidth = [-0.2, 0.1, 0.1]

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
                    "pstart":pstart
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
                    DMModel = 'dmspectrum'
                elif dmModel=='powerlaw':
                    bvary = [True, True, False]
                    pmin = [-14.0, 0.02, 1.0e-11]
                    pmax = [-6.5, 6.98, 3.0e-9]
                    pstart = [-13.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
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
                    "pstart":pstart
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
                        "pstart":[-7.0, -10.0]
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
                        "pmin":[-9.0, -18.0],
                        "pmax":[-5.0, -9.0],
                        "pwidth":[-0.1, -0.1],
                        "pstart":[-7.0, -10.0]
                        })
                    signals.append(newsignal)

        if incCW:
            #TODO implement this
            pass
 
        if incGWB:
            if gwbModel=='spectrum':
                bvary = [True]*nfreqs
                pmin = [-18.0]*nfreqs
                pmax = [-7.0]*nfreqs
                pstart = [-10.0]*nfreqs
                pwidth = [0.1]*nfreqs
            elif gwbModel=='powerlaw':
                bvary = [True, True, False]
                pmin = [-17.0, 1.02, 1.0e-11]
                pmax = [-10.0, 6.98, 3.0e-9]
                pstart = [-15.0, 2.01, 1.0e-10]
                pwidth = [0.1, 0.1, 5.0e-11]

            newsignal = OrderedDict({
                "stype":gwbModel,
                "corr":"gr",
                "pulsarind":-1,
                "bvary":bvary,
                "pmin":pmin,
                "pmax":pmax,
                "pwidth":pwidth,
                "pstart":pstart
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

        elif signal['stype'] in ['powerlaw', 'spectrum', 'spectralModel']:
            # Any time-correlated signal
            self.addSignalTimeCorrelated(signal)
            self.haveStochSources = True

        elif signal['stype'] in ['dmpowerlaw', 'dmspectrum']:
            # A DM variation signal
            self.addSignalDMV(signal)
            self.haveStochSources = True
        
        #TODO: not implemented correctly yet
        elif signal['stype'] == 'frequencyline':
            # Single free-floating frequency line
            psrSingleFreqs = self.getNumberOfSignals(stype='frequencyline', \
                    corr='single')
            signal['npsrfreqindex'] = psrSingleFreqs[signal['pulsarind']]
            self.addSignalFrequencyLine(signal)
            self.haveStochSources = True

        elif signal['stype'] == 'dmfrequencyline':
            # Single free-floating frequency line
            psrSingleFreqs = self.getNumberOfSignals(stype='dmfrequencyline', \
                    corr='single')
            signal['npsrfreqindex'] = psrSingleFreqs[signal['pulsarind']]
            self.addSignalFrequencyLine(signal)
            self.haveStochSources = True

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
    def initModel(self, fullmodel, fromFile=False, verbose=False):
        numNoiseFreqs = fullmodel['numNoiseFreqs']
        numDMFreqs = fullmodel['numDMFreqs']
        compression = fullmodel['compression']
        evalCompressionComplement = fullmodel['evalCompressionComplement']
        orderFrequencyLines = fullmodel['orderFrequencyLines']
        likfunc = fullmodel['likfunc']
        signals = fullmodel['signals']

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
        separateEfacs = numEfacs > 1
        
        # Find out how many jitter signals there are, and translate that to a
        # separateEfacs boolean array 
        numJitter = self.getNumberOfSignalsFromDict(signals, \
                stype='jitter', corr='single')
        separateJitter = numJitter > 1

        # Modify design matrices, and create pulsar Auxiliary quantities
        for pindex, p in enumerate(self.psr):
            # If we model DM variations, we will need to include QSD
            # marginalisation for DM. Modify design matrix accordingly
            #if dmModel[pindex] != 'None':
            #if numDMFreqs[pindex] > 0:
            #    p.addDMQuadratic()


            # We'll try to read the necessary quantities from the HDF5 file
            try:
                if not fromFile:
                    raise StandardError('Requested to re-create the Auxiliaries')
                # Read Auxiliaries
                if verbose:
                    print "Reading Auxiliaries for {0}".format(p.name)
                p.readPulsarAuxiliaries(self.h5df, Tmax, \
                        numNoiseFreqs[pindex], \
                        numDMFreqs[pindex], ~separateEfacs[pindex], \
                        nSingleFreqs=numSingleFreqs[pindex], \
                        nSingleDMFreqs=numSingleDMFreqs[pindex], \
                        likfunc=likfunc, compression=compression, \
                        memsave=True)
            except (StandardError, ValueError, KeyError, IOError, RuntimeError) as err:
                # Create the Auxiliaries ourselves

                # For every pulsar, construct the auxiliary quantities like the Fourier
                # design matrix etc
                if verbose:
                    print str(err)
                    print "Creating Auxiliaries for {0}".format(p.name)
                p.createPulsarAuxiliaries(self.h5df, Tmax, numNoiseFreqs[pindex], \
                        numDMFreqs[pindex], ~separateEfacs[pindex], \
                                nSingleFreqs=numSingleFreqs[pindex], \
                                nSingleDMFreqs=numSingleDMFreqs[pindex], \
                                likfunc=likfunc, compression=compression, \
                                write='no')

        # Initialise the ptasignal objects
        self.ptasignals = []
        index = 0
        for ii, signal in enumerate(signals):
            self.addSignal(signal, index, Tmax)
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

                elif sig['stype'] == 'spectrum':
                    flagname = 'frequency'
                    flagvalue = str(self.psr[psrindex].Ffreqs[2*jj])

                elif sig['stype'] == 'dmspectrum':
                    flagname = 'dmfrequency'
                    flagvalue = str(self.psr[psrindex].Fdmfreqs[2*jj])

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
    def initParameters(self, startEfacAtOne=True):
        
        p0 = []
        for ct, sig in enumerate(self.ptasignals):
            if np.any(sig['bvary']):
                for min, max in zip(sig['pmin'][sig['bvary']], sig['pmax'][sig['bvary']]):
                    if startEfacAtOne and sig['stype'] == 'efac':
                        p0.append(1)
                    else:
                        p0.append(min + np.random.rand()*(max - min))     
            
        return np.array(p0)
    

    """
    Determine intial covariance matrix for jumps


    """
    def initJumpCovariance(self):

        cov_diag = []
        for ct, sig in enumerate(self.ptasignals):
            if np.any(sig['bvary']):
                for step in sig['pwidth'][sig['bvary']]:
                    cov_diag.append((step/5)**2)
                    
        return np.diag(cov_diag)


    """
    Allocate memory for the ptaLikelihood attribute matrices that we'll need in
    the likelihood function.  This function does not perform any calculations,
    although it does initialise the 'counter' integer arrays like npf and npgs.
    """
    def allocateLikAuxiliaries(self):
        # First figure out how large we have to make the arrays
        self.npsr = len(self.psr)
        self.npf = np.zeros(self.npsr, dtype=np.int)
        self.npu = np.zeros(self.npsr, dtype=np.int)
        self.npff = np.zeros(self.npsr, dtype=np.int)
        self.npfdm = np.zeros(self.npsr, dtype=np.int)
        self.npffdm = np.zeros(self.npsr, dtype=np.int)
        self.npobs = np.zeros(self.npsr, dtype=np.int)
        self.npgs = np.zeros(self.npsr, dtype=np.int)
        self.npgos = np.zeros(self.npsr, dtype=np.int)
        
        for ii, p in enumerate(self.psr):

            # number of red and DM frequencies
            self.npf[ii] = len(p.Ffreqs)
            self.npfdm[ii] = len(p.Fdmfreqs)

            # noise vectors
            p.Nvec = np.zeros(len(p.toas))
            p.Nwvec = np.zeros(p.Hmat.shape[1])
            p.Nwovec = np.zeros(p.Homat.shape[1])

        # number of GW frequencies
        self.ngwf = np.max(self.npf)
        self.gwfreqs = self.psr[np.argmax(self.npf)].Ffreqs


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
    def setPsrNoise(self, parameters, incJitter=True):

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
                if self.psr[psrind].twoComponentNoise:
                    self.psr[psrind].Nwvec += self.psr[psrind].Wvec * pefac**2
                    self.psr[psrind].Nwovec += self.psr[psrind].Wovec * pefac**2

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
                if self.psr[psrind].twoComponentNoise:
                    self.psr[psrind].Nwvec += pequadsqr
                    self.psr[psrind].Nwovec += pequadsqr
                
                else:   # use Nvec stored in dictionary
                    self.psr[psrind].Nvec += sig['Nvec'] * pequadsqr

            # jitter signal
            elif sig['stype'] == 'jitter':

                # is this parameter being varied
                if sig['npars'] == 1:
                    pequadsqr = 10**(2*parameters[parind])

                # if not use reference value
                else:
                    pequadsqr = 10**(2*sig['pstart'][0])

                self.psr[psrind].Qamp += sig['Jvec'] * pequadsqr

                if incJitter:
                    # Need to include it just like the equad (for compressison)
                    self.psr[psrind].Nvec += sig['Nvec'] * pequadsqr

                    if self.psr[psrind].twoComponentNoise:
                        self.psr[psrind].Nwvec += pequadsqr
                        self.psr[psrind].Nwovec += pequadsqr



    """
    Construct complete Phi inverse matrix, including DM corrections if needed

    TODO: this code only works if all pulsars have the same number of frequencies
    want to make this more flexible
    """
    def constructPhiMatrix(self, parameters):
        
        tstart = time.time()
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
                

            # powerlaw spectrum
            if sig['stype'] == 'powerlaw':

                # pulsar independend red noise powerlaw
                if sig['corr'] == 'single':

                    # get Amplitude and spectral index
                    Amp = 10**sparameters[0]
                    gamma = sparameters[1]
                    
                    freqpy = self.psr[psrind].Ffreqs
                    f1yr = 1/3.16e7
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


        tstart = time.time()
        # now that we have obtained rho and kappa, we can construct Phiinv
        sigdiag = []
        sigoffdiag = []

        # no correlated signals (easy)
        if rho is None:

            # loop over all pulsars
            for ii, p in enumerate(self.psr):

                # have both red noise and DM variations
                if np.any(p.kappa) and np.any(p.kappadm):
                    kappa_tot = np.concatenate((p.kappa, p.kappadm))
                
                # red noise but no dm
                elif np.any(p.kappa) and ~np.any(p.kappadm):
                    kappa_tot = p.kappa
                
                # dm but no red noise
                elif ~np.any(p.kappa) and np.any(p.kappadm):
                    kappa_tot = p.kappadm
                
                # neither
                else:
                    kappa_tot = np.ones(p.kappa.shape) * -40

                # append to signal diagonal
                sigdiag.append(10**kappa_tot)

            # convert to array and flatten
            Phi = np.array(sigdiag).flatten()
            self.Phiinv = np.diag(1/Phi)
            self.logdetPhi = np.sum(np.log(Phi))


        # correlated signals (not as easy)
        if rho is not None:
        
            for ii, p in enumerate(self.psr):

                # have both red noise and DM variations
                if np.any(p.kappa) and np.any(p.kappadm):
                    kappa_tot = np.concatenate((p.kappa, p.kappadm))
                
                # red noise but no dm
                elif np.any(p.kappa) and ~np.any(p.kappadm):
                    kappa_tot = p.kappa
                
                # dm but no red noise
                elif ~np.any(p.kappa) and np.any(p.kappadm):
                    kappa_tot = p.kappadm

                # for now, assume that GW freqs is the same as 
                # number of freqs per pulsar

                # get number of DM freqs (not included in GW spectrum)
                ndmfreq = np.sum(p.kappadm != 0)

                # append to rho
                if ndmfreq > 0:
                    gwamp = np.concatenate((10**rho, np.zeros(ndmfreq)))
                else:
                    gwamp = 10**rho

                # append to diagonal elements
                if len(kappa_tot) > 0:
                    sigdiag.append(10**kappa_tot + gwamp)
                else:
                    sigdiag.append(gwamp)
                
                # append to off diagonal elements
                sigoffdiag.append(gwamp)


            # compute Phi inverse from Lindley's code
            nftot = self.ngwf + np.max(self.npfdm)
            smallMatrix = np.zeros((nftot, self.npsr, self.npsr))
            for ii in range(self.npsr):
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
            self.Phiinv = np.zeros((self.npsr*nftot, self.npsr*nftot))
            for ii in range(self.npsr):
                for jj in range(ii, self.npsr):
                    for kk in range(0,nftot):
                        self.Phiinv[kk+ii*nftot, kk+jj*nftot] = smallMatrix[kk,ii,jj]
            
            # symmeterize Phi
            self.Phiinv = self.Phiinv + self.Phiinv.T - np.diag(np.diag(self.Phiinv))



    """ 
    Update deterministic sources
    """

    def updateDetSources():

        # Set all the detresiduals equal to residuals
        for ct, p in enumerate(self.psr):
            p.detresiduals = p.residuals.copy()



    """
    mark 1 log likelihood. Note that this is not the same as mark1 in piccard

    EFAC + EQUAD + Red noise + DMV + GWs

    No jitter or frequency lines

    Uses Woodbury lemma

    """

    def mark1LogLikelihood(self, parameters):

        loglike = 0

        # set pulsar white noise parameters
        self.setPsrNoise(parameters, incJitter=False)

        # set red noise, DM and GW parameters
        self.constructPhiMatrix(parameters)

        # set deterministic sources
        if self.haveDetSources:
            self.updateDetSources(parameters)

        
        # compute the white noise terms in the log likelihood
        FGGNGGF = []
        for ct, p in enumerate(self.psr):

            if p.twoComponentNoise:
                
                # equivalent to F^TG(G^TNG)^{-1}G^T\delta t in two component basis
                if ct == 0:
                    d = np.dot(p.AGF.T, p.AGr/p.Nwvec)
                else:
                    d = np.append(d, np.dot(p.AGF.T, p.AGr/p.Nwvec))

                # compute F^TG(G^TNG)^{-1}G^TF
                right = ((1/p.Nwvec) * p.AGF.T).T
                FGGNGGF.append(np.dot(p.AGF.T, right))

                # log determinant of G^TNG
                logdet_N = np.sum(np.log(p.Nwvec))

                # triple product in likelihood function
                rGGNGGr = np.sum(p.AGr**2/p.Nwvec)
            
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
                if ct == 0:
                    d = np.dot(p.Ftot.T, Nir - NiGcNiGcr)
                else:
                    d = np.append(d, np.dot(p.Ftot.T, Nir - NiGcNiGcr))

                # triple product in likelihood function
                rGGNGGr = np.dot(p.detresiduals, Nir) - np.dot(GcNir, GcNiGcr)

                # compute F^TG(G^TNG)^{-1}G^TF
                FGGNGGF.append(np.dot(NiF.T, p.Ftot) - np.dot(GcNiF.T, GcNiGcF))
                
                
            # first component of likelihood function
            loglike += -0.5 * (logdet_N + rGGNGGr)

        # compute the red noise, DMV and GWB terms in the log likelihood
        
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
    mark 2 log likelihood. Note that this is not the same as mark1 in piccard

    EFAC + EQUAD + Jitter +Red noise + DMV + GWs

    No frequency lines

    Uses Woodbury lemma

    """

    def mark2LogLikelihood(self, parameters):

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
            UPhiU = np.dot(self.psr[0].UtF, np.dot(Phi0, self.psr[0].UtF.T))
            Phi = UPhiU + np.diag(self.psr[0].Qamp) 
            
            try:
                cf = sl.cho_factor(Phi)
                self.logdetPhi = 2*np.sum(np.log(np.diag(cf[0])))
                self.Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Phi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                self.logdetPhi = np.sum(np.log(s))
                self.Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

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
            U, s, Vh = sl.svd(Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            logdet_Sigma = np.sum(np.log(s))
            expval2 = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, d)))


        loglike += -0.5 * (self.logdetPhi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) 

        return loglike

    
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
