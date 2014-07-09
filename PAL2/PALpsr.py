#!/usr/bin/env python

"""

PALpsr.py

This file is a basic implementation of a pulsar class that stores all information
about a given pulsar, including likelihood specific pre-computed quantities.


This file was originally developed by Rutger van Haasteren and is modified here

"""

from __future__ import division

import numpy as np
import scipy.linalg as sl
import scipy.special as ss
import h5py as h5
import matplotlib.pyplot as plt
import os, sys
import json
import tempfile
import libstempo as t2

import PALutils
import PALdatafile

PAL_DMk = 4.15e3        # Units MHz^2 cm^3 pc sec

PAL_spd = 86400.0       # Seconds per day
#PAL_spy = 31556926.0   # Wrong definition of YEAR!!!
PAL_spy =  31557600.0   # Seconds per year (yr = 365.25 days, so Julian years)
PAL_T0 = 53000.0        # MJD to which all HDF5 toas are referenced




class Pulsar(object):

    """
    Pulsar class

    """
    parfile_content = None      # The actual content of the original par-file
    timfile_content = None      # The actual content of the original tim-file
    #t2psr = None                # A libstempo object, if libstempo is imported

    raj = 0
    decj = 0
    toas = None
    toaerrs = None
    prefitresiduals = None
    residuals = None
    detresiduals = None     # Residuals after subtraction of deterministic sources
    freqs = None
    Gmat = None
    Gcmat = None
    Mmat = None
    ptmpars = []
    ptmparerrs = []
    ptmdescription = []
    flags = None
    name = "J0000+0000"

    def __init__(self):
        """
        Initialize pulsar class
        """

        self.parfile_content = None
        self.timfile_content = None
        self.t2psr = None
        self.incRed = False
        self.incDM = False

        self.raj = 0                    # psr right ascension
        self.decj = 0                   # psr declination
        self.toas = None                # psr TOAs
        self.toaerrs = None             # psr toaerrs
        self.prefitresiduals = None     # psr prefit residuals
        self.residuals = None           # psr residuals
        self.detresiduals = None        # Residuals after subtraction of deterministic sources
        self.freqs = None               # frequencies used in red noise modeling
        #self.unitconversion = None
        self.Gmat = None                # pulsar G matrix
        self.Gcmat = None               # pulsar compementary G matrix
        self.Mmat = None                # pulsar design matrix
        self.ptmpars = []               # pulsar timing model parameters    
        self.ptmparerrs = []            # pulsar timing model parameter uncertainties
        self.ptmdescription = []        # pulsr timing model names
        self.flags = None               # pulsar flags (efac or equad)
        self.name = "J0000+0000"        # pulsar tname

        self.Fmat = None                # F-matrix for red noise modeling
        self.SFmat = None               # single frequency F matrix (for modeling lines)
        self.FFmat = None               # total F matrix if using SFmatrix
        self.Fdmmat = None              # DM F matrix
        self.Hmat = None                # compression matrix
        self.Homat = None               # orthogonal compression matrix
        self.Hcmat = None               # complement to compression matrix
        self.Hocmat = None              # orthogonal complement to compression matrix
        self.Umat = None                # exploder matrix
        self.avetoas = None             # vecotr of average toas
        self.Dmat = None                
        self.DF = None
        self.Ffreqs = None
        self.SFfreqs = None
        self.Fdmfreqs = None
        self.Emat = None
        self.EEmat = None
        self.Gr = None
        self.GGr = None
        self.GtF = None
        self.GtD = None
        #self.GGtFF = None
        self.GGtD = None


        self.Qam = 0.0

    """
    Read the pulsar data (TOAs, residuals, design matrix, etc..) from an HDF5
    file

    @param h5df:        The DataFile object we are reading from
    @param psrname:     Name of the Pulsar to be read from the HDF5 file
    """
    def readFromH5(self, h5df, psrname):
        h5df.readPulsar(self, psrname)

    
    """
    Initialise the libstempo object for use in nonlinear timing model modelling.
    No parameters are required, all content must already be in memory
    """
    def initLibsTempoObject(self):
        # Check that the parfile_content and timfile_content are set
        if self.parfile_content is None or self.timfile_content is None:
            raise ValueError('No par/tim file present for pulsar {0}'.format(self.name))

        # For non-linear timing models, libstempo must be imported
        if t2 is None:
            raise ImportError("libstempo")

        # Write a temporary par-file and tim-file for libstempo to read. First
        # obtain 
        parfilename = tempfile.mktemp()
        timfilename = tempfile.mktemp()
        parfile = open(parfilename, 'w')
        timfile = open(timfilename, 'w')
        parfile.write(self.parfile_content)
        timfile.write(self.timfile_content)
        parfile.close()
        timfile.close()

        # Create the libstempo object
        try:
            self.t2psr = t2.tempopulsar(parfilename, timfilename, maxobs=20000)
        except TypeError:
            self.t2psr = t2.tempopulsar(parfilename, timfilename)


        # Create the BATS?
        # tempresiduals = self.t2psr.residuals(updatebats=True, formresiduals=False)

        # Delete the temporary files
        os.remove(parfilename)
        os.remove(timfilename)

    """
    Constructs a new modified design matrix by adding some columns to it. Returns
    a list of new objects that represent the new timing model

    @param newpars:     Names of the parameters/columns that need to be added
                        For now, can only be [Offset, F0, F1, DM, DM0, DM1]
                        (or higher derivatives of F/DM)

    @param oldMat:      The old design matrix. If None, use current one

    @param oldGmat:     The old G-matrix. ''

    @param oldGcmat:    The old co-G-matrix ''

    @param oldptmpars:  The old timing model parameter values

    @param oldptmdescription:   The old timing model parameter labels

    @param noWarning:   If True, do not warn for duplicate parameters

    @return (list):     Return the elements: (newM, newG, newGc,
                        newptmpars, newptmdescription)
                        in order: the new design matrix, the new G-matrix, the
                        new co-Gmatrix (orthogonal complement), the new values
                        of the timing model parameters, the new descriptions of
                        the timing model parameters. Note that the timing model
                        parameters are not really 'new', just re-selected
    """
    def addToDesignMatrix(self, addpars, \
            oldMmat=None, oldGmat=None, oldGcmat=None, \
            oldptmpars=None, oldptmdescription=None, \
            noWarning=False):
        if oldMmat is None:
            oldMmat = self.Mmat
        if oldptmdescription is None:
            oldptmdescription = self.ptmdescription
        if oldptmpars is None:
            oldptmpars = self.ptmpars
        #if oldunitconversion is None:
        #    oldunitconversion = self.unitconversion
        if oldGmat is None:
            oldGmat = self.Gmat
        if oldGcmat is None:
            oldGcmat = self.Gcmat

        # First make sure that the parameters we are adding are not already in
        # the design matrix
        indok = np.array([1]*len(addpars), dtype=np.bool)
        addpars = np.array(addpars)
        #for ii, parlabel in enumerate(oldptmdescription):
        for ii, parlabel in enumerate(addpars):
            if parlabel in oldptmdescription:
                indok[ii] = False

        if sum(indok) != len(indok) and not noWarning:
            print "WARNING: cannot add parameters to the design matrix that are already present"
            print "         refusing to add:", map(str, addpars[indok == False])

        # Only add the parameters with indok == True
        if sum(indok) > 0:
            # We have some parameters to add
            addM = np.zeros((oldMmat.shape[0], np.sum(indok)))
            adddes = map(str, addpars[indok])
            addparvals = []
            addunitvals = []

            Dmatdiag = pic_DMk / (self.freqs**2)
            for ii, par in enumerate(addpars[indok]):
                addparvals.append(0.0)
                addunitvals.append(1.0)
                if par == 'DM':
                    addM[:, ii] = Dmatdiag.copy()
                elif par[:2] == 'DM':
                    power = int(par[2:])
                    addM[:, ii] = Dmatdiag * (self.toas ** power)
                elif par == 'Offset':
                    addM[:, ii] = 1.0
                elif par[0] == 'F':
                    try:
                        power = int(par[1:])
                        addM[:, ii] = (self.toas ** power)
                    except ValueError:
                        raise ValueError("ERROR: parameter {0} not implemented in 'addToDesignMatrix'".format(par))
                else:
                    raise ValueError("ERROR: parameter {0} not implemented in 'addToDesignMatrix'".format(par))

            newM = np.append(oldMmat, addM, axis=1)
            newptmdescription = np.append(oldptmdescription, adddes)
            newptmpars = np.append(oldptmpars, addparvals)
            #newunitconversion = np.append(oldunitconversion, addunitvals)

            # Construct the G-matrices
            U, s, Vh = sl.svd(newM)
            newG = U[:, (newM.shape[1]):].copy()
            newGc = U[:, :(newM.shape[1])].copy()
        else:
            newM = oldMmat.copy()
            newptmdescription = np.array(oldptmdescription)
            #newunitconversion = np.array(oldunitconversion)
            newptmpars = oldptmpars.copy()

            if oldGmat is not None:
                newG = oldGmat.copy()
                newGc = oldGcmat.copy()
            else:
                U, s, Vh = sl.svd(newM)
                newG = U[:, (newM.shape[1]):].copy()
                newGc = U[:, :(newM.shape[1])].copy()


        return newM, newG, newGc, newptmpars, map(str, newptmdescription)

    """
    Constructs a new modified design matrix by deleting some columns from it.
    Returns a list of new objects that represent the new timing model

    @param delpars:     Names of the parameters/columns that need to be deleted.
    
    @return (list):     Return the elements: (newM, newptmpars, newptmdescription)
                        in order: the new design matrix , the new values
                        of the timing model parameters, the new descriptions of
                        the timing model parameters. Note that the timing model
                        parameters are not really 'new', just re-selected
    """
    def delFromDesignMatrix(self, delpars):

        # First make sure that the parameters we are deleting are actually in
        # the design matrix
        inddel = np.array([1]*len(delpars), dtype=np.bool)
        indkeep = np.array([1]*self.Mmat.shape[1], dtype=np.bool)
        delpars = np.array(delpars)
        for ii, parlabel in enumerate(delpars):
            if not parlabel in self.ptmdescription:
                inddel[ii] = False
                print "WARNING: {0} not in design matrix. Not deleting".format(parlabel)
            else:
                index = np.flatnonzero(np.array(self.ptmdescription) == parlabel)
                indkeep[index] = False

        if np.sum(indkeep) != len(indkeep):
            # We have actually deleted some parameters
            newM = self.Mmat[:, indkeep]
            newptmdescription = np.array(self.ptmdescription)[indkeep]
            newptmpars = self.ptmpars[indkeep]

        else:
            newM = self.Mmat.copy()
            newptmdescription = np.array(self.ptmdescription)
            newptmpars = self.ptmpars.copy()

        return newM, newptmpars, map(str, newptmdescription)


    """
    Construct a modified design matrix, based on some options. Returns a list of
    new objects that represent the new timing model

    @param addDMQSD:    Whether we should make sure that the DM quadratics are
                        fit for. Should have 'DM', 'DM1', 'DM2'. If not present,
                        add them
    @param addQSD:      Same as addDMQSD, but now for pulsar spin frequency.
    @param removeJumps: Remove the jumps from the timing model.
    @param removeAll:   This removes all parameters from the timing model,
                        except for the DM parameters, and the QSD parameters

    @return (list):     Return the elements: (newM, newG, newGc,
                        newptmpars, newptmdescription)
                        in order: the new design matrix, the new G-matrix, the
                        new co-Gmatrix (orthogonal complement), the new values
                        of the timing model parameters, the new descriptions of
                        the timing model parameters. Note that the timing model
                        parameters are not really 'new', just re-selected

    TODO: Split this function in two parts. One that receives a list of
          names/identifiers of which parameters to include. The other that
          constructs the list, and calls that function.
    """
    def getModifiedDesignMatrix(self, addDMQSD=False, addQSD=False, \
            removeJumps=False, removeAll=False):
        
        (newM, newG, newGc, newptmpars, newptmdescription) = \
                (self.Mmat, self.Gmat, self.Gcmat, self.ptmpars, \
                self.ptmdescription)

        # DM and QSD parameter names
        dmaddes = ['DM', 'DM1', 'DM2']
        qsdaddes = ['Offset', 'F0', 'F1']

        # See which parameters need to be added
        addpar = []
        for parlabel in dmaddes:
            if addDMQSD and not parlabel in self.ptmpars:
                addpar += [parlabel]
        for parlabel in qsdaddes:
            if addQSD and not parlabel in self.ptmpars:
                addpar += [parlabel]

        # Add those parameters
        if len(addpar) > 0:
            (newM, newG, newGc, newptmpars, newptmdescription) = \
                    self.addToDesignMatrix(addpar, newM, newG, newGc, \
                    newptmpars, newptmdescription, \
                    noWarning=True)

        # See whether some parameters need to be deleted
        delpar = []
        for ii, parlabel in enumerate(self.ptmdescription):
            if (removeJumps or removeAll) and parlabel[:4].upper() == 'JUMP':
                delpar += [parlabel]
            elif removeAll and (not parlabel in dmaddes and not parlabel in qsdaddes):
                delpar += [parlabel]

        # Delete those parameters
        if len(delpar) > 0:
            (newM, newG, newGc, newptmpars, newptmdescription) = \
                    self.delFromDesignMatrix(delpar, newM, newG, newGc, \
                    newptmpars, newptmdescription)

        return newM, newG, newGc, newptmpars, newptmdescription




    # Modify the design matrix to include fitting for a quadratic in the DM
    # signal.
    # TODO: Check if the DM is fit for in the design matrix. Use ptmdescription
    #       for that. It should have a field with 'DM' in it.
    def addDMQuadratic(self):
        self.Mmat, self.Gmat, self.Gcmat, self.ptmpars, \
                self.ptmdescription = \
                self.getModifiedDesignMatrix(addDMQSD=True, removeJumps=False)

    """
    Figure out what the list of timing model parameters is that needs to be
    deleted from the design matrix in order to do nonlinear timing model
    parameter analysis, given 

    @param tmpars:  A list of suggested parameters to keep in the design
                    matrix. Only parameters not present in this list and present
                    in the design matrix will be returned.
    @param keep:    If True, return the parameters that we keep in the design
                    matrix. If False, return the parameters that we will delete
                    from the design matrix

    @return:        List of parameters to be deleted
    """
    def getNewTimingModelParameterList(self, keep=True, tmpars=None):

        # Remove from the timing model parameter list of the design matrix,
        # all parameters not in the list 'tmpars'. The parameters not in
        # tmpars are numerically included
        if tmpars is None:
            tmpars = ['Offset', 'F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', \
                    'PX', 'DM', 'DM1', 'DM2']

        tmparkeep = []
        tmpardel = []
        for tmpar in self.ptmdescription:
            if tmpar in tmpars:
                # This parameter stays in the compression matrix (so is
                # marginalised over
                tmparkeep += [tmpar]
            #elif tmpar == 'Offset':
            #    print "WARNING: Offset needs to be included in the design matrix. Including it anyway..."
            #    tmparkeep += [tmpar]
            else:
                tmpardel += [tmpar]

        if keep:
            returnpars = tmparkeep
        else:
            returnpars = tmpardel

        return returnpars

    
    """
    Estimate how many frequency modes are required for this pulsar. This
    function uses a simplified method, based on van Haasteren (2013). Given a
    red-noise spectrum (power-law here), a high-fidelity compression technique
    is used.

    @param noiseAmp:    the expected amplitude we want to be fully sensitive to
    @param noiseSi:     the spectral index of the signal we want to be fully
                        sensitive to
    @param Tmax:        time baseline, if not determined from this pulsar
    @param threshold:   the fidelity with which the signal has to be
                        reconstructed
    """
    def numfreqsFromSpectrum(self, noiseAmp, noiseSi, \
            Tmax=None, threshold=0.99, dm=False):
        ntoas = len(self.toas)
        nfreqs = int(ntoas/2)

        if Tmax is None:
            Tmax = np.max(self.toas) - np.min(self.toas)

        # Construct the Fourier modes, and the frequency coefficients (for
        # noiseAmp=1)
        Fmat, Ffreqs = PALutils.createfourierdesignmatrix(self.toas, \
                                            nfreqs, freq=True, Tspan=Tmax)
        #(Fmat, Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
        freqpy = Ffreqs * PAL_spy
        pcdoubled = (PAL_spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-noiseSi)

        if dm:
            # Make Fmat into a DM variation Fmat
            Dvec = PAL_DMk / (self.freqs**2)
            Fmat = (Dvec * Fmat.T).T

        # Check whether the Gmatrix exists
        if self.Gmat is None:
            U, s, Vh = sl.svd(self.Mmat)
            Gmat = U[:, self.Mmat.shape[1]:]
        else:
            Gmat = self.Gmat

        # Find the Cholesky decomposition of the projected radiometer-noise
        # covariance matrix
        GNG = np.dot(Gmat.T, (self.toaerrs**2 * Gmat.T).T)
        try:
            L = sl.cholesky(GNG).T
            cf = sl.cho_factor(L)
            Li = sl.cho_solve(cf, np.eye(GNG.shape[0]))
        except np.linalg.LinAlgError as err:
            raise ValueError("ERROR: GNG singular according to Cholesky")

        # Construct the transformed Phi-matrix, and perform SVD. That matrix
        # should have a few singular values (nfreqs not precisely determined)
        LGF = np.dot(Li, np.dot(Gmat.T, Fmat))
        Phiw = np.dot(LGF, (pcdoubled * LGF).T)
        U, s, Vh = sl.svd(Phiw)

        # From the eigenvalues in s, we can determine the number of frequencies
        fisherelements = s**2 / (1 + noiseAmp**2 * s)**2
        cumev = np.cumsum(fisherelements)
        totrms = np.sum(fisherelements)

        return int((np.flatnonzero( (cumev/totrms) >= threshold )[0] + 1)/2)

    
    """
    Construct the compression matrix and it's orthogonal complement. This is
    always done, even if in practice there is no compression. That is just the
    fidelity = 1 case.

        # U-compression:
        # W s V^{T} = G^{T} U U^{T} G    H = G Wl
        # F-compression
        # W s V^{T} = G^{T} F F^{T} G    H = G Wl

    @param compression: what kind of compression to use. Can be \
                        None/average/frequencies/avefrequencies
    @param nfmodes:     when using frequencies, use this number if not -1
    @param ndmodes:     when using dm frequencies, use this number if not -1
    @param likfunc:     which likelihood function is being used. Only useful when it
                        is mark4/not mark4. TODO: parameter can be removed?
    @param threshold:   To which fidelity will we compress the basis functions [1.0]
    @param tmpars:      When compressing to a list of timing model parameters,
                        this list of parameters is used.
    """
    # TODO: selection of timing-model parameters should apply to _all_ forms of
    # compression. Still possible to do frequencies and include timing model
    # parameters, as long as we include the complement function
    def constructCompressionMatrix(self, compression='None', \
            nfmodes=-1, ndmodes=-1, likfunc='mark4', threshold=1.0, \
            tmpars = None):

        if compression == 'average':
            
            (self.avetoas, self.Umat) = PALutils.exploderMatrix(self.toas, dt=10)
        
            GU = np.dot(self.Gmat.T, self.Umat)
            GUUG = np.dot(GU, GU.T)

            # Construct an orthogonal basis, and singular values
            Vmat, svec, Vhsvd = sl.svd(GUUG)

            # Decide how many basis vectors we'll take. (Would be odd if this is
            # not the number of columns of self.U. How to test? For now, use
            # 99.9% of rms power
            cumrms = np.cumsum(svec)
            totrms = np.sum(svec)
            #print "svec:   ", svec
            #print "cumrms: ", cumrms
            #print "totrms: ", totrms
            inds = (cumrms/totrms) >= threshold
            if np.sum(inds) > 0:
                # We can compress
                l = np.flatnonzero( inds )[0] + 1
            else:
                # We cannot compress, keep all
                l = self.Umat.shape[1]

            print "Number of U basis vectors for " + \
                    self.name + ": " + str(self.Umat.shape) + \
                    " --> " + str(l)
            """
            ll = self.Umat.shape[1]

            print "Umat: ", self.Umat.shape
            print "CumRMS:    ", cumrms
            print "cumrms[l] / tot = ", cumrms[l] / totrms
            print "svec range:   ", svec[130:150]
            print "cumrms range: ", cumrms[130:150]
            print "designmatrix: ", self.Mmat.shape
            print "TMPars: ", self.ptmdescription
            plt.plot(np.arange(120,ll+10), np.log10(svec[120:ll+10]), 'k-')
            """

            # H is the compression matrix
            Bmat = Vmat[:, :l].copy()
            Bomat = Vmat[:, l:].copy()
            H = np.dot(self.Gmat, Bmat)
            Ho = np.dot(self.Gmat, Bomat)

            # Use another SVD to construct not only Hmat, but also Hcmat
            # We use this version of Hmat, and not H from above, in case of
            # linear dependences...
            #svec, Vmat = sl.eigh(H)
            Vmat, s, Vh = sl.svd(H)
            self.Hmat = Vmat[:, :l]
            self.Hcmat = Vmat[:, l:]

            # For compression-complements, construct Ho and Hoc
            if Ho.shape[1] > 0:
                Vmat, s, Vh = sl.svd(Ho)
                self.Homat = Vmat[:, :Ho.shape[1]]
                self.Hocmat = Vmat[:, Ho.shape[1]:]
            else:
                self.Homat = np.zeros((Vmat.shape[0], 0))
                self.Hocmat = np.eye(Vmat.shape[0])

        elif compression == 'frequencies':
            # Use a power-law spectrum with spectral-index of 4.33
            freqpy = self.Ffreqs * PAL_spy
            phivec = (PAL_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-4.33)
            phivec = np.ones(len(freqpy))

            GF = np.dot(self.Gmat.T, self.Fmat * phivec)
            GFFG = np.dot(GF, GF.T)
            Vmat, svec, Vhsvd = sl.svd(GFFG)

            cumrms = np.cumsum(svec)
            totrms = np.sum(svec)
            # print "Freqs: ", cumrms / totrms
            l = np.flatnonzero( (cumrms/totrms) >= threshold )[0] + 1

            # choose L based on rough cadence of 2 weeks ^-1
            dt = 14 * 86400
            Tspan = self.toas.max() - self.toas.min()
            l = int(Tspan/dt)*2
            l = 51
            if l < self.Gmat.shape[1]:
                pass
            else:
                l = self.Gmat.shape[1]-1
            print 'Using {0} components for PSR {1}'.format(l, self.name)

            # H is the compression matrix
            Bmat = Vmat[:, :l].copy()
            Bomat = Vmat[:, l:].copy()
            H = np.dot(self.Gmat, Bmat)
            Ho = np.dot(self.Gmat, Bomat)

            # Use another SVD to construct not only Hmat, but also Hcmat
            # We use this version of Hmat, and not H from above, in case of
            # linear dependences...
            #svec, Vmat = sl.eigh(H)
            Vmat, s, Vh = sl.svd(H)
            self.Hmat = H
            #self.Hmat = Vmat[:, :l]
            self.Hcmat = Vmat[:, l:]

            # For compression-complements, construct Ho and Hoc
            Vmat, s, Vh = sl.svd(Ho)
            self.Homat = Vmat[:, :Ho.shape[1]]
            self.Hocmat = Vmat[:, Ho.shape[1]:]

        elif compression == 'dmfrequencies' or compression == 'avefrequencies':
            print "WARNING: compression on DM frequencies not normalised correctly!"

            Ftot = np.zeros((len(self.toas), 0))

            # Decide on the (dm)frequencies to include
            if nfmodes == -1:
                # Include all, and only all, frequency modes
                #Ftot = np.append(Ftot, self.Fmat, axis=1)

                # Produce an orthogonal basis for the frequencies
                l = self.Fmat.shape[1]
                Vmat, svec, Vhsvd = sl.svd(self.Fmat)
                Ftot = np.append(Ftot, Vmat[:, :l].copy(), axis=1)
            elif nfmodes == 0:
                # Why would anyone do this?
                pass
            else:
                # Should we check whether nfmodes is not too large?
                #Ftot = np.append(Ftot, self.Fmat[:, :nfmodes], axis=1)

                # Produce an orthogonal basis for the frequencies
                l = nfmodes
                Vmat, svec, Vhsvd = sl.svd(self.Fmat)
                Ftot = np.append(Ftot, Vmat[:, :l].copy(), axis=1)

            if ndmodes == -1:
                # Include all, and only all, frequency modes
                # Ftot = np.append(Ftot, self.DF, axis=1)

                # Produce an orthogonal basis for the frequencies
                l = self.DF.shape[1]
                Vmat, svec, Vhsvd = sl.svd(self.DF)
                Ftot = np.append(Ftot, Vmat[:, :l].copy(), axis=1)
            elif ndmodes == 0:
                # Do not include DM in the compression
                pass
            else:
                # Should we check whether nfmodes is not too large?
                # Ftot = np.append(Ftot, self.DF[:, :ndmodes], axis=1)

                # Produce an orthogonal basis for the frequencies
                l = self.DF.shape[1]
                Vmat, svec, Vhsvd = sl.svd(self.DF)
                Ftot = np.append(Ftot, Vmat[:, :l].copy(), axis=1)

        elif compression == 'dont' or compression is None or compression == 'None':     # Do not compress
            self.Hmat = self.Gmat
            self.Hcmat = self.Gcmat
            self.Homat = np.zeros(self.Gmat.shape)
            pass
        else:
            raise IOError, "Invalid compression argument"

  

    """
    For every pulsar, quite a few Auxiliary quantities (like GtF etc.) are
    necessary for the evaluation of various likelihood functions. This function
    calculates these quantities, and optionally writes them to the HDF5 file for
    quick use later.

    @param h5df:            The DataFile we will write things to
    @param Tmax:            The full duration of the experiment
    @param nfreqs:          The number of noise frequencies we require for this
                            pulsar
    @param ndmfreqs:        The number of DM frequencies we require for this pulsar
    @param twoComponent:    Whether or not we do the two-component noise
                            acceleration
    @param nSingleFreqs:    The number of single floating noise frequencies
    @param nSingleDMFreqs:  The number of single floating DM frequencies
    @param compression:     Whether we use compression (None/frequencies/average)
    @param likfunc:         Which likelihood function to do it for (all/markx/..)
    @param write:           Which data to write to the HDF5 file ('no' for no
                            writing, 'likfunc' for the current likfunc, 'all'
                            for all quantities

    """
    def createPulsarAuxiliaries(self, h5df, Tmax, nfreqs, ndmfreqs, \
            twoComponent=False, nSingleFreqs=0, nSingleDMFreqs=0, \
            compression='None', likfunc='mark1', write='no', \
            tmpars=None, memsave=True, logf=False):


        # For creating the auxiliaries it does not really matter: we are now
        # creating all quantities per default
        self.twoComponentNoise = twoComponent

        # get Tmax
        self.Tmax = Tmax

        # construct average quantities
        useAverage = likfunc == 'mark2'

        # write compression to hdf5 file
        if write != 'no':
            h5df.addData(self.name, 'PAL_compression', compression)

        if compression == 'red':
            threshold = 0.99
        else:
            threshold = 1.0

        # default for detresiduals
        self.detresiduals = self.residuals.copy()

        # construct std dev of data for use in priors
        self.sig_data = self.residuals.std()

        # Before writing anything to file, we need to know right away how many
        # fixed and floating frequencies this model contains.
        nf = 0 ; ndmf = 0 ; nsf = nSingleFreqs ; nsdmf = nSingleDMFreqs
        if nfreqs is not None and nfreqs != 0:
            nf = nfreqs
        if ndmfreqs is not None and ndmfreqs != 0:
            ndmf = ndmfreqs

        # allocate single frequency matrices
        self.nSingleFreqs = nsf
        self.nSingleDMFreqs = nsdmf
        if nsf > 0:
            self.SFfreqs = np.zeros(nsf)
        if nsdmf > 0:
            self.DMSFfreqs = np.zeros(nsdmf)

        # Write these numbers to the HDF5 file
        if write != 'no':
            # Check whether the frequencies already exist in the HDF5-file. If
            # so, compare with what we have here. If they differ, then print out
            # a warning.
            modelFrequencies = np.array([nf, ndmf, nsf, nsdmf])
            try:
                file_modelFreqs = np.array(h5df.getData(self.name, 'PAL_modelFrequencies'))
                if not np.all(modelFrequencies == file_modelFreqs):
                    print "WARNING: model frequencies already present in {0} differ from the current".format(h5df.filename)
                    print "         model. Overwriting..."
            except IOError:
                pass

            h5df.addData(self.name, 'PAL_modelFrequencies', modelFrequencies)
            h5df.addData(self.name, 'PAL_Tmax', [Tmax])
        
        # Create the daily averaged residuals
        if useAverage:
            (self.avetoas, self.avefreqs, self.aveflags, self.Umat) = \
                        PALutils.exploderMatrix(self.toas, freqs=self.freqs, \
                                            flags=np.array(self.flags), dt=10)
            
            # for now just call again with tobs to get average tobs
            (self.avetoas, self.avefreqs, self.avetobs, self.Umat) = \
                        PALutils.exploderMatrix(self.toas, freqs=self.freqs, \
                                            flags=np.array(self.tobsflags), dt=10)

        # create daily averaged residual matrix
        #(self.avetoas, self.aveerr, self.Qmat) = PALutils.dailyAveMatrix(self.toas, self.toaerrs, dt=10)

        # Create the Fourier design matrices for noise
        if nf > 0:
            self.incRed = True
            (self.Fmat, self.Ffreqs) = PALutils.createfourierdesignmatrix(self.toas, \
                                                            nf, Tspan=Tmax, freq=True, \
                                                            logf=logf)
            if useAverage:
                (self.FAvmat, tmp) = PALutils.createfourierdesignmatrix(self.avetoas, \
                                                                nf, Tspan=Tmax, freq=True, \
                                                                logf=logf)
            self.kappa = np.zeros(2*nf)
        else:
            self.Fmat = np.zeros((len(self.toas), 0))
            self.Ffreqs = np.zeros(0)
            self.kappa = np.zeros(2*nf)

        # Create the Fourier design matrices for DM variations
        if ndmf > 0:
            self.incDM = True
            (self.Fdmmat, self.Fdmfreqs) = PALutils.createfourierdesignmatrix(self.toas, \
                                                            ndmf, Tspan=Tmax, freq=True, \
                                                            logf=logf)
            if useAverage:
                (self.FdmAvmat, tmp) = PALutils.createfourierdesignmatrix(self.avetoas, \
                                                                ndmf, Tspan=Tmax, freq=True, \
                                                                logf=logf)
                self.DAvmat = PAL_DMk / (self.avefreqs**2)
                self.DFAv = (self.DAvmat * self.FdmAvmat.T).T
                
            Dmat = PAL_DMk / (self.freqs**2)
            self.DF = (Dmat * self.Fdmmat.T).T

            self.kappadm = np.zeros(2*ndmf)
        else:
            self.Fdmmat = np.zeros((len(self.freqs), 0))
            self.Fdmfreqs = np.zeros(0)
            Dmat = PAL_DMk / (self.freqs**2)
            self.DF = np.zeros((len(self.freqs), 0))
            self.kappadm = np.zeros(2*ndmf)
            if useAverage:
                self.DFAv = np.zeros((len(self.freqs), 0))

        # create total F matrix if both red and DM
        if ndmf > 0 and nf > 0:
            self.Ftot = np.concatenate((self.Fmat, self.DF), axis=1)
            if useAverage:
                self.FtotAv = np.concatenate((self.FAvmat, self.DFAv), axis=1)
        elif ndmf > 0 and nf == 0:
            self.Ftot = self.DF
            if useAverage:
                self.FtotAv = self.DFAv
        elif ndmf == 0 and nf > 0:
            self.Ftot = self.Fmat
            if useAverage:
                self.FtotAv = self.FAvmat

        # Write these quantities to disk
        if write != 'no':
            h5df.addData(self.name, 'Fmat', self.Fmat)
            h5df.addData(self.name, 'Ftot', self.Ftot)
            h5df.addData(self.name, 'Ffreqs', self.Ffreqs)
            h5df.addData(self.name, 'DF', self.DF)
            h5df.addData(self.name, 'Fdmfreqs', self.Fdmfreqs)
            
            if useAverage:
                h5df.addData(self.name, 'FtotAv', self.FtotAv)
                h5df.addData(self.name, 'FAvmat', self.FAvmat)
                h5df.addData(self.name, 'avetoas', self.avetoas)
                h5df.addData(self.name, 'avefreqs', self.avefreqs)
                h5df.addData(self.name, 'aveflags', self.aveflags)
                h5df.addData(self.name, 'avetobs', self.avetobs)
                h5df.addData(self.name, 'Umat', self.Umat)
                h5df.addData(self.name, 'DFAv', self.DFAv)

        # Next we'll need the G-matrices, and the compression matrices.
        if tmpars is not None:
            # list of parameters to delete from design matrix
            if likfunc == 'mark4' or likfunc == 'mark5':
                print 'Including all timing model parameters Numerically'
                Mmat = self.Mmat
            else:
                tmparkeep = self.getNewTimingModelParameterList(keep=True, tmpars=tmpars)
                print 'Numerically including', tmparkeep
                Mmat, newptmpars, newptmdescription = self.delFromDesignMatrix(tmparkeep)
            
            tmpardel = self.getNewTimingModelParameterList(keep=False, tmpars=tmpars)
            print 'Analytically marginalizing over', tmpardel
            self.Mmat, newptmpars, newptmdescription = self.delFromDesignMatrix(tmpardel)

            w = 1.0 / self.toaerrs**2
            Sigi = np.dot(self.Mmat.T, (w * self.Mmat.T).T)
            #try:
            #    cf = sl.cho_factor(Sigi)
            #    Sigma = sl.cho_solve(cf, np.eye(Sigi.shape[0]))
            #except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(Sigi)
            if not np.all(s > 0):
                raise ValueError("Sigi singular according to SVD")
            Sigma = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))
            
            # set fisher matrix
            self.fisher = Sigma
            self.fisherU = U
            self.fisherS = s

        else:
            Mmat = self.Mmat

        U, s, Vh = sl.svd(Mmat)
        self.Gmat = U[:, Mmat.shape[1]:].copy()
        self.Gcmat = U[:, :Mmat.shape[1]].copy()

        # Construct the compression matrix
        self.constructCompressionMatrix(compression, nfmodes=2*nf,
                ndmodes=2*ndmf, threshold=threshold)

        if write != 'no':
            if memsave == False:
                h5df.addData(self.name, 'Gmat', self.Gmat)
                h5df.addData(self.name, 'Gcmat', self.Gcmat)
                h5df.addData(self.name, 'Hmat', self.Hmat)
                h5df.addData(self.name, 'Homat', self.Homat)
                h5df.addData(self.name, 'Hocmat', self.Hocmat)

            # write Hcmat if compression is none
            if compression == None or compression == 'None':
                h5df.addData(self.name, 'Hcmat', self.Hcmat)

        # basic quantities
        self.Gr = np.dot(self.Hmat.T, self.residuals)
        self.GGr = np.dot(self.Hmat, self.Gr)
        GtF = np.dot(self.Hmat.T, self.Ftot)
        
        if useAverage:
            GtU = np.dot(self.Hmat.T, self.Umat)
            self.UtF = self.FtotAv


        if write != 'no':
            h5df.addData(self.name, 'Gr', self.Gr)
            h5df.addData(self.name, 'GGr', self.GGr)
            
            if useAverage:
                h5df.addData(self.name, 'UtF', self.UtF)

        
        # two component noise stuff
        if self.twoComponentNoise:
            GNG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
            self.Amat, self.Wvec, v = sl.svd(GNG)
            #self.Wvec, self.Amat = sl.eigh(GNG) 

            self.AGr = np.dot(self.Amat.T, self.Gr)
            self.AGF = np.dot(self.Amat.T, GtF)
            if useAverage:
                self.AGU = np.dot(self.Amat.T, GtU)

            if write != 'no':

                if memsave == False:
                    h5df.addData(self.name, 'Amat', self.Amat)

                h5df.addData(self.name, 'Wvec', self.Wvec)
                h5df.addData(self.name, 'AGr', self.AGr)
                h5df.addData(self.name, 'AGF', self.AGF)
                
                if useAverage:
                    h5df.addData(self.name, 'AGU', self.AGU)


            # Diagonalise HotEfHo
            if self.Homat.shape[1] > 0:
                HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                self.Wovec, Aomat = sl.eigh(HotNeHo)

                Hor = np.dot(self.Homat.T, self.residuals)
                self.AoGr = np.dot(Aomat.T, Hor)
                if useAverage:
                    HotU = np.dot(self.Homat.T, self.Umat)
                    self.AoGU = np.dot(Aomat.T, HotU)
            else:
                self.Wovec = np.zeros(0)
                Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                if useAverage:
                    self.AoGU = np.zeros((0, GtU.shape[1]))

            if write != 'no':
                h5df.addData(self.name, 'Wovec', self.Wovec)
                h5df.addData(self.name, 'AoGr', self.AoGr)

                if useAverage:
                    h5df.addData(self.name, 'AoGU', self.AoGU)

       
        self.nbasis = self.Hmat.shape[1]
        self.nobasis = self.Homat.shape[1]

        if write != 'no':
            h5df.addData(self.name, 'nbasis', self.nbasis)
            h5df.addData(self.name, 'nobasis', self.nobasis)

        if memsave:
            # clear out G and Gc matrices
            self.Gmat = None
            self.Gcmat = None
            self.Hocmat
            self.Hmat = None
            self.Amat = None

            
    """
    For every pulsar, quite a few Auxiliary quantities (like GtF etc.) are
    necessary for the evaluation of various likelihood functions. This function
    reads them from the HDF5 file for

    @param h5df:            The DataFile we read things from
    @param Tmax:            The full duration of the experiment
    @param nfreqs:          The number of noise frequencies we require for this
                            pulsar
    @param ndmfreqs:        The number of DM frequencies we require for this pulsar
    @param twoComponent:    Whether or not we do the two-component noise
                            acceleration
    @param nSingleFreqs:    The number of single floating noise frequencies
    @param nSingleDMFreqs:  The number of single floating DM frequencies
    @param compression:     Whether we use compression (None/frequencies/average)
    @param likfunc:         Which likelihood function to do it for (all/markx/..)
    @param write:           Which data to write to the HDF5 file ('no' for no
                            writing, 'likfunc' for the current likfunc, 'all'
                            for all quantities

    """
    def readPulsarAuxiliaries(self, h5df, Tmax, nfreqs, ndmfreqs, \
            twoComponent=False, nSingleFreqs=0, nSingleDMFreqs=0, \
            compression='None', likfunc='mark1', tmpars=None,  memsave=True):


        # For creating the auxiliaries it does not really matter: we are now
        # creating all quantities per default
        self.twoComponentNoise = twoComponent

        # get Tmax
        self.Tmax = Tmax

        # construct average quantities
        useAverage = likfunc == 'mark2'

        # check for compression from hdf5 file. If it doesn't match we have to re-compute
        try:
            file_compression = str(h5df.getData(self.name, 'PAL_compression'))
        except IOError:
            print 'Assuming compression is None!'
            file_compression = 'None'
        if file_compression != compression:
            raise ValueError('ERROR: compression argument does not match one in hdf5 file! Must re-compute everything :(')

        if compression == 'red':
            threshold = 0.99
        else:
            threshold = 1.0

        # default for detresiduals
        self.detresiduals = self.residuals.copy()
        
        # construct std dev of data for use in priors
        self.sig_data = self.residuals.std()

        # Before writing anything to file, we need to know right away how many
        # fixed and floating frequencies this model contains.
        nf = 0 ; ndmf = 0 ; nsf = nSingleFreqs ; nsdmf = nSingleDMFreqs
        if nfreqs is not None and nfreqs != 0:
            nf = nfreqs
        if ndmfreqs is not None and ndmfreqs != 0:
            ndmf = ndmfreqs
        
        # allocate single frequency matrices
        self.nSingleFreqs = nsf
        self.nSingleDMFreqs = nsdmf
        if nsf > 0:
            self.SFfreqs = np.zeros(nsf)
        if nsdmf > 0:
            self.DMSFfreqs = np.zeros(nsdmf)

        modelFrequencies = h5df.getData(self.name, 'PAL_modelFrequencies')

        
        # read the daily averaged residuals
        if useAverage:
            self.avetoas = h5df.getData(self.name, 'avetoas')
            self.avefreqs = h5df.getData(self.name, 'avefreqs')
            self.aveflags = h5df.getData(self.name, 'aveflags')
            self.avetobs = h5df.getData(self.name, 'avetobs')
            self.Umat = h5df.getData(self.name, 'Umat')


        # Read in the Fourier design matrices for noise
        reComputeF = False
        reComputeFDM = False
        if nf > 0:
            self.incRed = True
            self.Fmat = h5df.getData(self.name, 'Fmat')
            self.Ffreqs = h5df.getData(self.name, 'Ffreqs')

            if len(self.Ffreqs) != 2*nf:
                recomputeF = True
                print 'WARNING: different number of frequencies in file! Recomputing F matrix'
                (self.Fmat, self.Ffreqs) = PALutils.createfourierdesignmatrix(self.toas, \
                                                            nf, Tspan=Tmax, freq=True, \
                                                            logf=logf)
            if useAverage:
                self.FAvmat = h5df.getData(self.name, 'FAvmat')
                if len(self.Ffreqs) != 2*nf:
                    reComputeF = True
                    print 'WARNING: different number of frequencies in file! Recomputing F matrix'
                    (self.FAvmat, tmp) = PALutils.createfourierdesignmatrix(self.avetoas, \
                                                                nf, Tspan=Tmax, freq=True,
                                                                logf=logf)

            self.kappa = np.zeros(2*nf)
        else:
            self.Fmat = np.zeros((len(self.toas), 0))
            self.Ffreqs = np.zeros(0)
            self.kappa = np.zeros(2*nf)

        # Read in the Fourier design matrices for DM variations
        if ndmf > 0:
            self.incDM = True
            self.DF = h5df.getData(self.name, 'DF')
            self.Fdmfreqs = h5df.getData(self.name, 'Fdmfreqs')
            if len(self.Fdmfreqs) != 2*ndmf:
                reComputeFDM = True
                print 'WARNING: different number of frequencies in file! Recomputing DM F matrix'
                (self.Fdmmat, self.Fdmfreqs) = PALutils.createfourierdesignmatrix(self.toas, \
                                                        ndmf, Tspan=Tmax, freq=True, \
                                                        logf=logf)
                Dmat = PAL_DMk / (self.freqs**2)
                self.DF = (Dmat * self.Fdmmat.T).T

            if useAverage:
                self.DFAv = h5df.getData(self.name, 'DFAv')
                if len(self.Fdmfreqs) != 2*ndmf:
                    reComputeFDM = True
                    print 'WARNING: different number of frequencies in file! Recomputing DM F matrix'
                    (self.FdmAvmat, tmp) = PALutils.createfourierdesignmatrix(self.avetoas, \
                                                                ndmf, Tspan=Tmax, freq=True, \
                                                                logf=logf)
                    self.DAvmat = PAL_DMk / (self.avefreqs**2)
                    self.DFAv = (self.DAvmat * self.FdmAvmat.T).T
                
            self.kappadm = np.zeros(2*ndmf)
        else:
            self.Fdmmat = np.zeros((len(self.freqs), 0))
            self.Fdmfreqs = np.zeros(0)
            Dmat = PAL_DMk / (self.freqs**2)
            self.DF = np.zeros((len(self.freqs), 0))
            self.kappadm = np.zeros(2*ndmf)

        # create total F matrix if both red and DM
        if ndmf > 0 and nf > 0:
            self.Ftot = np.concatenate((self.Fmat, self.DF), axis=1)
            if useAverage:
                self.FtotAv = np.concatenate((self.FAvmat, self.DFAv), axis=1)
        elif ndmf > 0 and nf == 0:
            self.Ftot = self.DF
            if useAverage:
                self.FtotAv = self.DFAv
        elif ndmf == 0 and nf > 0:
            self.Ftot = self.Fmat
            if useAverage:
                self.FtotAv = self.FAvmat


        # Next we'll need the G-matrices, and the compression matrices.
        if memsave == False:
            self.Gmat = h5df.getData(self.name, 'Gmat')
            self.Gcmat = h5df.getData(self.name, 'Gcmat')


        # Read in compression matrix
        if compression == None or compression == 'None':
            self.Hcmat = h5df.getData(self.name, 'Hcmat')

        if memsave == False:
            self.Hmat = h5df.getData(self.name, 'Hmat')
            self.Homat = h5df.getData(self.name, 'Homat')
            self.Hocmat = h5df.getData(self.name, 'Hocmat')
        

        # basic quantities
        self.Gr = h5df.getData(self.name, 'Gr')
        self.GGr = h5df.getData(self.name, 'GGr')
        
        if useAverage:
            self.UtF = h5df.getData(self.name, 'UtF')
            if reComputeF or reComputeFDM:
                self.UtF = self.FtotAv
        
        # two component noise stuff
        if self.twoComponentNoise:
            self.Wvec = h5df.getData(self.name, 'Wvec')
            if memsave == False:
                self.Amat = h5df.getData(self.name, 'Amat')


            self.AGr = h5df.getData(self.name, 'AGr')
            self.AGF = h5df.getData(self.name, 'AGF')

            # raise error if Frequencies don't match
            if self.AGF.shape[1] != self.Ftot.shape[1]:
                raise ValueError('ERROR: AGF must be recomputed!!')

            if useAverage:
                self.AGU = h5df.getData(self.name, 'AGU')

            # don't really need
            self.Wovec = h5df.getData(self.name, 'Wovec')
            self.AoGr = h5df.getData(self.name, 'AoGr')
            
            if useAverage:
                self.AoGU = h5df.getData(self.name, 'AoGU')

         
        self.nbasis = h5df.getData(self.name, 'nbasis')
        self.nobasis = h5df.getData(self.name, 'nobasis')

    
    def rms(self):

        """
        Return weighted RMS in seconds

        """

        W = 1/self.toaerrs**2

        return np.sqrt(np.sum(self.residuals**2*W)/np.sum(W))

    def cosMu(self, gwtheta, gwphi):
     """
     Calculate cosine of angle between pulsar and GW

     """
     # calculate unit vector pointing at GW source
     omhat = [np.sin(gwtheta)*np.cos(gwphi), np.sin(gwtheta)*np.sin(gwphi), np.cos(gwtheta)]

     # calculate unit vector pointing to pulsar
     phat = [np.sin(self.theta)*np.cos(self.phi), np.sin(self.theta)*np.sin(self.phi), \
             np.cos(self.theta)]

     return np.dot(omhat, phat)

     

    # TODO: add frequency line stuff

    #def readNoiseFromFile()






    

