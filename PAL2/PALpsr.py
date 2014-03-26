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
    @param threshold:   To which fidelity will we compress the basis functions [1.0]

    """

    def constructCompressionMatrix(self, compression=None, \
                                   nfmodes=-1, ndmodes=-1, threshold=1.0, \
                                   targetAmp=1e-14):

        # initialize matrices
        self.Hmat = self.Gmat
        self.Hcmat = self.Gcmat
        self.Homat = np.zeros((self.Hmat.shape[0], 0))      # There is no complement
        self.Hocmat = np.zeros((self.Hmat.shape[0], 0))

        
        if compression == 'average':

            self.avetoas, self.aveerr, self.Qmat = \
                    PALutils.dailyAveMatrix(self.toas, self.toaerrs, dt=10)
                                                                           

            # projection
            QG = np.dot(self.Qmat.T, self.Gmat)
            GQQG = np.dot(QG.T, QG)


            # Construct an orthogonal basis, and singular values
            Vmat, svec, Vhsvd = sl.svd(GQQG)

        elif compression == 'jitter':

            self.avetoas, self.Umat = PALutils.exploderMatrix(self.toas, freqs=None, dt=10)

            GU = np.dot(self.Gmat.T, self.Umat)
            GUUG = np.dot(GU, GU.T)

            # Construct an orthogonal basis, and singular values
            Vmat, svec, Vhsvd = sl.svd(GUUG)

    
        elif compression == 'frequencies':


            # hard code noise spectral index to GWB index
            noiseSi = 4.33

            # set Maximum observing time span
            Tmax = np.max(self.toas) - np.min(self.toas)

            # Construct the Fourier modes, and the frequency coefficients (for
            # noiseAmp=1)
            Fmat = self.Ftot
            freqpy = self.Ffreqs * PAL_spy
            pcdoubled = (PAL_spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-noiseSi)


            # Check whether the Gmatrix exists
            if self.Gmat is None:
                U, s, Vh = sl.svd(self.Mmat)
                Gmat = U[:, self.Mmat.shape[1]:]
            else:
                Gmat = self.Gmat

            GF = np.dot(Gmat.T, Fmat)
            GFFG = np.dot(GF, (pcdoubled * GF).T)

            # Construct an orthogonal basis, and singular values
            Vmat, svec, Vhsvd = sl.svd(GFFG)

        
        elif compression =='red':


            # use all frequencies
            ntoas = len(self.toas)
            nfreqs = int(ntoas/2)

            # hard code noise spectral index to GWB index
            noiseSi = 4.33
            
            # set Maximum observing time span
            Tmax = np.max(self.toas) - np.min(self.toas)
            
            # option to construct targetAmp from weighted rms
            if targetAmp == 0:
                #rms = self.rms()
                rms = np.std(self.residuals)
                targetAmp = np.sqrt(noiseSi-1)/2.05e-9 * (Tmax/3.16e7)**(2/(noiseSi-1)) \
                        * 1e-15 * rms

            # Construct the Fourier modes, and the frequency coefficients (for
            # noiseAmp=1)
            Fmat, Ffreqs = PALutils.createfourierdesignmatrix(self.toas, \
                                                nfreqs, freq=True, Tspan=Tmax)
            freqpy = Ffreqs * PAL_spy
            pcdoubled = (PAL_spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-noiseSi)


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

            # get number of eigenvectors to keep
            fisherelements = s**2 / (1 + targetAmp**2 * s)**2
            cumev = np.cumsum(fisherelements)
            totrms = np.sum(fisherelements)
            
            l = int((np.flatnonzero( (cumev/totrms) >= threshold )[0] + 1))


            print 'Constructing compression matrix for PSR {0} using {1} components for targetAmp = {2}'.format(self.name, l, targetAmp)

            # construct H
            H = np.dot(Li, U[:,:l])
            GH = np.dot(Gmat, H)
            self.Hmat = GH

        ## NO compression
        elif compression == 'None' or compression is None:
            self.Hmat = self.Gmat
            self.Hcmat = self.Gcmat
            self.Homat = np.zeros((self.Hmat.shape[0], 0))      # There is no complement
            self.Hocmat = np.zeros((self.Hmat.shape[0], 0))
        else:
            raise IOError, "Invalid compression argument"


        # construct compression matrix
        if compression is not None and compression != 'None' and compression != 'red':
        
            # Decide how many basis vectors we'll take.
            cumrms = np.cumsum(svec)
            totrms = np.sum(svec)
            inds = (cumrms/totrms) >= threshold
            if np.sum(inds) > 0:
                # We can compress
                l = np.flatnonzero( inds )[0] + 1
                l = 16
                print 'Using {0} basis components for puslar {1} using {2} compression'\
                        .format(l, self.name, compression)
            else:
                # We cannot compress, keep all
                l = self.Umat.shape[1]
                l = len(svec)

            # H is the compression matrix
            Bmat = Vmat[:, :l].copy()
            Bomat = Vmat[:, l:].copy()
            H = np.dot(self.Gmat, Bmat)
            Ho = np.dot(self.Gmat, Bomat)

            # Use another SVD to construct not only Hmat, but also Hcmat
            # We use this version of Hmat, and not H from above, in case of
            # linear dependences...
            Vmat, s, Vh = sl.svd(H)
            self.Hmat = Vmat[:,:l]
            self.Hcmat = Vmat[:, l:]

            # For compression-complements, construct Ho and Hoc
            if Ho.shape[1] > 0:
                Vmat, s, Vh = sl.svd(Ho)
                self.Homat = Vmat[:, :Ho.shape[1]]
                self.Hocmat = Vmat[:, Ho.shape[1]:]
            else:
                self.Homat = np.zeros((Vmat.shape[0], 0))
                self.Hocmat = np.eye(Vmat.shape[0])

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
            targetAmp=1e-14, memsave=True):


        # For creating the auxiliaries it does not really matter: we are now
        # creating all quantities per default
        self.twoComponentNoise = twoComponent

        # construct average quantities
        useAverage = likfunc == 'mark2'

        if compression == 'red':
            threshold = 0.99
        else:
            threshold = 0.9999

        # default for detresiduals
        self.detresiduals = self.residuals.copy()

        # Before writing anything to file, we need to know right away how many
        # fixed and floating frequencies this model contains.
        nf = 0 ; ndmf = 0 ; nsf = nSingleFreqs ; nsdmf = nSingleDMFreqs
        if nfreqs is not None and nfreqs != 0:
            nf = nfreqs
        if ndmfreqs is not None and ndmfreqs != 0:
            ndmf = ndmfreqs

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
                                                            nf, Tspan=Tmax, freq=True)
            if useAverage:
                (self.FAvmat, tmp) = PALutils.createfourierdesignmatrix(self.avetoas, \
                                                                nf, Tspan=Tmax, freq=True)
            self.kappa = np.zeros(2*nf)
        else:
            self.Fmat = np.zeros((len(self.toas), 0))
            self.Ffreqs = np.zeros(0)
            self.kappa = np.zeros(2*nf)

        # Create the Fourier design matrices for DM variations
        if ndmf > 0:
            self.incDM = True
            (self.Fdmmat, self.Fdmfreqs) = PALutils.createfourierdesignmatrix(self.toas, \
                                                            ndmf, Tspan=Tmax, freq=True)
            if useAverage:
                (self.FdmAvmat, tmp) = PALutils.createfourierdesignmatrix(self.avetoas, \
                                                                ndmf, Tspan=Tmax, freq=True)
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
            h5df.addData(self.name, 'PAL_Fmat', self.Fmat)
            h5df.addData(self.name, 'PAL_Ffreqs', self.Ffreqs)
            h5df.addData(self.name, 'PAL_Fdmmat', self.Fdmmat)
            h5df.addData(self.name, 'PAL_Fdmfreqs', self.Fdmfreqs)
            h5df.addData(self.name, 'PAL_Dmat', self.Dmat)
            h5df.addData(self.name, 'PAL_DF', self.DF)

            h5df.addData(self.name, 'PAL_avetoas', self.avetoas)
            h5df.addData(self.name, 'PAL_Umat', self.Umat)

        # Next we'll need the G-matrices, and the compression matrices.
        U, s, Vh = sl.svd(self.Mmat)
        self.Gmat = U[:, self.Mmat.shape[1]:].copy()
        self.Gcmat = U[:, :self.Mmat.shape[1]].copy()

        #R = PALutils.createRmatrix(self.Mmat, self.toaerrs)
        #self.QR = np.dot(self.Qmat.T, R)
        #self.QRr = np.dot(self.QR, self.residuals)
        #self.QRF = np.dot(self.QR, self.Ftot)

        # Construct the compression matrix
        self.constructCompressionMatrix(compression, nfmodes=2*nf,
                ndmodes=2*ndmf, threshold=threshold, targetAmp=targetAmp)

        if write != 'no':
            h5df.addData(self.name, 'PAL_Gmat', self.Gmat)
            h5df.addData(self.name, 'PAL_Gcmat', self.Gcmat)
            h5df.addData(self.name, 'PAL_Hmat', self.Hmat)
            h5df.addData(self.name, 'PAL_Hcmat', self.Hcmat)
            h5df.addData(self.name, 'PAL_Homat', self.Homat)
            h5df.addData(self.name, 'PAL_Hocmat', self.Hocmat)

        # basic quantities
        self.Gr = np.dot(self.Hmat.T, self.residuals)
        self.GGr = np.dot(self.Hmat, self.Gr)
        GtF = np.dot(self.Hmat.T, self.Ftot)
        
        if useAverage:
            GtU = np.dot(self.Hmat.T, self.Umat)
            self.UtF = self.FtotAv
        
        
        # two component noise stuff
        if self.twoComponentNoise:
            GNG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
            self.Amat, self.Wvec, v = sl.svd(GNG)
            #self.Wvec, self.Amat = sl.eigh(GNG) 

            self.AGr = np.dot(self.Amat.T, self.Gr)
            self.AGF = np.dot(self.Amat.T, GtF)
            if useAverage:
                self.AGU = np.dot(self.Amat.T, GtU)
            

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
       
        self.nbasis = self.Hmat.shape[1]
        self.nobasis = self.Homat.shape[1]
        if memsave:
            # clear out G and Gc maatrices
            self.Gmat = None
            self.Gcmat = None
            self.Hocmat
            self.Hmat = None
            self.Amat = None
    
    
    def rms(self):

        """
        Return weighted RMS in seconds

        """

        W = 1/self.toaerrs**2

        return np.sqrt(np.sum(self.residuals**2*W)/np.sum(W))

     

    # TODO: add frequency line stuff

    #def readNoiseFromFile()






    

