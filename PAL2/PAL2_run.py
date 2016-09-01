#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.special as ss
import glob
import time, os

from PAL2 import PALdatafile
from PAL2 import PALmodels
from PAL2 import PALInferencePTMCMC
from PAL2 import PALpsr
from PAL2 import PALutils

try:
    from mpi4py import MPI
except ImportError:
    print('WARNING: mpi4py not found. Will not perform parallel tempering')
    from PAL2 import nompi4py as MPI


import argparse

parser = argparse.ArgumentParser(description = 'Run PAL2 Data analysis pipeline')

# options
parser.add_argument('--h5File', dest='h5file', action='store', type=str, required=True,
                   help='Full path to hdf5 file containing PTA data')
parser.add_argument('--jsonfile', dest='jsonfile', action='store', type=str, default=None,
                   help='Full path to json file containing model')
parser.add_argument('--fromFile', dest='fromFile', action='store_true', \
                    default=False, help='Init model from file?')

parser.add_argument('--outDir', dest='outDir', action='store', type=str, default='./',
                   help='Full path to output directory (default = ./)')

parser.add_argument('--pulsar', dest='pname', action='store', nargs='+',
                    type=str, required=True, 
                    help='Names of pulsars to use separated by single space')

parser.add_argument('--incRed', dest='incRed', action='store_true',default=False,
                   help='Include Red Noise? [default=False]')
parser.add_argument('--redModel', dest='redModel', action='store', type=str,
                    default='powerlaw', 
                    help='Red noise model [powerlaw, spectrum, broken, interpolate]')
parser.add_argument('--nf', dest='nfreqs', action='store', type=int, default=30,
                   help='number of red noise frequencies to use [default=30]')
parser.add_argument('--fixRedSi', dest='fixRedSi', action='store_true', default=False, \
                    help='Fix red noise spectral index to 4.33')
parser.add_argument('--redAmpPrior', dest='redAmpPrior', action='store', type=str, \
                    default='log', help='Prior on red noise Amplitude [uniform, log]')

parser.add_argument('--incSingleRed', dest='incSingleRed', action='store_true',default=False,
                   help='Include single frequency red noise? [default=False]')

parser.add_argument('--incRedBand', dest='incRedBand', action='store_true',default=False,
                   help='Include band limited red noise (i.e. red noise per frequency band) [default=False]')

parser.add_argument('--incRedEnv', dest='incRedEnv', action='store_true',default=False,
                   help='Include Red Noise Envelope model? [default=False]')
parser.add_argument('--redEnvModel', dest='redEnvModel', action='store', 
                    type=str, default='powerlaw',
                    help='Red noise envelope model [powerlaw, spectrum]')

parser.add_argument('--incRedExt', dest='incRedExt', action='store_true',default=False,
                   help='Include Red Noise Extended model')
parser.add_argument('--redExtModel', dest='redExtModel', action='store', 
                    type=str, default='powerlaw',
                    help='red noise extended model [powerlaw, spectrum]')
parser.add_argument('--nfext', dest='nfext', action='store', type=int, default=10,
                   help='number of red noise frequencies to use in extended model(default=10)')

parser.add_argument('--Tspan', dest='Tspan', action='store', type=float, default=None,
                   help='Tmax to use in red noise and GW expansion (default=None)')


parser.add_argument('--incDM', dest='incDM', action='store_true',default=False,
                   help='Include DM variations [default=False]')
parser.add_argument('--dmModel', dest='dmModel', action='store', type=str, default='powerlaw',
                   help='Red DM noise model [powerlaw, spectrum]')
parser.add_argument('--ndmf', dest='ndmfreqs', action='store', type=int, default=30,
                   help='number of DM noise frequencies to use (default=30)')
parser.add_argument('--DMAmpPrior', dest='DMAmpPrior', action='store', type=str, \
                    default='log', help='prior on DM Amplitude [uniform, log]')

parser.add_argument('--incEphemError', dest='incEphemError', action='store_true',
                    default=False, help='Include Ephemeris error variations [default=False]')


parser.add_argument('--incScat', dest='incScat', action='store_true',default=False,
                   help='Include Scattering variations [default=False]')
parser.add_argument('--scatModel', dest='scatModel', action='store', type=str, default='powerlaw',
                   help='Red Scattering noise model [powerlaw, spectrum]')
parser.add_argument('--nscatf', dest='nscatf', action='store', type=int, default=30,
                   help='number of Scattering noise frequencies to use (default=30)')

parser.add_argument('--incSingleDM', dest='incSingleDM', action='store_true',default=False,
                   help='include single frequency DM [default=False]')

parser.add_argument('--incDMshapelet', dest='incDMshapelet', action='store_true',default=False,
                   help='Include shapelet event for DM [default=False]')
parser.add_argument('--nshape', dest='nshape', type=int, action='store', default=3, \
                   help='number of coefficients for shapelet event for DM [default=3]')
parser.add_argument('--margShapelet', dest='margShapelet', action='store_true', default=False, \
                   help='Analytically marginalize over shapelet coefficients [default=False]')

parser.add_argument('--incDMX', dest='incDMX', action='store_true', default=False, \
                   help='include DMX? [default=False]')
parser.add_argument('--dmxKernelModel', dest='dmxKernelModel', action='store', type=str, \
                    default='None', help='DMX Kernel Model (not currently working)')

parser.add_argument('--incDMBand', dest='incDMBand', action='store_true',default=False,
                   help='Include band limited DM noise for different frequency band [default=False]')

parser.add_argument('--incGWB', dest='incGWB', action='store_true',default=False,
                   help='include GWB? [default=False]')
parser.add_argument('--gwbModel', dest='gwbModel', action='store', type=str, default='powerlaw',
                   help='GWB model [powerlaw, spectrum, turnover]')
parser.add_argument('--fixKappa', dest='fixKappa', action='store', type=float,
                    default=0, help='fix turnover kappa to user value [default=False]')
parser.add_argument('--fixSi', dest='fixSi', action='store', type=float, default=0, \
                    help='Fix GWB spectral index to user defined value [default=False]')
parser.add_argument('--noCorrelations', dest='noCorrelations', action='store_true', \
                    default=False, help='Do not in include GWB correlations [default=False]')
parser.add_argument('--GWBAmpPrior', dest='GWBAmpPrior', action='store', type=str, \
                    default='log', 
                    help='prior on GWB Amplitude [uniform, log, sesana, mcwilliams] [default=log]')
parser.add_argument('--incORF', dest='incORF', action='store_true', \
                    default=False, 
                    help='Include generic ORF to be parameterized [default=False]')
parser.add_argument('--gwbSingleModel', dest='gwbSingleModel', action='store', 
                    type=str, default='powerlaw', 
                    help='GWB point source model [powerlaw, spectrum]')

parser.add_argument('--incGWBSingle', dest='incGWBSingle', action='store_true',default=False,
                   help='include GWB point source? [default=False]')

parser.add_argument('--incGWBAni', dest='incGWBAni', action='store_true',default=False,
                   help='include Anisotropic GWB [default=False]')
parser.add_argument('--lmax', dest='lmax', action='store', type=int, default=2,
                   help='Number of ls to use in anisitropic search [default=2]')
parser.add_argument('--clmPrior', dest='clmPrior', action='store', type=str, 
                    default='uniform', help='clm prior [uniform, phys] [default=uniform]')

parser.add_argument('--incEquad', dest='incEquad', action='store_true',default=False,
                   help='Include Equad [default=False]')
parser.add_argument('--separateEquads', dest='separateEquads', action='store', type=str, \
                    default='frequencies', 
                    help='separate equads [None, backend, frequencies] [default=frequencies]')

parser.add_argument('--noVaryEfac', dest='noVaryEfac', action='store_true',default=False,
                   help='Option to not vary efac [default=False]')
parser.add_argument('--separateEfacs', dest='separateEfacs', action='store', type=str, \
                    default='frequencies', 
                    help='separate efacs [None, backend, frequencies] [default=frequencies]')


parser.add_argument('--incJitterEquad', dest='incJitterEquad', action='store_true',\
                    default=False, help='include Jitter Equad aka ECORR')
parser.add_argument('--separateJitterEquad', dest='separateJitterEquad', action='store', \
                    type=str, default='frequencies', \
                    help='separate Jitter Equad [None, backend, frequencies] [default=frequencies]')

parser.add_argument('--fixNoise', dest='fixNoise', action='store_true',\
                    default=False, help='Fix Noise values')
parser.add_argument('--noVaryNoise', dest='noVaryNoise', action='store_true',\
                    default=False, 
                    help='Fix Noise values to default values [i.e. timing uncertainties only]')
parser.add_argument('--fixWhite', dest='fixWhite', action='store_true',\
                    default=False, help='Fix White Noise values [efac, equad, ecorr]')
parser.add_argument('--noisedir', dest='noisedir', action='store', type=str,
                   help='Full path to directory containting noise values')


parser.add_argument('--incTimingModel', dest='incTimingModel', action='store_true', \
                    default=False, help='Include timing model parameters in run')
parser.add_argument('--tmmodel', dest='tmmodel', action='store', type=str, \
                    default='linear', \
                    help='linear or non-linear timing model [linear, nonlinear] [default=linear]')
parser.add_argument('--fullmodel', dest='fullmodel', action='store_true', \
                    default=False, \
                    help='Use full timing model, no marginalization [default=False]')
parser.add_argument('--noMarg', dest='noMarg', action='store_true',default=False,
                   help='No analytic marginalization [default=False]')
parser.add_argument('--addpars', dest='addpars', action='store', nargs='+', \
                    type=str, required=False, 
                    help='Extra parameters to add to timing model separated by space')
parser.add_argument('--delpars', dest='delpars', action='store', nargs='+', \
                    type=str, required=False, 
                    help='parameters to remove from timing model separated by space')
parser.add_argument('--addAllPars', dest='addAllPars', action='store_true',
                    required=False, help='Add all timing model parameters')

parser.add_argument('--incNonGaussian', dest='incNonGaussian', action='store_true', \
                    default=False, \
                    help='Use non-gaussian likelihood function [default=False]')
parser.add_argument('--nnongauss', dest='nnongauss', action='store', type=int, default=3,
                   help='number of non-guassian components (default=3)')

parser.add_argument('--incGWwavelet', dest='incGWwavelet', action='store_true', \
                    default=False, help='Include GWwavelt signal in run [default=False]')
parser.add_argument('--nGWwavelets', dest='nGWwavelets', action='store', type=int, default=1,
                   help='Number of GW wavelets(default=1)')

parser.add_argument('--incWavelet', dest='incWavelet', action='store_true', \
                    default=False, help='Include noise wavelet signal in run')
parser.add_argument('--nWavelets', dest='nWavelets', action='store', type=int, default=1,
                   help='Number of noise wavelets(default=1)')
parser.add_argument('--waveletModel', dest='waveletModel', action='store', type=str, \
                    default='standard', help='Wavelet model [default=standard]')

parser.add_argument('--incSysWavelet', dest='incSysWavelet', action='store_true', \
                    default=False, help='Include system wavelet signal in run')
parser.add_argument('--nSysWavelets', dest='nSysWavelets', action='store', type=int, default=1,
                   help='Number of system wavelets(default=1)')
parser.add_argument('--sysWaveletModel', dest='sysWaveletModel', action='store', type=str, \
                    default='standard', help='system Wavelet model [default=standard]')

parser.add_argument('--incDMWavelet', dest='incDMWavelet', action='store_true', \
                    default=False, help='Include chromatic noise wavelet signal in run [default=False]')
parser.add_argument('--nDMWavelets', dest='nDMWavelets', action='store', 
                    type=int, default=1, help='Number of chromatic noise wavelets(default=1)')
parser.add_argument('--fixcBeta', dest='fixcBeta', action='store', type=float,
                    default=0, help='fix chromatic wavelet spectral index to user value')

parser.add_argument('--incCW', dest='incCW', action='store_true', \
                    default=False, help='Include CW signal in run [default=False]')
parser.add_argument('--nCW', dest='nCW', action='store', type=int, default=1,
                   help='Number of CW sources (default=1)')
parser.add_argument('--cwModel', dest='cwModel', action='store', type=str, \
                    default='standard', 
                    help='Which CW model to use [standard, upperLimit, mass_ratio, free] [default=standard]')
parser.add_argument('--incPdist', dest='incPdist', action='store_true', \
                    default=False, help='Include pulsar distances for CW signal in run')
parser.add_argument('--CWupperLimit', dest='CWupperLimit', action='store_true', \
                    default=False, help='Calculate CW upper limit (use h not d_L and use uniform prior on h)')
parser.add_argument('--fixf', dest='fixf', action='store', type=float, default=0.0,
                   help='value of GW frequency for upper limits')
parser.add_argument('--cwtheta', dest='cwtheta', action='store', type=float, default=None,
                   help='value of GW theta for CW source')
parser.add_argument('--cwphi', dest='cwphi', action='store', type=float, default=None,
                   help='value of GW phi for CW source')
parser.add_argument('--cwdist', dest='cwdist', action='store', type=float, default=None,
                   help='value of distance for CW source')
parser.add_argument('--cwsnrprior', dest='cwsnrprior', action='store_true', 
                    default=False, help='Use CW snr prior')

parser.add_argument('--incBWM', dest='incBWM', action='store_true', \
                    default=False, help='Include BWM signal in run [default=False]')

parser.add_argument('--BWMmodel', dest='BWMmodel', action='store', type=str,
                    default='gr', help='BWM correlation [default=gr]')

parser.add_argument('--incGP', dest='incGP', action='store_true', \
                    default=False, help='Include GP GW signal in run [default=False]')


parser.add_argument('--incGlitch', dest='incGlitch', action='store_true', \
                    default=False, help='Include Glitch signal in run [default=False]')
parser.add_argument('--incGlitchBand', dest='incGlitchBand', action='store_true', \
                    default=False, help='Include Glitch Band signal in run [default=False]')


parser.add_argument('--niter', dest='niter', action='store', type=int, default=1000000,
                   help='number MCMC iterations (default=1000000)')
parser.add_argument('--thin', dest='thin', action='store', type=int, default=10,
                   help='Thinning factor (default=10)')
parser.add_argument('--compression', dest='compression', action='store', type=str, \
                    default='None', help='compression type [None, frequencies, average, red]')
parser.add_argument('--sampler', dest='sampler', action='store', type=str, \
                    default='mcmc', help='sampler [mcmc, multinest] default=mcmc')
parser.add_argument('--zerologlike', dest='zerologlike', action='store_true', default=False, \
                    help='Zero log likelihood to test prior and jump proposals')
parser.add_argument('--neff', dest='neff', type=int, action='store', \
                    default=1000, help='Number of effective samples [default=1000]')
parser.add_argument('--resume', dest='resume', action='store_true', \
                    default=False, help='resume from previous run?')
parser.add_argument('--writeHotChains', dest='writeHotChains', action='store_true', \
                    default=False, help='Write hot chains in MCMC sampler?')
parser.add_argument('--hotChain', dest='hotChain', action='store_true', \
                    default=False, help='Sample from prior for hottest chain?')

parser.add_argument('--mark6', dest='mark6', action='store_true', \
                    default=False, help='Use T matrix formalism')
parser.add_argument('--Tmatrix', dest='Tmatrix', action='store_true', \
                    default=False, help='Use T matrix formalism')
parser.add_argument('--mark9', dest='mark9', action='store_true', \
                    default=False, help='Use mark9 likelihood')
parser.add_argument('--mark10', dest='mark10', action='store_true', \
                    default=False, help='Use mark10 likelihood')
parser.add_argument('--mark11', dest='mark11', action='store_true', \
                    default=False, help='Use mark11 likelihood')

parser.add_argument('--Tmin', dest='Tmin', type=float, action='store', \
                     default=1, help='Minimum temperature for parallel tempering')
parser.add_argument('--Tmax', dest='Tmax', type=float, action='store', \
                     default=1, help='Max temperature for parallel tempering.')

# parse arguments
args = parser.parse_args()

##### Begin Code #####

# MPI initialization
comm = MPI.COMM_WORLD
MPIrank = comm.Get_rank()
MPIsize = comm.Get_size()

if args.Tmax == 1:
    args.Tmax = None

if not os.path.exists(args.outDir):
    try:
        os.makedirs(args.outDir)
    except OSError:
        pass

# open file
incGWB = args.incGWB
if args.pname[0] != 'all':
    model = PALmodels.PTAmodels(args.h5file, pulsars=args.pname)
elif args.pname[0] == 'all':
    print 'Using all pulsars'
    pulsars = 'all'
    model = PALmodels.PTAmodels(args.h5file, pulsars=pulsars)


# model options
separateEfacs = args.separateEfacs == 'backend'
separateEfacsByFreq = args.separateEfacs == 'frequencies'

separateEquads = args.separateEquads == 'backend'
separateEquadsByFreq = args.separateEquads == 'frequencies'

separateJitterEquad = args.separateJitterEquad == 'backend'
separateJitterEquadByFreq = args.separateJitterEquad == 'frequencies'

# no marginalization setting
incRedFourierMode, incDMFourierMode, incGWFourierMode = False, False, False
if args.noMarg:
    if args.incRed:
        incRedFourierMode = True
    if args.incGWB:
        incGWFourierMode = True
    if args.incDM:
        incDMFourierMode = True

if args.incJitterEquad:
    likfunc = 'mark2'
elif args.incTimingModel and args.fullmodel and not args.incNonGaussian:
    likfunc = 'mark4'
elif args.incTimingModel and args.fullmodel and args.incNonGaussian:
    likfunc='mark5'
else:
    likfunc = 'mark1'

if args.noMarg:
    likfunc = 'mark7'

if args.mark6 or args.margShapelet or args.Tmatrix:
    likfunc = 'mark6'
if args.mark9:
    likfunc = 'mark9'
if args.mark10:
    likfunc = 'mark10'
if args.mark11:
    likfunc = 'mark11'

if args.margShapelet:
    dmEventModel = 'shapeletmarg'
else:
    dmEventModel = 'shapelet'

if args.dmxKernelModel != 'None':
    incDMXKernel = True
    DMXKernelModel = args.dmxKernelModel
else:
    incDMXKernel = False
    DMXKernelModel = args.dmxKernelModel

if args.jsonfile is None:

    fullmodel = model.makeModelDict(
        incRedNoise=True, noiseModel=args.redModel, 
        incRedBand=args.incRedBand, 
        incDMBand=args.incDMBand, 
        incDM=args.incDM, dmModel=args.dmModel, 
        incDMEvent=args.incDMshapelet, dmEventModel=dmEventModel, 
        ndmEventCoeffs=args.nshape, 
        incEphemError=args.incEphemError,
        incDMX=args.incDMX, 
        incORF=args.incORF, 
        incBWM=args.incBWM, BWMmodel=args.BWMmodel,
        incSingleGWGP=args.incGP,
        incGlitch=args.incGlitch, incGlitchBand=args.incGlitchBand,
        incGWWavelet=args.incGWwavelet, nGWWavelets=args.nGWwavelets,
        incWavelet=args.incWavelet, nWavelets=args.nWavelets,
        waveletModel=args.waveletModel,
        incSysWavelet=args.incSysWavelet, nSysWavelets=args.nSysWavelets,
        sysWaveletModel=args.sysWaveletModel,
        incDMWavelet=args.incDMWavelet, 
        nDMWavelets=args.nDMWavelets,
        incGWBAni=args.incGWBAni, lmax=args.lmax,
        clmPrior=args.clmPrior,
        incDMXKernel=incDMXKernel, DMXKernelModel=DMXKernelModel, 
        separateEfacs=separateEfacs, separateEfacsByFreq=separateEfacsByFreq, 
        separateEquads=separateEquads, separateEquadsByFreq=separateEquadsByFreq, 
        separateJitterEquad=separateJitterEquad, 
        separateJitterEquadByFreq=separateJitterEquadByFreq, 
        incRedFourierMode=incRedFourierMode, incDMFourierMode=incDMFourierMode, 
        incScattering=args.incScat, scatteringModel=args.scatModel, nscatfreqs=args.nscatf,
        incGWFourierMode=incGWFourierMode, 
        incEquad=args.incEquad,
        incTimingModel=args.incTimingModel, nonLinear=args.tmmodel=='nonlinear', 
        addPars=args.addpars, subPars=args.delpars, 
        add_all_timing_pars=args.addAllPars,
        fulltimingmodel=args.fullmodel, incNonGaussian=args.incNonGaussian, 
        nnongaussian=args.nnongauss, 
        incRedExt=args.incRedExt, redExtModel=args.redExtModel, 
        redExtNf=args.nfext, 
        incEnvelope=args.incRedEnv, envelopeModel=args.redEnvModel,
        incCW=args.incCW, incPulsarDistance=args.incPdist, 
        cwsnrprior=args.cwsnrprior,
        CWModel=args.cwModel, nCW=args.nCW, 
        CWupperLimit=args.CWupperLimit, 
        incJitterEquad=args.incJitterEquad, 
        redAmpPrior=args.redAmpPrior, GWAmpPrior=args.GWBAmpPrior, 
        redSpectrumPrior=args.redAmpPrior, GWspectrumPrior=args.GWBAmpPrior, 
        incSingleFreqNoise=args.incSingleRed, numSingleFreqLines=1, 
        incSingleFreqDMNoise=args.incSingleDM, numSingleFreqDMLines=1, 
        DMAmpPrior=args.DMAmpPrior, 
        incGWB=incGWB, nfreqs=args.nfreqs, ndmfreqs=args.ndmfreqs, 
        incGWBSingle=args.incGWBSingle, gwbSingleModel=args.gwbSingleModel,
        gwbModel=args.gwbModel, 
        Tmax = args.Tspan,
        compression=args.compression, 
        likfunc=likfunc)


    # fix spectral index
    if args.fixSi:
        print 'Fixing GWB spectral index to {0}'.format(args.fixSi)
        for sig in fullmodel['signals']:
            if sig['corr'] == 'gr' and sig['stype'] in ['powerlaw', 'turnover']:
                sig['bvary'][1] = False
                sig['pstart'][1] = args.fixSi
            elif sig['corr'] == 'gr_sph' and sig['stype'] == 'powerlaw':
                sig['bvary'][1] = False
                sig['pstart'][1] = args.fixSi

    # fix spectral index
    if args.fixKappa:
        print 'Fixing GWB kappa to {0}'.format(args.fixKappa)
        for sig in fullmodel['signals']:
            if sig['corr'] == 'gr' and sig['stype'] in ['turnover']:
                sig['bvary'][3] = False
                sig['pstart'][3] = args.fixKappa

    # fix spectral indexa of chromatic wavelet
    if args.fixcBeta:
        print 'Fixing chromatic wavelet index to {0}'.format(args.fixcBeta)
        for sig in fullmodel['signals']:
            if sig['stype'] in ['chrowavelet']:
                sig['bvary'][0] = False
                sig['pstart'][0] = args.fixcBeta

    # fix spectral index for red nosie
    if args.fixRedSi:
        print 'Fixing red noise spectral index to 4.33'
        for sig in fullmodel['signals']:
            if sig['corr'] == 'single' and sig['stype'] == 'powerlaw':
                sig['bvary'][1] = False
                sig['pstart'][1] = 4.33

    if not(args.incRed):
        print 'Warning: Not varying red noise' 
        for sig in fullmodel['signals']:
            if sig['corr'] == 'single' and sig['stype'] == 'powerlaw':
                sig['bvary'][1] = False
                sig['bvary'][0] = False
                sig['pstart'][0] = -20


    if args.fixf != 0.0:
        print 'Warning: Fixing CW frequency to {0}'.format(args.fixf) 
        for sig in fullmodel['signals']:
            if sig['stype'] == 'cw':
                sig['bvary'][4] = False
                sig['pstart'][4] = np.log10(args.fixf)

    if args.cwtheta is not None:
        print 'Warning: Fixing CW theta to {0}'.format(args.cwtheta) 
        for sig in fullmodel['signals']:
            if sig['stype'] == 'cw':
                sig['bvary'][0] = False
                sig['pstart'][0] = args.cwtheta

    if args.cwphi is not None:
        print 'Warning: Fixing CW phi to {0}'.format(args.cwphi) 
        for sig in fullmodel['signals']:
            if sig['stype'] == 'cw':
                sig['bvary'][1] = False
                sig['pstart'][1] = args.cwphi

    if args.cwdist is not None:
        print 'Warning: Fixing dist to {0} Mpc'.format(args.cwdist) 
        for sig in fullmodel['signals']:
            if sig['stype'] == 'cw':
                sig['bvary'][3] = False
                sig['pstart'][3] = np.log10(args.cwdist)

    memsave = True
    if args.noVaryEfac:
        print 'Not Varying EFAC'
        for sig in fullmodel['signals']:
            if sig['corr'] == 'single' and sig['stype'] == 'efac':
                sig['bvary'][0] = False
                sig['pstart'][0] = 1

    if args.noVaryNoise:
        print 'Fixing Noise values to defaults'
        nflags = ['efac', 'equad', 'jitter', 'jitter_equad']
        for sig in fullmodel['signals']:
            if sig['stype'] in nflags:
                sig['bvary'][0] = False
                print '{0} for {1} set to {2}'.format(sig['stype'],
                                                     sig['flagvalue'],
                                                     sig['pstart'])
    
    #for sig in fullmodel['signals']:
    #    if sig['stype'] == 'jitter_equad':
    #        if sig['flagvalue'] == 'J1741+1351-430_ASP':
    #            sig['bvary'][0] = False


    if args.fixNoise:
        noisedir = args.noisedir
        for ct, p in enumerate(model.psr):
            d = np.genfromtxt(noisedir + p.name + '_noise.txt', dtype='S42')
            pars = d[:,0]
            vals = np.array([float(d[ii,1]) for ii in range(d.shape[0])])
            sigs = [psig for psig in fullmodel['signals'] if psig['pulsarind'] == ct]
            sigs = PALutils.fixNoiseValues(sigs, vals, pars, bvary=False, verbose=True)

    # turn red noise back on
    if args.fixNoise and args.fixWhite:
        print 'Turning on red noise and only fixing white noise'
        for sig in fullmodel['signals']:
            if sig['corr'] == 'single' and sig['stype'] == 'powerlaw':
                sig['bvary'][1] = True
                sig['bvary'][0] = True
    
    # write JSON file
    if not args.incTimingModel:
        model.writeModelToFile(fullmodel, args.outDir + '/model.json')


# check for single efacs
if args.incCW or args.incTimingModel or args.incSingleRed or args.incSingleDM:
    for p in model.psr:
        numEfacs = model.getNumberOfSignalsFromDict(fullmodel['signals'], \
                stype='efac', corr='single')
        memsave = np.any(numEfacs > 1)

if args.compression != 'None' and args.separateEfacs == 'frequencies':
    memsave = False

# initialize model
if args.fromFile:
    write = True
else:
    write = 'no'

if args.jsonfile is None:
    model.initModel(fullmodel, memsave=memsave, fromFile=args.fromFile, 
                    verbose=True, write=write)

else:
    print 'Initializing Model from JSON file {0}\n'.format(args.jsonfile)
    model.initModelFromFile(args.jsonfile, memsave=True, fromFile=args.fromFile, 
                    verbose=True, write=write)
    

pardes = model.getModelParameterList()
par_names = [p['id'] for p in pardes if p['index'] != -1]
par_out = []
for pname in par_names:
    if args.pname[0] in pname and len(args.pname) == 1:
        par_out.append(''.join(pname.split('_'+args.pname[0])))
    else:
        par_out.append(pname)

print '{0} Free parameters'.format(len(par_out))
print 'Search Parameters: {0}'.format(par_out)        
    
# output parameter names
fout = open(args.outDir + '/pars.txt', 'w')
for nn in par_out:
    fout.write('%s\n'%(nn))
fout.close()

# define likelihood functions
if args.sampler == 'mcmc' or args.sampler == 'minimize' or args.sampler=='multinest' \
   or args.sampler=='polychord':
    if args.incJitterEquad:
        loglike = model.mark2LogLikelihood
    elif args.incTimingModel and args.fullmodel and not args.incNonGaussian:
        loglike = model.mark4LogLikelihood
    elif args.incTimingModel and args.fullmodel and args.incNonGaussian:
        loglike = model.mark5LogLikelihood
    else:
        loglike = model.mark1LogLikelihood

    if args.mark6 or args.margShapelet or args.Tmatrix:
        loglike = model.mark6LogLikelihood
    if args.noMarg:
        loglike = model.mark7LogLikelihood
    if args.mark9:
        loglike = model.mark9LogLikelihood
    if args.mark10:
        loglike = model.mark10loglikelihood
    if args.mark11:
        loglike = model.mark11LogLikelihood
                                

    # if zero log-likeihood
    if args.zerologlike:
        print 'WARNING: Using zero log-like'
        loglike = model.zeroLogLikelihood

    # log prior
    if args.pname == 'all':
        logprior = model.mark3LogPrior
    else:
        logprior = model.mark3LogPrior
    
    # log likelihood arguments
    loglkwargs = {}
    if args.noCorrelations or not(np.any([
        args.incGWB, args.incGWBAni, args.incGP, args.incGWBSingle])):
        print 'Running model with no GWB correlations'
        loglkwargs['incCorrelations'] = False
    else:
        loglkwargs['incCorrelations'] = True
        print 'Running model with GWB correlations'
    if args.incJitterEquad and np.any([args.mark6, args.Tmatrix, args.mark10]):
        loglkwargs['incJitter'] = True

   
    # get initial parameters for MCMC
    inRange = False
    pstart = False
    fixpstart = False
    if args.incTimingModel:#or args.fixNoise or args.noVaryNoise:
        fixpstart=True
    if MPIrank == 0:
        pstart = True
    startSpectrumMin = False
    while not(inRange):
        p0 = model.initParameters(startEfacAtOne=True, fixpstart=fixpstart)
        startSpectrumMin = True
        print loglike(p0, **loglkwargs), logprior(p0)
        if logprior(p0) != -np.inf and loglike(p0, **loglkwargs) != -np.inf:
            inRange = True
    
    # add extra kwargs if fixed noise 
    if np.any([args.fixNoise, args.noVaryNoise]) and not \
       args.fixWhite and np.any([args.mark9, args.mark6]):
        loglkwargs['varyNoise'] = False
    elif np.any([args.fixNoise, args.noVaryNoise]) and args.fixWhite and \
            np.any([args.mark9, args.mark6]):
        loglkwargs['varyNoise'] = True
        loglkwargs['fixWhite'] = True
    
    if args.zerologlike:
        loglkwargs = {}
    
    # if fixed noise, must call likelihood once to initialize matrices
    #if args.fixNoise or args.fixWhite or args.noVaryNoise:
    #    if not args.zerologlike:
    #        loglike(p0, incCorrelations=False)

    if args.sampler == 'minimize':
        import pyswarm

        # define function
        def fun(x):
            if logprior(x) != -np.inf:
                ret = -loglike(x, **loglkwargs)
            else:
                ret = 1e10

            return ret

        maxpars, maxf = pyswarm.pso(fun, model.pmin, model.pmax, swarmsize=1000, \
                        omega=0.5, phip=0.5, phig=0.5, maxiter=1000, debug=True, \
                        minfunc=1e-3)

        np.savetxt(args.outDir + '/pso_maxpars.txt', maxpars)


    if args.sampler == 'mcmc':

        cov = model.initJumpCovariance()

        ind = []

        ##### white noise #####
        if not args.noVaryEfac and not args.noVaryNoise:
            ids = model.get_parameter_indices('efac', corr='single', split=True)
            [ind.append(id) for id in ids if len(id) > 0]

        if args.incEquad:
            ids = model.get_parameter_indices('equad', corr='single', split=True)
            [ind.append(id) for id in ids if len(id) > 0]
        
        if args.incJitterEquad:
            ids = model.get_parameter_indices('jitter_equad', corr='single', split=False)
            [ind.append(id) for id in ids if len(id) > 0]
        
        
        ##### red noise #####
        if args.incRed:
            if args.redModel == 'powerlaw':
                ids = model.get_parameter_indices('powerlaw', corr='single', split=True)
                [ind.append(id) for id in ids if len(id) > 0]
            if args.redModel == 'spectrum':
                ids = model.get_parameter_indices('spectrum', corr='single', split=True)
                [ind.append(id) for id in ids if len(id) > 0]
            if args.redModel == 'interpolate':
                ids = model.get_parameter_indices('interpolate', corr='single', split=False)
                [ind.append(id) for id in ids if len(id) > 0]
        
        ##### red band noise #####
        if args.incRedBand:
            ids = model.get_parameter_indices('powerlaw_band', corr='single', split=True)
            [ind.append(id) for id in ids]
        
        ##### DM noise #####
        if args.incDM:
            if args.dmModel == 'powerlaw':
                ids = model.get_parameter_indices('dmpowerlaw', corr='single', split=True)
                [ind.append(id) for id in ids]
            if args.dmModel == 'spectrum':
                ids = model.get_parameter_indices('dmspectrum', corr='single', split=False)
                [ind.append(id) for id in ids]

        ##### Scattering noise #####
        if args.incScat:
            if args.scatModel == 'powerlaw':
                ids = model.get_parameter_indices('scatpowerlaw', corr='single', split=True)
                [ind.append(id) for id in ids]
            if args.scatModel == 'spectrum':
                ids = model.get_parameter_indices('scatspectrum', corr='single', split=False)
                [ind.append(id) for id in ids]

        ##### wavelets #####
        if args.incWavelet:
            ids = model.get_parameter_indices('wavelet', corr='single', split=True)
            [ind.append(id) for id in ids if id != []]
        
        if args.incDMWavelet:
            ids = model.get_parameter_indices('dmwavelet', corr='single', split=True)
            [ind.append(id) for id in ids]
        
        if args.incSysWavelet:
            ids = model.get_parameter_indices('syswavelet', corr='single', split=True)
            [ind.append(id) for id in ids]
        
        if args.incGWwavelet:
            ids = model.get_parameter_indices('gwwavelet', corr='gr', split=True)
            [ind.append(id) for id in ids]
        
        ##### GWB #####
        if args.incGWB:
            if args.gwbModel == 'powerlaw':
                ids = model.get_parameter_indices('powerlaw', corr='gr', split=False)
                [ind.append(id) for id in ids]
            if args.gwbModel == 'spectrum':
                ids = model.get_parameter_indices('spectrum', corr='gr', split=False)
                [ind.append(id) for id in ids]
            if args.gwbModel == 'turnover':
                ids = model.get_parameter_indices('turnover', corr='gr', split=False)
                [ind.append(id) for id in ids]

        ##### Ephemeris Error #####
        if args.incEphemError:
            ids = model.get_parameter_indices('ephemeris', corr='single', split=False)
            [ind.append(id) for id in ids]
        
        ##### GWB Point Source #####
        if args.incGWBSingle:
            if args.gwbSingleModel == 'powerlaw':
                ids = model.get_parameter_indices('powerlaw', corr='grs', split=False)
                [ind.append(id) for id in ids]
            if args.gwbSingleModel == 'spectrum':
                ids = model.get_parameter_indices('spectrum', corr='grs', split=False)
                [ind.append(id) for id in ids]
        
        ##### Anisotropic GWB #####
        if args.incGWBAni:
            if args.gwbModel == 'powerlaw':
                ids = model.get_parameter_indices('powerlaw', corr='gr_sph', split=False)
                [ind.append(id) for id in ids]
            if args.gwbModel == 'spectrum':
                ids = model.get_parameter_indices('spectrum', corr='gr_sph', split=False)
                [ind.append(id) for id in ids]
        
        ##### Glitch #####
        if args.incGlitch:
            ids = model.get_parameter_indices('glitch', corr='single', split=True)
            [ind.append(id) for id in ids]

        ##### Band Glitch #####
        if args.incGlitchBand:
            ids = model.get_parameter_indices('glitch_band', corr='single', split=True)
            [ind.append(id) for id in ids]

        ##### BWM #####
        if args.incBWM:
            ids = model.get_parameter_indices('bwm', corr=args.BWMmodel, split=True)
            [ind.append(id) for id in ids]
        
        ##### GP #####
        if args.incGP:
            ids = model.get_parameter_indices('gw-gp', corr='gr', split=True)
            [ind.append(id) for id in ids]
        
        ##### CW #####
        if args.incCW:
            ids = model.get_parameter_indices('cw', corr='gr', split=True)
            [ind.append(id) for id in ids]
            for id in ids:
                for idd in id:
                    ind.append([idd])
            
            # pulsar distances
            if args.incPdist:
                ids = model.get_parameter_indices('pulsardistance', corr='single', split=False)
                [ind.append(id) for id in ids]

            # pulsar phase terms
            if args.cwModel in ['free', 'freephase', 'eccgam', 'upperLimit_phase']:
                ids = model.get_parameter_indices('pulsarTerm', corr='single', split=True)
                [ind.append(id) for id in ids]

        # timing model
        if args.incTimingModel:
            if args.tmmodel == 'nonlinear':
                ids = model.get_parameter_indices('nonlineartimingmodel',
                                              corr='single', split=False)
            else:
                ids = model.get_parameter_indices('lineartimingmodel',
                                              corr='single', split=False)

            [ind.append(id) for id in ids]



        ##### all parameters #####
        ind.insert(0, range(len(p0)))
        print ind

        # define MCMC sampler
        sampler = PALInferencePTMCMC.PTSampler(len(p0), loglike, logprior, cov, comm=comm, \
                                               outDir=args.outDir, loglkwargs=loglkwargs, \
                                               resume=args.resume, groups=ind)

        # add jump proposals
        if incGWB:
            if args.gwbModel == 'powerlaw':
                sampler.addProposalToCycle(model.drawFromGWBPrior, 10)
            elif args.gwbModel == 'spectrum':
                sampler.addProposalToCycle(model.drawFromGWBSpectrumPrior, 10)
            elif args.gwbModel == 'turnover':
                sampler.addProposalToCycle(model.drawFromGWBTurnoverPrior, 10)
        if args.incGWBAni and args.gwbModel == 'powerlaw':
                sampler.addProposalToCycle(model.drawFromaGWBPrior, 10)
        if args.incGWBSingle and args.gwbSingleModel == 'spectrum':
                sampler.addProposalToCycle(model.drawFromsGWBPrior, 10)
        if args.incRed and args.redModel=='powerlaw':
            sampler.addProposalToCycle(model.drawFromRedNoisePrior, 5)
        if args.incRedBand and args.redModel=='powerlaw':
            sampler.addProposalToCycle(model.drawFromRedNoiseBandPrior, 5)
        if args.incDMBand and args.dmModel=='powerlaw':
            sampler.addProposalToCycle(model.drawFromDMNoiseBandPrior, 5)
        if args.incORF:
            sampler.addProposalToCycle(model.drawFromORFPrior, 10)
        if args.incDM and args.dmModel=='powerlaw':
            sampler.addProposalToCycle(model.drawFromDMPrior, 5)
        if args.incRed and args.redModel=='spectrum':
            sampler.addProposalToCycle(model.drawFromRedNoiseSpectrumPrior, 10)
        if args.incRedExt and args.redExtModel=='spectrum':
            sampler.addProposalToCycle(model.drawFromRedNoiseExtSpectrumPrior, 10)
        if args.incEquad and not args.fixWhite:
            sampler.addProposalToCycle(model.drawFromEquadPrior, 5)
        #if args.incJitterEquad:
        #    sampler.addProposalToCycle(model.drawFromJitterEquadPrior, 5)
        if args.incTimingModel:
            sampler.addProposalToCycle(model.drawFromTMfisherMatrix, 40)
            #sampler.addProposalToCycle(model.drawFromTMPrior, 5)
        #if args.incBWM:
        #    sampler.addProposalToCycle(model.drawFromBWMPrior, 10)

        if args.incCW:
            if args.cwModel in ['upperLimit', 'upperLimit_phase']:
                wgt = 30
            else:
                wgt = 5
            sampler.addProposalToCycle(model.drawFromCWPrior, wgt)
            #sampler.addProposalToCycle(model.massDistanceJump, 2)
            sampler.addProposalToCycle(model.phaseAndPolarizationReverseJump, 5)
            sampler.addAuxilaryJump(model.fix_cyclic_pars)
            if args.cwModel in ['ecc', 'eccgam']:
                sampler.addProposalToCycle(model.gammaAndPolarizationReverseJump, 5)
            if args.cwModel in ['free', 'freephase', 'ecc']:
                sampler.addProposalToCycle(model.pulsarPhaseJump, 5)
            if args.incPdist:
                sampler.addProposalToCycle(model.pulsarDistanceJump, 10)
                if args.cwModel not in ['freephase', 'free', 'ecc', 'upperLimit_phase']:
                    print 'Adding auxiliary pulsar phase jump'
                    sampler.addAuxilaryJump(model.pulsarPhaseFix)
                elif args.cwModel == 'ecc':
                    sampler.addAuxilaryJump(model.pulsarGammaFix)

        # always include draws from efac
        if not args.noVaryEfac and not args.noVaryNoise and not args.fixWhite:
            sampler.addProposalToCycle(model.drawFromEfacPrior, 2)


        # run MCMC
        print 'Engage!'
        sampler.sample(p0, args.niter, covUpdate=1000, AMweight=15, SCAMweight=30, \
                       DEweight=50, neff=args.neff, KDEweight=0, Tmin=args.Tmin,
                       Tmax=args.Tmax, writeHotChains=args.writeHotChains,
                      hotChain=args.hotChain, thin=args.thin)

    if args.sampler == 'polychord':
        print 'Using PolyChord Sampler'
        import pypolychord

        prior_array = np.append(model.pmin, model.pmax)

        def chord_like(ndim, theta, phi):
            
            # check prior
            if model.mark3LogPrior(theta) != -np.inf:
                return loglike(theta, **loglkwargs)
            else:
                #print 'WARNING: Prior returns -np.inf!!'
                return -np.inf

        n_live = 1000
        ndim = len(p0)

        pypolychord.run(chord_like, ndim, prior_array, n_live=n_live, n_chords=5, \
                       output_basename=args.outDir+"/pcord-")



    if args.sampler == 'multinest':
        print 'WARNING: Using MultiNest, will use uniform priors on all parameters' 
        import pymultinest

        p0 = model.initParameters(startEfacAtOne=True, fixpstart=False)

        # mark2 loglike
        if args.incJitterEquad:

            ndim = len(p0)

            def myloglike(cube, ndim, nparams):
                
                acube = np.zeros(ndim)
                for ii in range(ndim):
                    acube[ii] = cube[ii]

                # check prior
                if model.mark3LogPrior(acube) != -np.inf:
                    return loglike(acube, **loglkwargs) #+ model.mark3LogPrior(acube)
                else:
                    #print 'WARNING: Prior returns -np.inf!!'
                    return -np.inf

            def myprior(cube, ndim, nparams):

                for ii in range(ndim):
                    cube[ii] = model.pmin[ii] + cube[ii] * (model.pmax[ii]-model.pmin[ii])
        
        
        # mark1 loglike
        else:

            ndim = len(p0)

            def myloglike(cube, ndim, nparams):
                
                acube = np.zeros(ndim)
                for ii in range(ndim):
                    acube[ii] = cube[ii]
                
                # check prior
                if model.mark3LogPrior(acube) != -np.inf:
                    ll = loglike(acube, **loglkwargs) #+ model.mark3LogPrior(acube)
                    return ll
                else:
                    #print 'WARNING: Prior returns -np.inf!!'
                    return -np.inf

                #return model.mark1LogLikelihood(acube)

            def myprior(cube, ndim, nparams):

                for ii in range(ndim):
                    if args.GWBAmpPrior == 'sesana' and par_out[ii] == 'GWB-Amplitude':
                        m = -15
                        s = 0.22
                        cube[ii] = m + s*np.sqrt(2) * ss.erfcinv(2*(1-cube[ii]))
                    if args.GWBAmpPrior == 'mcwilliams' and par_out[ii] == 'GWB-Amplitude':
                        m = np.log10(4.1e-15)
                        s = 0.26
                        cube[ii] = m + s*np.sqrt(2) * ss.erfcinv(2*(1-cube[ii]))
                    else:
                        cube[ii] = model.pmin[ii] + cube[ii] * (model.pmax[ii]-model.pmin[ii])

                    # number of live points
        nlive = 2000
        n_params = ndim

        # run MultiNest
        pymultinest.run(myloglike, myprior, n_params, resume = args.resume, \
                        verbose = True, sampling_efficiency = 0.3, \
                        outputfiles_basename =  args.outDir+'/mn'+'-', \
                        n_iter_before_update=5, n_live_points=nlive, \
                        const_efficiency_mode=False, importance_nested_sampling=False, \
                        n_clustering_params=n_params, init_MPI=False)


