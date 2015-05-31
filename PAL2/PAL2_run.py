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
    import nompi4py as MPI


import argparse

parser = argparse.ArgumentParser(description = 'Run PAL2 Data analysis pipeline')

# options
parser.add_argument('--h5File', dest='h5file', action='store', type=str, required=True,
                   help='Full path to hdf5 file containing PTA data')
parser.add_argument('--fromFile', dest='fromFile', action='store_true', \
                    default=False, help='Init model from file')

parser.add_argument('--outDir', dest='outDir', action='store', type=str, default='./',
                   help='Full path to output directory (default = ./)')

parser.add_argument('--pulsar', dest='pname', action='store', nargs='+', \
                    type=str, required=True, help='names of pulsars to use')

parser.add_argument('--incRed', dest='incRed', action='store_true',default=False,
                   help='include Red Noise')
parser.add_argument('--redModel', dest='redModel', action='store', type=str, default='powerlaw',
                   help='red noise model [powerlaw, spectrum]')
parser.add_argument('--nf', dest='nfreqs', action='store', type=int, default=10,
                   help='number of red noise frequencies to use (default=10)')
parser.add_argument('--fixRedSi', dest='fixRedSi', action='store_true', default=False, \
                    help='Fix red noise spectral index to 4.33')
parser.add_argument('--redAmpPrior', dest='redAmpPrior', action='store', type=str, \
                    default='log', help='prior on red noise Amplitude [uniform, log]')
parser.add_argument('--logfrequencies', dest='logfrequencies', action='store_true', \
                    default=False, help='Use log sampling in frequencies.')
parser.add_argument('--incSingleRed', dest='incSingleRed', action='store_true',default=False,
                   help='include single frequency red noise')
parser.add_argument('--incRedBand', dest='incRedBand', action='store_true',default=False,
                   help='include band limited red noise')
parser.add_argument('--incRedEnv', dest='incRedEnv', action='store_true',default=False,
                   help='include Red Noise Envelope model')
parser.add_argument('--redEnvModel', dest='redEnvModel', action='store', 
                    type=str, default='powerlaw',
                    help='red noise envelope model [powerlaw, spectrum]')
parser.add_argument('--incRedExt', dest='incRedExt', action='store_true',default=False,
                   help='include Red Noise Extended model')
parser.add_argument('--redExtModel', dest='redExtModel', action='store', 
                    type=str, default='powerlaw',
                    help='red noise extended model [powerlaw, spectrum]')
parser.add_argument('--nfext', dest='nfext', action='store', type=int, default=10,
                   help='number of red noise frequencies to use in extended model(default=10)')

parser.add_argument('--incScat', dest='incScat', action='store_true',default=False,
                   help='include stochastic scattering process')
parser.add_argument('--scatModel', dest='scatModel', action='store', 
                    type=str, default='powerlaw',
                    help='scattering model [powerlaw, spectrum]')
parser.add_argument('--nfscat', dest='nfscat', action='store', type=int, default=10,
                   help='number of frequencies for scattering model(default=10)')


parser.add_argument('--Tspan', dest='Tspan', action='store', type=float, default=None,
                   help='Tmax to use in red noise and GW expansion (default=None)')


parser.add_argument('--incDM', dest='incDM', action='store_true',default=False,
                   help='include DM variations')
parser.add_argument('--dmModel', dest='dmModel', action='store', type=str, default='powerlaw',
                   help='red DM noise model [powerlaw, spectrum]')
parser.add_argument('--ndmf', dest='ndmfreqs', action='store', type=int, default=10,
                   help='number of DM noise frequencies to use (default=10)')
parser.add_argument('--DMAmpPrior', dest='DMAmpPrior', action='store', type=str, \
                    default='log', help='prior on DM Amplitude [uniform, log]')
parser.add_argument('--incSingleDM', dest='incSingleDM', action='store_true',default=False,
                   help='include single frequency DM')
parser.add_argument('--incDMshapelet', dest='incDMshapelet', action='store_true',default=False,
                   help='include shapelet event for DM')
parser.add_argument('--nshape', dest='nshape', type=int, action='store', default=3, \
                   help='number of coefficients for shapelet event for DM')
parser.add_argument('--margShapelet', dest='margShapelet', action='store_true', default=False, \
                   help='Analytically marginalize over shapelet coefficients')
parser.add_argument('--incDMX', dest='incDMX', action='store_true', default=False, \
                   help='include DMX')
parser.add_argument('--dmxKernelModel', dest='dmxKernelModel', action='store', type=str, \
                    default='None',
                   help='dmx Kernel Model')
parser.add_argument('--incDMBand', dest='incDMBand', action='store_true',default=False,
                   help='include band limited DM noise')

parser.add_argument('--incGWB', dest='incGWB', action='store_true',default=False,
                   help='include GWB')
parser.add_argument('--gwbModel', dest='gwbModel', action='store', type=str, default='powerlaw',
                   help='GWB model [powerlaw, spectrum]')
parser.add_argument('--fixSi', dest='fixSi', action='store_true', default=False, \
                    help='Fix GWB spectral index to 4.33')
parser.add_argument('--noCorrelations', dest='noCorrelations', action='store_true', \
                    default=False, help='Do not in include GWB correlations')
parser.add_argument('--GWBAmpPrior', dest='GWBAmpPrior', action='store', type=str, \
                    default='log', help='prior on GWB Amplitude [uniform, log]')
parser.add_argument('--incORF', dest='incORF', action='store_true', \
                    default=False, help='Include generic ORF')

parser.add_argument('--incGWBAni', dest='incGWBAni', action='store_true',default=False,
                   help='include GWB')
parser.add_argument('--nls', dest='nls', action='store', type=int, default=2,
                   help='number of ls to use in anisitropic search')

parser.add_argument('--incEquad', dest='incEquad', action='store_true',default=False,
                   help='include Equad')
parser.add_argument('--separateEquads', dest='separateEquads', action='store', type=str, \
                    default='frequencies', help='separate equads [None, backend, frequencies]')

parser.add_argument('--noVaryEfac', dest='noVaryEfac', action='store_true',default=False,
                   help='Option to not vary efac')
parser.add_argument('--separateEfacs', dest='separateEfacs', action='store', type=str, \
                    default='frequencies', help='separate efacs [None, backend, frequencies]')

parser.add_argument('--incJitter', dest='incJitter', action='store_true',default=False,
                   help='include Jitter')
parser.add_argument('--separateJitter', dest='separateJitter', action='store', type=str, \
                    default='frequencies', help='separate Jitter [None, backend, frequencies]')

parser.add_argument('--incJitterEquad', dest='incJitterEquad', action='store_true',\
                    default=False, help='include Jitter')
parser.add_argument('--separateJitterEquad', dest='separateJitterEquad', action='store', \
                    type=str, default='frequencies', \
                    help='separate Jitter Equad [None, backend, frequencies]')
parser.add_argument('--fixNoise', dest='fixNoise', action='store_true',\
                    default=False, help='fix Noise values')
parser.add_argument('--noVaryNoise', dest='noVaryNoise', action='store_true',\
                    default=False, help='fix Noise values to default values')
parser.add_argument('--fixWhite', dest='fixWhite', action='store_true',\
                    default=False, help='fix White Noise values')
parser.add_argument('--noisedir', dest='noisedir', action='store', type=str,
                   help='Full path to directory containting maximum likeihood noise values')


parser.add_argument('--incJitterEpoch', dest='incJitterEpoch', action='store_true',\
                    default=False, help='include Jitter by epoch')

parser.add_argument('--incTimingModel', dest='incTimingModel', action='store_true', \
                    default=False, help='Include timing model parameters in run')
parser.add_argument('--tmmodel', dest='tmmodel', action='store', type=str, \
                    default='linear', \
                    help='linear or non-linear timing model [linear, nonlinear]')
parser.add_argument('--fullmodel', dest='fullmodel', action='store_true', \
                    default=False, \
                    help='Use full timing model, no marginalization')
parser.add_argument('--noMarg', dest='noMarg', action='store_true',default=False,
                   help='No analytic marginalization')
parser.add_argument('--addpars', dest='addpars', action='store', nargs='+', \
                    type=str, required=False, help='Extra parameters to add to timing model')
parser.add_argument('--delpars', dest='delpars', action='store', nargs='+', \
                    type=str, required=False, help='parameters to remove from timing model')

parser.add_argument('--incNonGaussian', dest='incNonGaussian', action='store_true', \
                    default=False, \
                    help='Use non-gaussian likelihood function')
parser.add_argument('--nnongauss', dest='nnongauss', action='store', type=int, default=3,
                   help='number of non-guassian components (default=3)')

parser.add_argument('--incGWwavelet', dest='incGWwavelet', action='store_true', \
                    default=False, help='Include GWwavelt signal in run')
parser.add_argument('--nGWwavelets', dest='nGWwavelets', action='store', type=int, default=1,
                   help='Number of GW wavelets(default=1)')

parser.add_argument('--incCW', dest='incCW', action='store_true', \
                    default=False, help='Include CW signal in run')
parser.add_argument('--cwModel', dest='cwModel', action='store', type=str, \
                    default='standard', 
                    help='Which CW model to use [standard, upperLimit, mass_ratio, free]')
parser.add_argument('--incPdist', dest='incPdist', action='store_true', \
                    default=False, help='Include pulsar distances for CW signal in run')
parser.add_argument('--CWupperLimit', dest='CWupperLimit', action='store_true', \
                    default=False, help='Calculate CW upper limit (use h not d_L)')
parser.add_argument('--CWmass_ratio', dest='CWmass_ratio', action='store_true', 
                    default=False, help='Use model that parameterizes mass ratio and M')
parser.add_argument('--fixf', dest='fixf', action='store', type=float, default=0.0,
                   help='value of GW frequency for upper limits')


parser.add_argument('--niter', dest='niter', action='store', type=int, default=1000000,
                   help='number MCMC iterations (default=1000000)')
parser.add_argument('--compression', dest='compression', action='store', type=str, \
                    default='None', help='compression type [None, frequencies, average, red]')
parser.add_argument('--sampler', dest='sampler', action='store', type=str, \
                    default='mcmc', help='sampler [mcmc, multinest] default=mcmc')
parser.add_argument('--zerologlike', dest='zerologlike', action='store_true', default=False, \
                    help='Zero log likelihood to test prior and jump proposals')
parser.add_argument('--neff', dest='neff', type=int, action='store', \
                    default=1000, help='Number of effective samples')
parser.add_argument('--resume', dest='resume', action='store_true', \
                    default=False, help='resume from previous run')
parser.add_argument('--writeHotChains', dest='writeHotChains', action='store_true', \
                    default=False, help='Write hot chains in MCMC sampler')

parser.add_argument('--mark6', dest='mark6', action='store_true', \
                    default=False, help='Use T matrix formalism')
parser.add_argument('--Tmatrix', dest='Tmatrix', action='store_true', \
                    default=False, help='Use T matrix formalism')
parser.add_argument('--mark9', dest='mark9', action='store_true', \
                    default=False, help='Use mark9 likelihoodk')
parser.add_argument('--mark10', dest='mark10', action='store_true', \
                    default=False, help='Use mark10 likelihoodk')

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
    args.Tmax=None


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
    #pulsars = list(np.loadtxt('pulsars.txt', dtype='S42'))
    model = PALmodels.PTAmodels(args.h5file, pulsars=pulsars)

# get number of epochs
nepoch = None
if args.incJitterEpoch:
    nepoch = [int(p.nepoch) for p in model.psr]


# model options
separateEfacs = args.separateEfacs == 'backend'
separateEfacsByFreq = args.separateEfacs == 'frequencies'

separateEquads = args.separateEquads == 'backend'
separateEquadsByFreq = args.separateEquads == 'frequencies'

separateJitter = args.separateJitter == 'backend'
separateJitterByFreq = args.separateJitter == 'frequencies'

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

if args.incJitter or args.incJitterEquad or args.incJitterEpoch:
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

#import libstempo as t2
#psr = t2.tempopulsar('/Users/jaellis/Work/pulsars/NANOGrav/1713_21yr/feb/new.par', \
#                     '/Users/jaellis/Work/pulsars/NANOGrav/1713_21yr/feb/1713.Feb.T2.tim', \
#                     maxobs=20000)
#print model.psr[0].Mmat.shape
#print model.psr[0].Mmat[:,0]
#model.psr[0].Mmat = psr.designmatrix(fixunits=True)
#model.psr[0].residuals = psr.residuals()
#print model.psr[0].Mmat.shape
#print model.psr[0].Mmat[:,0]



#likfunc= 'mark5'
print likfunc
fullmodel = model.makeModelDict(incRedNoise=True, noiseModel=args.redModel, \
                    incRedBand=args.incRedBand, \
                    incDMBand=args.incDMBand, \
                    logf=args.logfrequencies, \
                    incDM=args.incDM, dmModel=args.dmModel, \
                    incDMEvent=args.incDMshapelet, dmEventModel=dmEventModel, \
                    ndmEventCoeffs=args.nshape, \
                    incDMX=args.incDMX, \
                    incORF=args.incORF, \
                    incScattering=args.incScat, scatteringModel=args.scatModel,
                    incGWWavelet=args.incGWwavelet, nGWWavelets=args.nGWwavelets,
                    nscatfreqs=args.nfscat,
                    incGWBAni=args.incGWBAni, lmax=args.nls,\
                    incDMXKernel=incDMXKernel, DMXKernelModel=DMXKernelModel, \
                    separateEfacs=separateEfacs, separateEfacsByFreq=separateEfacsByFreq, \
                    separateEquads=separateEquads, separateEquadsByFreq=separateEquadsByFreq, \
                    separateJitter=separateJitter, separateJitterByFreq=separateJitterByFreq, \
                    separateJitterEquad=separateJitterEquad, \
                    separateJitterEquadByFreq=separateJitterEquadByFreq, \
                    incRedFourierMode=incRedFourierMode, incDMFourierMode=incDMFourierMode, \
                    incGWFourierMode=incGWFourierMode, \
                    incEquad=args.incEquad, incJitter=args.incJitter, \
                    incTimingModel=args.incTimingModel, nonLinear=args.tmmodel=='nonlinear', \
                    addPars=args.addpars, subPars=args.delpars, \
                    fulltimingmodel=args.fullmodel, incNonGaussian=args.incNonGaussian, \
                    nnongaussian=args.nnongauss, \
                    incRedExt=args.incRedExt, redExtModel=args.redExtModel, \
                    redExtNf=args.nfext, \
                    incEnvelope=args.incRedEnv, envelopeModel=args.redEnvModel,
                    incCW=args.incCW, incPulsarDistance=args.incPdist, \
                    CWModel=args.cwModel, \
                    CWupperLimit=args.CWupperLimit, \
                    mass_ratio=args.CWmass_ratio, \
                    incJitterEquad=args.incJitterEquad, \
                    incJitterEpoch=args.incJitterEpoch, nepoch=nepoch, \
                    redAmpPrior=args.redAmpPrior, GWAmpPrior=args.GWBAmpPrior, \
                    redSpectrumPrior=args.redAmpPrior, GWspectrumPrior=args.GWBAmpPrior, \
                    incSingleFreqNoise=args.incSingleRed, numSingleFreqLines=1, \
                    incSingleFreqDMNoise=args.incSingleDM, numSingleFreqDMLines=1, \
                    DMAmpPrior=args.DMAmpPrior, \
                    incGWB=incGWB, nfreqs=args.nfreqs, ndmfreqs=args.ndmfreqs, \
                    gwbModel=args.gwbModel, \
                    Tmax = args.Tspan,
                    compression=args.compression, \
                    likfunc=likfunc)


#### Complete Hack to test CW source ####

if args.CWmass_ratio:
    print 'Fixing CW params'
    for sig in fullmodel['signals']:
        if sig['stype'] == 'cw':

            # uniform prior on mass ratio
            sig['prior'][-1] = 'uniform'

            # gaussian prior on total mass
            sig['prior'][2] = 'gaussian'
            sig['mu'][2] = 9.97 + np.log10(3.06)
            sig['sigma'][2] = 0.5

            # gaussian prior on frequency
            sig['prior'][4] = 'gaussian'
            #sig['mu'][4] = -7.67 # orbital period
            sig['mu'][4] = -7.369 # twice orbital period
            sig['sigma'][4] = 0.004

            # fix luminosity distance to 16331.8 Mpc
            sig['bvary'][3] = False
            sig['pstart'][3] = np.log10(16331.8)

            # fix sky location
            sig['bvary'][0] = False
            sig['pstart'][0] = 1.54
            
            sig['bvary'][1] = False
            sig['pstart'][1] = 5.83




# fix spectral index
if args.fixSi:
    print 'Fixing GWB spectral index to 4.33'
    for sig in fullmodel['signals']:
        if sig['corr'] == 'gr' and sig['stype'] in ['powerlaw', 'turnover']:
            sig['bvary'][1] = False
            sig['pstart'][1] = 4.33
        elif sig['corr'] == 'gr_sph' and sig['stype'] == 'powerlaw':
            sig['bvary'][1] = False
            sig['pstart'][1] = 4.33

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

######################################
#                                    #
#   ONLY FOR 1713 21 YR ANALYSIS     #
#                                    #
######################################                                    
for sig in fullmodel['signals']:
    if sig['stype'] == 'efac':
        if np.any([e in sig['flagvalue'] for e in ['M4', 'M3', 'ABPP']]):
            sig['bvary'] = [False]

#for sig in fullmodel['signals']:
#    if sig['stype'] == 'jitter_equad':
#        if np.any([e in sig['flagvalue'] for e in ['430_ASP']]):
#            sig['bvary'] = [False]

memsave = True
if args.noVaryEfac:
    print 'Not Varying EFAC'
    #args.fromFile = False
    #memsave = False
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


if args.fixNoise:
    #noisedir = '/Users/jaellis/Work/pulsars/NANOGrav/9yr/noisefiles/'
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
#args.fromFile = False
model.initModel(fullmodel, memsave=memsave, fromFile=args.fromFile, verbose=True, write=write)

if args.CWmass_ratio:
    print 'Setting Tref'
    model.Tref = 53500.0 * 86400.0

#import matplotlib.pyplot as plt
#for p in model.psr:
#    plt.figure()
#    plt.errorbar(p.toas, p.residuals, p.toaerrs, fmt='.')
#    plt.title(p.name)
#    plt.show()


pardes = model.getModelParameterList()
par_names = [p['id'] for p in pardes if p['index'] != -1]
par_out = []
for pname in par_names:
    if args.pname[0] in pname and len(args.pname) == 1:
        par_out.append(''.join(pname.split('_'+args.pname[0])))
    else:
        par_out.append(pname)

print 'Search Parameters: {0}'.format(par_out)        
    
# output parameter names
fout = open(args.outDir + '/pars.txt', 'w')
for nn in par_out:
    fout.write('%s\n'%(nn))
fout.close()


# define likelihood functions
if args.sampler == 'mcmc' or args.sampler == 'minimize' or args.sampler=='multinest' \
   or args.sampler=='polychord':
    if args.incJitter or args.incJitterEquad or args.incJitterEpoch:
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
        loglike = model.mark10LogLikelihood
                                

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
    if args.noCorrelations or not(np.any([args.incGWB, args.incGWBAni])):
        print 'Running model with no GWB correlations'
        loglkwargs['incCorrelations'] = False
    else:
        loglkwargs['incCorrelations'] = True
        print 'Running model with GWB correlations'
    if args.incJitterEquad and np.any([args.mark6, args.Tmatrix, args.mark10]):
        loglkwargs['incJitter'] = True
    if np.any([args.fixNoise, args.noVaryNoise]) and not args.fixWhite:
        loglkwargs['varyNoise'] = False
    elif np.any([args.fixNoise, args.noVaryNoise]) and args.fixWhite:
        loglkwargs['varyNoise'] = True
        loglkwargs['fixWhite'] = True
    
    if args.zerologlike:
        loglkwargs = {}
   
    # get initial parameters for MCMC
    inRange = False
    pstart = False
    fixpstart=False
    if args.incTimingModel or args.fixNoise or args.noVaryNoise:
        fixpstart=True
    if MPIrank == 0:
        pstart = True
    startSpectrumMin = False
    fixpstart = True
    while not(inRange):
        p0 = model.initParameters(startEfacAtOne=True, fixpstart=fixpstart)
        startSpectrumMin = True
        if args.fixWhite or args.fixNoise or args.noVaryNoise:
            if logprior(p0) != -np.inf and loglike(p0) != -np.inf:
                inRange = True
        else:
            print loglike(p0, **loglkwargs), logprior(p0)
            if logprior(p0) != -np.inf and loglike(p0, **loglkwargs) != -np.inf:
                inRange = True
    
    # if fixed noise, must call likelihood once to initialize matrices
    if args.fixNoise or args.fixWhite or args.noVaryNoise:
        if not args.zerologlike:
            loglike(p0, incCorrelations=False)

    if args.sampler == 'minimize':
        import pyswarm

        # define function
        def fun(x):
            if logprior(x) != -np.inf:
                ret = -loglike(x, **loglkwargs)
            else:
                ret = 1e10

            return ret

        maxpars, maxf = pyswarm.pso(fun, model.pmin, model.pmax, swarmsize=300, \
                        omega=0.5, phip=0.5, phig=0.5, maxiter=1000, debug=True, \
                        minfunc=1e-6)

        np.savetxt(args.outDir + '/pso_maxpars.txt', maxpars)


    if args.sampler == 'mcmc':

        cov = model.initJumpCovariance()
        
        if args.incTimingModel and not args.fixNoise:
            idx = np.arange(len(par_out))
            ind = [np.array([ct for ct, par in enumerate(par_out) if 'efac' in par or \
                    'equad' in par or 'jitter' in par or 'RN' in par or 'DM' in par \
                    or 'red_' in par or 'dm_' in par or 'GWB' in par])]
            ind += [idx[(ind[-1][-1]+1):]]
        elif args.incCW:
            if (args.fixNoise and not args.fixWhite) or args.noVaryEfac or args.noVaryNoise:
                ind = []
            else:
                ind = [np.array([ct for ct, par in enumerate(par_out) if 'efac' in par or \
                        'equad' in par or 'jitter' in par or 'RN' in par or 'DM' in par \
                        or 'red_' in par or 'dm_' in par or 'GWB' in par])]

            ind.append(np.array([ct for ct, par in enumerate(par_out) if \
                        'pdist' not in par and 'efac' not in par and \
                        'equad' not in par and 'jitter' not in par and \
                        'RN' not in par and 'DM' not in par and \
                        'red_' not in par and 'dm_' not in par and \
                        'GWB' not in par and 'lpfgw' not in par and \
                        'pphase' not in par]))

            if args.incPdist:
                ind.append(np.array([ct for ct, par in enumerate(par_out) if \
                        'pdist' in par]))
            if args.cwModel == 'free':
                ind.append(np.array([ct for ct, par in enumerate(par_out) if \
                        'pphase' in par]))
                ind.append(np.array([ct for ct, par in enumerate(par_out) if \
                        'lpfgw' in par]))
        else:
            ind = None
        #ind = None
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
        if args.incEquad:
            sampler.addProposalToCycle(model.drawFromEquadPrior, 5)
        if args.incJitterEquad:
            sampler.addProposalToCycle(model.drawFromJitterEquadPrior, 5)
        if args.incJitterEpoch:
            sampler.addProposalToCycle(model.drawFromJitterEpochPrior, 5)
        if args.incTimingModel:
            sampler.addProposalToCycle(model.drawFromTMfisherMatrix, 40)
            #sampler.addProposalToCycle(model.drawFromTMPrior, 5)
        if args.incCW:
            sampler.addProposalToCycle(model.drawFromCWPrior, 2)
            #sampler.addProposalToCycle(model.massDistanceJump, 2)
            sampler.addProposalToCycle(model.phaseAndPolarizationReverseJump, 5)
            sampler.addAuxilaryJump(model.fix_cyclic_pars)
            if args.cwModel == 'free':
                sampler.addProposalToCycle(model.pulsarPhaseJump, 5)
            if args.incPdist:
                sampler.addAuxilaryJump(model.pulsarPhaseFix)
                sampler.addProposalToCycle(model.pulsarDistanceJump, 10)

        # always include draws from efac
        if not args.noVaryEfac and not args.noVaryNoise:
            sampler.addProposalToCycle(model.drawFromEfacPrior, 2)

        if args.incCW and MPIrank == 0 and not args.zerologlike:

            # call likelihood once to set noise
            loglike(p0, **loglkwargs)
            f = np.logspace(-9, -7, 1000)
            fpstat = np.zeros(len(f))
            for ii in range(1000):
                fpstat[ii] = model.fpStat(f[ii])

            ind = np.argmax(fpstat)
            print 'Starting MCMC CW frequency at {0}'.format(f[ind])
            lfstart = np.log10(f[ind])

            for sig in fullmodel['signals']:
                if sig['stype'] == 'cw':

                    # start frequency
                    sig['pstart'][4] = lfstart

        # run MCMC
        print 'Engage!'
        sampler.sample(p0, args.niter, covUpdate=1000, AMweight=15, SCAMweight=50, \
                       DEweight=20, neff=args.neff, KDEweight=0, Tmin=args.Tmin,
                       Tmax=args.Tmax, writeHotChains=args.writeHotChains)

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
        if args.incJitter or args.incJitterEquad:

            ndim = len(p0)

            def myloglike(cube, ndim, nparams):
                
                acube = np.zeros(ndim)
                for ii in range(ndim):
                    acube[ii] = cube[ii]

                # check prior
                if model.mark3LogPrior(acube) != -np.inf:
                    return loglike(acube, **loglkwargs)
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
                    ll = loglike(acube, **loglkwargs)
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
                        verbose = True, sampling_efficiency = 0.2, \
                        outputfiles_basename =  args.outDir+'/mn'+'-', \
                        n_iter_before_update=5, n_live_points=nlive, \
                        const_efficiency_mode=False, importance_nested_sampling=False, \
                        n_clustering_params=n_params, init_MPI=False)


