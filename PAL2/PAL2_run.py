#!/usr/bin/env python

from __future__ import division
import numpy as np
import PALdatafile
import PALmodels
import PALInferencePTMCMC
import PALpsr
import glob
import time, os

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

parser.add_argument('--incEquad', dest='incEquad', action='store_true',default=False,
                   help='include Equad')
parser.add_argument('--separateEquads', dest='separateEquads', action='store', type=str, \
                    default='None', help='separate equads [None, backend, frequencies]')

parser.add_argument('--noVaryEfac', dest='noVaryEfac', action='store_true',default=False,
                   help='Option to not vary efac')
parser.add_argument('--separateEfacs', dest='separateEfacs', action='store', type=str, \
                    default='None', help='separate efacs [None, backend, frequencies]')

parser.add_argument('--incJitter', dest='incJitter', action='store_true',default=False,
                   help='include Jitter')
parser.add_argument('--separateJitter', dest='separateJitter', action='store', type=str, \
                    default='None', help='separate Jitter [None, backend, frequencies]')

parser.add_argument('--incJitterEquad', dest='incJitterEquad', action='store_true',\
                    default=False, help='include Jitter')
parser.add_argument('--separateJitterEquad', dest='separateJitterEquad', action='store', \
                    type=str, default='None', \
                    help='separate Jitter Equad [None, backend, frequencies]')

parser.add_argument('--noMarg', dest='noMarg', action='store_true',default=False,
                   help='No analytic marginalization')


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

parser.add_argument('--incNonGaussian', dest='incNonGaussian', action='store_true', \
                    default=False, \
                    help='Use non-gaussian likelihood function')
parser.add_argument('--nnongauss', dest='nnongauss', action='store', type=int, default=3,
                   help='number of non-guassian components (default=3)')

parser.add_argument('--incCW', dest='incCW', action='store_true', \
                    default=False, help='Include CW signal in run')
parser.add_argument('--incPdist', dest='incPdist', action='store_true', \
                    default=False, help='Include pulsar distances for CW signal in run')

parser.add_argument('--incBurst', dest='incBurst', action='store_true', \
                    default=False, help='Include Burst signal in run')
parser.add_argument('--burstModel', dest='burstModel', action='store', type=str, \
                    default='bin', \
                    help='which kind of busrt model [interpolate, bin]')
parser.add_argument('--nBurstAmps', dest='nBurstAmps', type=int, action='store', \
                    default=40, help='Number of burst amplitudes, for interpolate and piecewise')

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

# parse arguments
args = parser.parse_args()

##### Begin Code #####

# MPI initialization
comm = MPI.COMM_WORLD
MPIrank = comm.Get_rank()
MPIsize = comm.Get_size()


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
    model = PALmodels.PTAmodels(args.h5file)

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
    likfunc = 'mark6'

#likfunc= 'mark5'
print likfunc

fullmodel = model.makeModelDict(incRedNoise=True, noiseModel=args.redModel, logf=args.logfrequencies, \
                    incDM=args.incDM, dmModel=args.dmModel, \
                    separateEfacs=separateEfacs, separateEfacsByFreq=separateEfacsByFreq, \
                    separateEquads=separateEquads, separateEquadsByFreq=separateEquadsByFreq, \
                    separateJitter=separateJitter, separateJitterByFreq=separateJitterByFreq, \
                    separateJitterEquad=separateJitterEquad, \
                    incRedFourierMode=incRedFourierMode, incDMFourierMode=incDMFourierMode, \
                    incGWFourierMode=incGWFourierMode, \
                    separateJitterEquadByFreq=separateJitterEquadByFreq, \
                    incEquad=args.incEquad, incJitter=args.incJitter, \
                    incTimingModel=args.incTimingModel, nonLinear=args.tmmodel=='nonlinear', \
                    fulltimingmodel=args.fullmodel, incNonGaussian=args.incNonGaussian, \
                    nnongaussian=args.nnongauss, \
                    incBurst=args.incBurst, burstModel='burst_'+args.burstModel, \
                    nBurstAmps=args.nBurstAmps, \
                    incCW=args.incCW, incPulsarDistance=args.incPdist, \
                    incJitterEquad=args.incJitterEquad, \
                    incJitterEpoch=args.incJitterEpoch, nepoch=nepoch, \
                    redAmpPrior=args.redAmpPrior, GWAmpPrior=args.GWBAmpPrior, \
                    redSpectrumPrior=args.redAmpPrior, GWspectrumPrior=args.GWBAmpPrior, \
                    incSingleFreqNoise=args.incSingleRed, numSingleFreqLines=1, \
                    incSingleFreqDMNoise=args.incSingleDM, numSingleFreqDMLines=1, \
                    DMAmpPrior=args.DMAmpPrior, \
                    incGWB=incGWB, nfreqs=args.nfreqs, ndmfreqs=args.ndmfreqs, \
                    gwbModel=args.gwbModel, \
                    compression=args.compression, \
                    likfunc=likfunc)


# fix spectral index
if args.fixSi:
    print 'Fixing GWB spectral index to 4.33'
    for sig in fullmodel['signals']:
        if sig['corr'] == 'gr':
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

memsave = True
if args.noVaryEfac:
    print 'Not Varying EFAC'
    args.fromFile = False
    memsave = False
    for sig in fullmodel['signals']:
        if sig['corr'] == 'single' and sig['stype'] == 'efac':
            sig['bvary'][0] = False
            sig['pstart'][0] = 1

# check for single efacs
if args.incCW or args.incTimingModel or args.incSingleRed or args.incSingleDM or args.incBurst:
    for p in model.psr:
        numEfacs = model.getNumberOfSignalsFromDict(fullmodel['signals'], \
                stype='efac', corr='single')
        memsave = np.any(numEfacs > 1)

# initialize model
if args.fromFile:
    write = True
else:
    write = 'no'
model.initModel(fullmodel, memsave=memsave, fromFile=args.fromFile, verbose=True, write=write)

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
if args.sampler == 'mcmc':
    if args.incJitter or args.incJitterEquad or args.incJitterEpoch:
        loglike = model.mark2LogLikelihood
    elif args.incTimingModel and args.fullmodel and not args.incNonGaussian:
        loglike = model.mark4LogLikelihood
    elif args.incTimingModel and args.fullmodel and args.incNonGaussian:
        loglike = model.mark5LogLikelihood
    else:
        loglike = model.mark1LogLikelihood
    
    if args.noMarg:
        loglike = model.mark6LogLikelihood

    #loglike = model.mark5LogLikelihood

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
    if args.noCorrelations or not(args.incGWB):
        print 'Running model with no GWB correlations'
        loglkwargs['incCorrelations'] = False
    if args.zerologlike:
        loglkwargs = {}
    
    # get initial parameters for MCMC
    inRange = False
    pstart = False
    fixpstart=False
    if args.incTimingModel:
        fixpstart=True
    if MPIrank == 0:
        pstart = True
    startSpectrumMin = False
    while not(inRange):
        p0 = model.initParameters(startEfacAtOne=True, fixpstart=fixpstart)
        #for ct, nm in enumerate(par_out):
        #    print nm, p0[ct]
        startSpectrumMin = True
        if logprior(p0) != -np.inf and loglike(p0, incCorrelations=False) != -np.inf:
            inRange = True

    cov = model.initJumpCovariance()

    # define MCMC sampler
    sampler = PALInferencePTMCMC.PTSampler(len(p0), loglike, logprior, cov, comm=comm, \
                                           outDir=args.outDir, loglkwargs=loglkwargs, \
                                           resume=args.resume, \
                                           gradfun=model.constructGradients)

    # add jump proposals
    if incGWB:
        if args.gwbModel == 'powerlaw':
            sampler.addProposalToCycle(model.drawFromGWBPrior, 10)
        elif args.gwbModel == 'spectrum':
            sampler.addProposalToCycle(model.drawFromGWBSpectrumPrior, 10)
    if args.incRed and args.redModel=='powerlaw':
        sampler.addProposalToCycle(model.drawFromRedNoisePrior, 10)
    if args.incRed and args.redModel=='spectrum':
        sampler.addProposalToCycle(model.drawFromRedNoiseSpectrumPrior, 10)
    if args.incEquad:
        sampler.addProposalToCycle(model.drawFromEquadPrior, 10)
    if args.incJitterEquad:
        sampler.addProposalToCycle(model.drawFromJitterEquadPrior, 5)
    if args.incJitterEpoch:
        sampler.addProposalToCycle(model.drawFromJitterEpochPrior, 5)
    if args.incTimingModel:
        sampler.addProposalToCycle(model.drawFromTMfisherMatrix, 20)
    if args.incCW:
        sampler.addProposalToCycle(model.drawFromCWPrior, 3)
        sampler.addProposalToCycle(model.massDistanceJump, 5)
        if args.incPdist:
            sampler.addAuxilaryJump(model.pulsarPhaseFix)
            sampler.addProposalToCycle(model.pulsarDistanceJump, 5)

    # always include draws from efac
    if not args.noVaryEfac:
        sampler.addProposalToCycle(model.drawFromEfacPrior, 2)


    #sampler.addProposalToCycle(sampler.HMCJump, 50)

    # run MCMC
    print 'Starting Sampling'
    sampler.sample(p0, args.niter, covUpdate=1000, AMweight=15, SCAMweight=30, DEweight=20, \
                   neff=args.neff, KDEweight=0, MALAweight=50)


elif args.sampler == 'multinest':
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
                return model.mark2LogLikelihood(acube)
            else:
                print 'WARNING: Prior returns -np.inf!!'
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
                return model.mark1LogLikelihood(acube)
            else:
                print 'WARNING: Prior returns -np.inf!!'
                return -np.inf

            return model.mark1LogLikelihood(acube)

        def myprior(cube, ndim, nparams):

            for ii in range(ndim):
                cube[ii] = model.pmin[ii] + cube[ii] * (model.pmax[ii]-model.pmin[ii])

                # number of live points
    nlive = 500
    n_params = ndim

    # run MultiNest
    pymultinest.run(myloglike, myprior, n_params, resume = False, \
                    verbose = True, sampling_efficiency = 0.3, \
                    outputfiles_basename =  args.outDir+'/mn'+'-', \
                    n_iter_before_update=5, n_live_points=nlive, \
                    const_efficiency_mode=False, \
                    n_clustering_params=n_params, init_MPI=False)


