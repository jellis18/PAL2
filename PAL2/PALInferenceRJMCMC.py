#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import os, sys, time
import PALInferencePTMCMC as ptmcmc
import bayesutils as bu
import scipy.stats as ss

try:
    from mpi4py import MPI
except ImportError:
    from . import nompi4py as MPI


class RJMCMCSampler(object):
    
    """
    Reverse Jump Markov Chain Monte Carlo (RJMCMC) Sampler.
    This implementation is very basic at the moment and 
    simply used the covariance matrix and MAP estimates
    from fixed dimension runs of the various models for 
    trans-dimensional jump proposals. For intra-model jumps
    we just use default PTMCMC jumps (no PT for now).


    @param models: list of model identifiers (used as keys for dictionaries)
    @param ndims: list of number of dimensionas for each model
    @param logls: log likelihood function (must be normalized for evidence evaluation)
    @param logps: log prior function (must be normalized for evidence evaluation)
    @param covs: Initial covariance matrix of model parameters for jump proposals
    @param chains: list of chains from single model MCMCs
    @param outDir: Full path to output directory for chain files (default = ./rj_chains)

    """


    def __init__(self, models, ndims, logls, logps, covs, chains, outDir='./rj_chains/'):

        # initialize dictionaries and lists
        self.samplerDict = {}
        self.TDJumpDict = {}
        self.logpDict = {}
        self.loglDict = {}
        self.models = models
        self.outDir = outDir
        
        # setup output file
        if not os.path.exists(self.outDir):
            try:
                os.makedirs(self.outDir)
            except OSError:
                pass
        
        # initialize counters
        self.nmodels = 0
        self.naccepted = []
        self.TDproposed = []
        self.tnaccepted = 0
        self.tTDproposed = 0
        
        # initialize proposal cycle
        self.propCycle = []

        # fill in dictionaries
        # TODO for now just use default arguments, later include a
        # better way to initialize
        for model, ndim, cov, logl, logp, chain in zip(models, ndims, \
                                            covs, logls, logps, chains):

            # log prior and log likelihood dicts
            self.logpDict[model] = logp
            self.loglDict[model] = logl

            # initialize sampler
            self.addSampler(model, ndim, logl, logp, cov)

            # make TD jump proposals from chains
            self.constructGaussianKDE(model, chain)

        # make acceptance and proposed lists for each transition
        for ii in range(self.nmodels):
            for jj in range(self.nmodels):
                if ii != jj:
                    self.naccepted.append([1,1])
                    self.TDproposed.append([1,1])


    
    def addSampler(self, model, ndim, logl, logp, cov, loglargs=[], loglkwargs={}, \
                    logpargs=[], logpkwargs={}, comm=MPI.COMM_WORLD, \
                    outDir=None, verbose=False):
        """
        Add sampler class from PALInferencePTMCMC.

        @param model: name of model (used in dictionaries to distinguish models)
        @param ndim: number of dimensions in problem
        @param logl: log-likelihood function
        @param logp: log prior function (must be normalized for evidence evaluation)
        @param cov: Initial covariance matrix of model parameters for jump proposals
        @param loglargs: any additional arguments (apart from the parameter vector) for 
        log likelihood
        @param loglkwargs: any additional keyword arguments (apart from the parameter vector) 
        for log likelihood
        @param logpargs: any additional arguments (apart from the parameter vector) for 
        log like prior
        @param logpkwargs: any additional keyword arguments (apart from the parameter vector) 
        for log prior
        @param outDir: Full path to output directory for chain files (default = ./chains)
        @param verbose: Update current run-status to the screen (default=False)

        """

        # default output directory
        if outDir is None:
            outDir = self.outDir + '/' + str(model) + '/'
        
        # setup output file
        if not os.path.exists(outDir):
            try:
                os.makedirs(outDir)
            except OSError:
                pass
        
        # add PTMCMC sampler to sampler dictionary
        self.samplerDict[model] = ptmcmc.PTSampler(ndim, logl, logp, cov, \
                                    loglargs=loglargs, loglkwargs=loglkwargs, \
                                    logpargs=logpargs, logpkwargs=logpkwargs, \
                                    comm=comm, outDir=outDir, verbose=verbose)
        
        # update counter
        self.nmodels += 1


    def constructGaussianKDE(self, model, chain):

        """
        Construct a gaussian KDE from a previously run MCMC chain.
        Uses scipy.stats.gaussian_kde

        @param model: name of model
        @param chain: post burn in mcmc chain size nsamples x nparams

        """

        self.TDJumpDict[model] = ss.gaussian_kde(chain.T)


    def _getModelIndex(self, model):
        """
        Return index of model given model key

        """
        return np.flatnonzero(np.array(self.models) == model)[0]
    

    def _updatePriorOdds(self, model0, model1):
        """
        Update prior Odds ratio in attempt to make Odds ratio ~ 1

        """

        # check for adaptive odds ratio
        modelindex1 = self._getModelIndex(model1)
        modelindex0 = self._getModelIndex(model0)
        if self.iterations > 1000 and self.iterations <= self.burn:
            n1 = np.sum(self._modelchain[0:self.iterations] == modelindex1)
            n0 = np.sum(self._modelchain[0:self.iterations] == modelindex0)
            if n1 == 0:
                n1 = 1
            if n0 == 0:
                n0 = 1
            odds = n1/n0
            self.odds[modelindex0, modelindex1] = np.log(1/odds)
            self.odds[modelindex1, modelindex0] = np.log(odds)

        #print modelindex1, modelindex0, self.odds[modelindex0, modelindex1]

        return self.odds[modelindex0, modelindex1]


    def sample(self, model0, p0, Niter, burn=10000, \
                    adaptiveOdds=False, TDprob=0.5, \
                    thin=1, isave=1000):

        """
        Perform RJMCMC sampler

        @param model0: Initial model string
        @param p0: Initial parameter vector for model0
        @param Niter: Number of iterations for RJMCMC
        @param burn: number of iterations for burn in 
        @param adaptiveOdds: update prior odds ratio during burn in to keep Odds ~ 1
        @param TDprob: Probability of proposing TD jump at each iteration
        @param thin: How much to thin chain (save every thin'th value)
        @param isave: How many iterations before writing to file

        """
        
        # number of thinned samples
        N = int(Niter/thin)
        
        # set up output file
        fname = self.outDir + '/chain.txt'
        self._chainfile = open(fname, 'w')
        self._chainfile.close()

        # initialize model chains
        self.burn = burn
        self._modelchain = np.zeros(N)
        self.iterations = 0

        # set up adaptive odds ratio matrix
        odds = 0
        self.odds = np.zeros((self.nmodels, self.nmodels))

        # initialize iterations dictionary to keep track of which
        # iteration is used in intra-model MCMCs. Also initialize MCMC
        # sampler attributes
        self.iterDict = {}
        for m in self.models:
            self.iterDict[m] = 0
            self.samplerDict[m].initialize(Niter)

        # get values for initial parameters
        lp = self.logpDict[model0](p0)
        if lp == -np.inf:
            lnprob0 = -np.inf
        else:
            lnlike0 = self.loglDict[model0](p0) 
            lnprob0 = lnlike0 + lp

        # save initial values in single-model MCMC chain
        self.samplerDict[model0].updateChains(p0, lnlike0, lnprob0, \
                                              self.iterDict[model0])
        
        # save model
        self._modelchain[0] = self._getModelIndex(model0)

        # add KDE jumps
        self.addProposalToCycle(self.gaussianKDEJump, 20)
        
        # randomize cycle
        self.randomizeProposalCycle()

        # start loop over iterations
        tstart = time.time()
        for ii in range(Niter-1):
            self.iterations += 1
            self.iterDict[model0] += 1
            
            # fix prior odds to get odds ratio ~1
            if self.iterations == self.burn:
                for mm in range(self.nmodels):
                    for nn in range(mm+1, self.nmodels):
                        odds = self._updatePriorOdds(self.models[mm], self.models[nn])

            # propose TD jump 
            alpha = np.random.rand()
            if alpha <= TDprob:
                self.tTDproposed += 1
                m0 = self._getModelIndex(model0)
                newpar, m1, qxy = self._jump(p0, m0, self.iterations)
                newmod = self.models[m1]
                self.TDproposed[m0][m1] += 1

                # evaluate likelihood in new model
                lp = self.logpDict[newmod](newpar)
                if lp == -np.inf:
                    newlnprob = -np.inf
                else:
                    newlnlike = self.loglDict[newmod](newpar) 
                    newlnprob = newlnlike + lp

                # check for adaptive odds ratio
                if adaptiveOdds and self.iterations > self.burn:
                    odds = self._updatePriorOdds(model0, newmod)

                # trans-dimensional hastings ratio
                diff = newlnprob - lnprob0 + qxy + odds
                if diff >= np.log(np.random.rand()):

                    # accept jump
                    self.naccepted[m0][m1] += 1
                    self.tnaccepted += 1

                    #print 'Jumping from model {0} to model {1}'.format(model0, newmod)
                    p0, lnlike0, lnprob0, model0 = newpar, newlnlike, newlnprob, newmod

                # save new values in individual chains
                self.samplerDict[model0].updateChains(p0, lnlike0, lnprob0, \
                                                    self.iterDict[model0])

            # Normal MCMC in model0 
            else:
                p0, lnlike0, lnprob0 = self.samplerDict[model0].PTMCMCOneStep(
                                                        p0, lnlike0, lnprob0, \
                                                        self.iterDict[model0])

            # save model chain
            if self.iterations % thin == 0:
                ind = int(self.iterations/thin)
                self._modelchain[ind] = self._getModelIndex(model0)

            #if self.iterations % 10000 == 0:
            #    print self.odds

            # write to file
            if self.iterations % isave == 0:
                self._writeToFile(fname, self.iterations, isave, thin)

                sys.stdout.write('\r')
                sys.stdout.write('Finished %2.2f percent in %f s Acceptance rate = %g'\
                                 %(self.iterations/Niter*100, time.time() - tstart, \
                                   self.tnaccepted/self.tTDproposed))
                sys.stdout.flush()


    def _writeToFile(self, fname, iter, isave, thin):

        """
        Function to write chain file. File has 2 columns,
        the first is the acceptance rate of TD jumps, and
        the second is the model number.

        Also writes out file containing the adaptive prior odds ratio
        used to improve mixing. Columns are model1, model2, prior odds.
        
        @param fname: chainfile name
        @param iter: Iteration of sampler
        @param isave: Number of iterations between saves
        @param thin: Fraction at which to thin chain

        """

        # print out adaptive odds to file. Columns are model1, model2, odds
        fout = open(self.outDir + '/odds.txt', 'w')
        for ii in range(self.nmodels):
            for jj in range(ii+1, self.nmodels):
                fout.write('%d %d %e\n'%(ii, jj, \
                                        self.odds[ii,jj]))

        fout.close()

        self._chainfile = open(fname, 'a+')
        for jj in range((iter-isave), iter, thin):
            ind = int(jj/thin)
            self._chainfile.write('\t'.join(['%22.22f'%(self.naccepted[ii][kk]/\
                                                        self.TDproposed[ii][kk]) \
                                            for ii in range(self.nmodels) \
                                            for kk in range(self.nmodels) if ii!=kk]))

            self._chainfile.write('\t%d\t'%(self._modelchain[jj]))
            self._chainfile.write('\n')
        self._chainfile.close()
    
    # add jump proposal distribution functions
    def addProposalToCycle(self, func, weight):
        """
        Add jump proposal distributions to cycle with a given weight.

        @param func: jump proposal function
        @param weight: jump proposal function weight in cycle

        """

        # get length of cycle so far
        length = len(self.propCycle)

        # check for 0 weight
        if weight == 0:
            print 'ERROR: Can not have 0 weight in proposal cycle!'
            sys.exit()

        # add proposal to cycle
        for ii in range(length, length + weight):
            self.propCycle.append(func)


    # randomized proposal cycle
    def randomizeProposalCycle(self):
        """
        Randomize proposal cycle that has already been filled

        """

        # get length of full cycle
        length = len(self.propCycle)

        # get random integers
        index = np.random.randint(0, (length-1), length)

        # randomize proposal cycle
        self.randomizedPropCycle = [self.propCycle[ind] for ind in index]


    # call proposal functions from cycle
    def _jump(self, x, mx, iter):
        """
        Call Jump proposals

        """

        # get length of cycle
        length = len(self.propCycle)

        # call function
        q, mq, qxy = self.randomizedPropCycle[np.mod(iter, length)](x, mx, iter)

        # increment proposal cycle counter and re-randomize if at end of cycle
        if iter % length == 0: self.randomizeProposalCycle()

        return q, mq, qxy


    def gaussianKDEJump(self, x0, m0, iter):

        """
        Uses gaussian KDE to propose jump in new model parameter space.

        @param x0: parameter vector in current model
        @param m0: the current model
        @param iter: iteration of the RJMCMC chain

        @return x1: proposed parameter vector in new model
        @return m1: proposed model
        @return qxy: forward-backward jump probability

        """

        # determine which model to propose jump into
        randind = np.random.randint(0, self.nmodels-1)
        m1 = np.flatnonzero(np.array(self.models) != self.models[m0])[randind]

        # new parameters
        x1 = self.TDJumpDict[self.models[m1]].resample(1).flatten()

        # forward-backward jump probability
        p0 = self.TDJumpDict[self.models[m0]].evaluate(x0)
        p1 = self.TDJumpDict[self.models[m1]].evaluate(x1)
        qxy = np.log(p0/p1)

        return x1, m1, qxy




        



