#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import os, sys, time
import PALInferencePTMCMC as ptmcmc
import bayesutils as bu

class RJMCMCSampler(object):
    
    """
    Reverse Jump Markov Chain Monte Carlo (RJMCMC) Sampler.
    This implementation is very basic at the moment and 
    simply used the covariance matrix and MAP estimates
    from fixed dimension runs of the various models for 
    trans-dimensional jump proposals. For intra-model jumps
    we just use default PTMCMC jumps (no PT for now).


    @param models: list of model identifiers (used as keys for dictionaries)
    @param pars: list of initial parameter vectors for each model
    @param logp: log prior function (must be normalized for evidence evaluation)
    @param cov: Initial covariance matrix of model parameters for jump proposals
    @param outDir: Full path to output directory for chain files (default = ./chains)

    """


    def __init__(self, models, pars, logls, logps):

        self.modelDict = {}
        self.samplerDict = {}
        self.TDJumpDict = {}
        self.nmodels = 0


    
    def addSampler(self, model, ndim, logl, logp, cov, loglargs=[], loglkwargs={}, \
                    logpargs=[], logpkwargs={}, comm=MPI.COMM_WORLD, \
                    outDir='./chains', verbose=False):
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
        
        # add PTMCMC sampler to sampler dictionary
        self.samplerDict[model] = ptmcmc.PTSampler(ndim, logl, logp, cov, \
                                    loglargs=loglargs, loglkwargs=loglkwargs, \
                                    logpargs=logpargs, logpkwargs=logpkwargs, \
                                    comm=comm, outDir=outDir, verbose=verbose)

        # update model counter
        self.nmodels += 1


    def constructSimpleGaussianTDJump(self, model, fixedchain, chaincov):

        """

        Construct a simple gaussian jump proposal from fixed-dimension
        chain and covariance matrix. Uses MAP values for mean.

        @param model: name of model (used in dictionaries to distinguish models)



