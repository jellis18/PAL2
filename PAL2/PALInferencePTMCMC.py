#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import os, sys, time

try:
    from mpi4py import MPI
except ImportError:
    from . import nompi4py as MPI


class PTSampler(object):

    """
    Parallel Tempering Markov Chain Monte-Carlo (PTMCMC) sampler. 
    This implementation uses an adaptive jump proposal scheme
    by default using both standard and single component Adaptive
    Metropolis (AM) and Differential Evolution (DE) jumps.

    This implementation also makes use of MPI (mpi4py) to run
    the parallel chains.

    Along with the AM and DE jumps, the user can add custom 
    jump proposals with the ``addProposalToCycle`` fuction. 

    @param ndim: number of dimensions in problem
    @param logl: log-likelihood function
    @param logp: log prior function (must be normalized for evidence evaluation)
    @param cov: Initial covariance matrix of model parameters for jump proposals
    @param outDir: Full path to output directory for chain files (default = ./chains)
    @param verbose: Update current run-status to the screen (default=True)

    """

    def __init__(self, ndim, logl, logp, cov, comm=MPI.COMM_WORLD, \
                 outDir='./chains', verbose=True, nowrite=False):

        # MPI initialization
        self.comm = comm
        self.MPIrank = self.comm.Get_rank()
        self.nchain = self.comm.Get_size()

        self.ndim = ndim
        self.logl = logl
        self.logp = logp
        self.outDir = outDir
        self.verbose = verbose
        self.nowrite = nowrite

        # setup output file
        if not os.path.exists(self.outDir) and not nowrite:
            try:
                os.makedirs(self.outDir)
            except OSError:
                pass

        # set up covariance matrix
        self.cov = cov
        self.U, self.S, v = np.linalg.svd(self.cov)
        self.M2 = np.zeros((self.ndim, self.ndim))
        self.mu = np.zeros(self.ndim)

        # initialize proposal cycle
        self.propCycle = []

        # indicator for auxilary jumps
        self.aux = None
        

    def sample(self, p0, Niter, ladder=None, Tmin=1, Tmax=10, Tskip=100, \
               isave=1000, covUpdate=1000, SCAMweight=20, \
               AMweight=20, DEweight=20, burn=10000, \
               maxIter=None, thin=10, i0=0):

        """
        Function to carry out PTMCMC sampling.

        @param p0: Initial parameter vector
        @param Niter: Number of iterations to use for T = 1 chain
        @param ladder: User defined temperature ladder
        @param Tmin: Minimum temperature in ladder (default=1) 
        @param Tmax: Maximum temperature in ladder (default=10) 
        @param Tskip: Number of steps between proposed temperature swaps (default=100)
        @param isave: Number of iterations before writing to file (default=1000)
        @param covUpdate: Number of iterations between AM covariance updates (default=5000)
        @param SCAMweight: Weight of SCAM jumps in overall jump cycle (default=20)
        @param AMweight: Weight of AM jumps in overall jump cycle (default=20)
        @param DEweight: Weight of DE jumps in overall jump cycle (default=20)
        @param burn: Burn in time (DE jumps added after this iteration) (default=5000)
        @param maxIter: Maximum number of iterations for high temperature chains (default=2*Niter)
        @param thin: Save every thin MCMC samples
        @param i0: Iteration to start MCMC (if i0 !=0, do not re-initialize)

        """

        # get maximum number of iterations
        if maxIter is None:
            maxIter = 2*Niter

        # set jump parameters
        self.ladder = ladder
        self.covUpdate = covUpdate
        self.SCAMweight = SCAMweight
        self.AMweight = AMweight
        self.DEweight = DEweight
        self.burn = burn
        self.Tskip = Tskip

        # set up arrays to store lnprob, lnlike and chain
        N = int(maxIter/thin)
        
        # if picking up from previous run, don't re-initialize
        if i0 == 0:
            self._lnprob = np.zeros(N)
            self._lnlike = np.zeros(N)
            self._chain = np.zeros((N, self.ndim))
            self.naccepted = 0
            self.swapProposed = 0
            self.nswap_accepted = 0

            # set up covariance matrix and DE buffers
            # TODO: better way of allocating this to save memory
            if self.MPIrank == 0:
                self._AMbuffer = np.zeros((maxIter, self.ndim))
                self._DEbuffer = np.zeros((self.burn, self.ndim))

            ### setup default jump proposal distributions ###

            # add SCAM
            self.addProposalToCycle(self.covarianceJumpProposalSCAM, self.SCAMweight)
            
            # add AM
            self.addProposalToCycle(self.covarianceJumpProposalAM, self.AMweight)
            
            # randomize cycle
            self.randomizeProposalCycle()

            # setup default temperature ladder
            if self.ladder is None:
                self.ladder = self.temperatureLadder(Tmin)
        
            # temperature for current chain
            self.temp = self.ladder[self.MPIrank]

            # set up output file
            fname = self.outDir + '/chain_{0}.txt'.format(self.temp)
            if not self.nowrite:
                self._chainfile = open(fname, 'w')
                self._chainfile.close()


        ### compute lnprob for initial point in chain ###
        self._chain[i0,:] = p0

        # compute prior
        lp = self.logp(p0)

        if lp == float(-np.inf):

            lnprob0 = -np.inf
            lnlike0 = -np.inf

        else:

            lnlike0 = self.logl(p0) 
            lnprob0 = 1/self.temp * lnlike0 + lp

        # record first values
        self._lnprob[i0] = lnprob0
        self._lnlike[i0] = lnlike0

        self.comm.barrier()


        # start iterations
        iter = i0
        tstart = time.time()
        runComplete = False
        getCovariance = 0
        getDEbuf = 0
        while runComplete == False:
            iter += 1
            accepted = 0
            
            # call PTMCMCOneStep
            p0, lnlike0, lnprob0 = self.PTMCMCOneStep(p0, lnlike0, lnprob0, iter)

            # update buffer
            if self.MPIrank == 0:
                self._AMbuffer[iter,:] = p0
            
            # put results into arrays
            if iter % thin == 0:
                ind = int(iter/thin)
                self._chain[ind,:] = p0
                self._lnlike[ind] = lnlike0
                self._lnprob[ind] = lnprob0

            # write to file
            if iter % isave == 0:
                if not self.nowrite:
                    self._writeToFile(fname, iter, isave, thin)
                if self.MPIrank == 0 and self.verbose:
                    sys.stdout.write('\r')
                    sys.stdout.write('Finished %2.2f percent in %f s Acceptance rate = %g'\
                                     %(iter/Niter*100, time.time() - tstart, \
                                       self.naccepted/iter))
                    sys.stdout.flush()

                    # write output covariance matrix
                    np.save(self.outDir + '/cov.npy', self.cov)

            # stop
            if self.MPIrank == 0 and iter >= Niter-1:
                if self.verbose:
                    print '\nRun Complete'
                runComplete = True

            if self.MPIrank == 0 and runComplete:
                for jj in range(1, self.nchain):
                    self.comm.send(runComplete, dest=jj, tag=55)

            # check for other chains
            if self.MPIrank > 0:
                runComplete = self.comm.Iprobe(source=0, tag=55)
                time.sleep(0.000001) # trick to get around 


    def PTMCMCOneStep(self, p0, lnlike0, lnprob0, iter):

        """
        Function to carry out PTMCMC sampling.

        @param p0: Initial parameter vector
        @param lnlike0: Initial log-likelihood value
        @param lnprob0: Initial log probability value
        @param iter: iteration number

        @return p0: next value of parameter vector after one MCMC step
        @return lnlike0: next value of likelihood after one MCMC step
        @return lnprob0: next value of posterior after one MCMC step

        """ 
        # update covariance matrix
        if (iter-1) % self.covUpdate == 0 and (iter-1) != 0 and self.MPIrank == 0:
            self._updateRecursive(iter-1, self.covUpdate)

            # broadcast to other chains
            [self.comm.send(self.cov, dest=rank+1, tag=111) for rank 
                                            in range(self.nchain-1)]
        
        # check for sent covariance matrix from T = 0 chain
        getCovariance = self.comm.Iprobe(source=0, tag=111)
        time.sleep(0.000001) 

        if getCovariance and self.MPIrank > 0:
            self.cov = self.comm.recv(source=0, tag=111)
            getCovariance = 0

        # update DE buffer
        if (iter-1) % self.burn == 0 and (iter-1) != 0 and self.MPIrank == 0:
            self._updateDEbuffer(iter-1, self.burn)

            # broadcast to other chains
            [self.comm.send(self._DEbuffer, dest=rank+1, tag=222) for rank 
                                                    in range(self.nchain-1)]
        
        # check for sent DE buffer from T = 0 chain
        getDEbuf = self.comm.Iprobe(source=0, tag=222)
        time.sleep(0.000001) 

        if getDEbuf and self.MPIrank > 0:
            self._DEbuffer = self.comm.recv(source=0, tag=222)
            self.addProposalToCycle(self.DEJump, self.DEweight)
            
            # randomize cycle
            self.randomizeProposalCycle()
            getDEbuf = 0

        # after burn in, add DE jumps
        if (iter-1) == self.burn and self.MPIrank == 0:
            self.addProposalToCycle(self.DEJump, self.DEweight)
            
            # randomize cycle
            self.randomizeProposalCycle()
        
        
        # jump proposal
        y, qxy = self._jump(p0, iter)


        # compute prior and likelihood
        lp = self.logp(y)
        
        if lp == -np.inf:

            newlnprob = -np.inf

        else:

            newlnlike = self.logl(y) 
            newlnprob = 1/self.temp * newlnlike + lp

        # hastings step
        diff = newlnprob - lnprob0 + qxy

        if diff >= np.log(np.random.rand()):

            # accept jump
            p0, lnlike0, lnprob0 = y, newlnlike, newlnprob

            # update acceptance counter
            self.naccepted += 1
            accepted = 1


        ##################### TEMPERATURE SWAP ###############################
        readyToSwap = 0
        swapAccepted = 0

        # if Tskip is reached, block until next chain in ladder is ready for swap proposal
        if iter % self.Tskip == 0 and self.MPIrank < self.nchain-1:
            self.swapProposed += 1

            # send current likelihood for swap proposal
            self.comm.send(lnlike0, dest=self.MPIrank+1)

            # determine if swap was accepted
            swapAccepted = self.comm.recv(source=self.MPIrank+1)

            # perform swap
            if swapAccepted:
                self.nswap_accepted += 1

                # exchange likelihood
                lnlike0 = self.comm.recv(source=self.MPIrank+1)

                # exchange parameters
                self.comm.send(p0, dest=self.MPIrank+1)
                p0 = self.comm.recv(source=self.MPIrank+1)

                # calculate new posterior values
                lnprob0 = 1/self.temp * lnlike0 + self.logp(p0)


        # check if next lowest temperature is ready to swap
        elif self.MPIrank > 0:

            readyToSwap = self.comm.Iprobe(source=self.MPIrank-1)
             # trick to get around processor using 100% cpu while waiting
            time.sleep(0.000001) 

            # hotter chain decides acceptance
            if readyToSwap:
                newlnlike = self.comm.recv(source=self.MPIrank-1)
                
                # determine if swap is accepted and tell other chain
                logChainSwap = (1/self.ladder[self.MPIrank-1] - 1/self.ladder[self.MPIrank]) \
                        * (lnlike0 - newlnlike)

                if logChainSwap >= np.log(np.random.rand()):
                    swapAccepted = 1
                else:
                    swapAccepted = 0

                # send out result
                self.comm.send(swapAccepted, dest=self.MPIrank-1)

                # perform swap
                if swapAccepted:
                    self.nswap_accepted += 1

                    # exchange likelihood
                    self.comm.send(lnlike0, dest=self.MPIrank-1)
                    lnlike0 = newlnlike

                    # exchange parameters
                    self.comm.send(p0, dest=self.MPIrank-1)
                    p0 = self.comm.recv(source=self.MPIrank-1)
                
                    # calculate new posterior values
                    lnprob0 = 1/self.temp * lnlike0 + self.logp(p0)


        ##################################################################

        return p0, lnlike0, lnprob0



    def temperatureLadder(self, Tmin, Tmax=None, tstep=None):

        """
        Method to compute temperature ladder. At the moment this uses
        a geometrically spaced temperature ladder with a temperature
        spacing designed to give 25 % temperature swap acceptance rate.

        """

        #TODO: make options to do other temperature ladders

        if self.nchain > 1:
            if tstep is None:
                tstep = 1 + np.sqrt(2/self.ndim)
            ladder = np.zeros(self.nchain)
            for ii in range(self.nchain): ladder[ii] = Tmin*tstep**ii
        else:
            ladder = np.array([1])

        return ladder


    def _writeToFile(self, fname, iter, isave, thin):

        """
        Function to write chain file. File has 3+ndim columns,
        the first is log-posterior (unweighted), log-likelihood,
        and accepatence probability, followed by parameter values.
        
        @param fname: chainfile name
        @param iter: Iteration of sampler
        @param isave: Number of iterations between saves
        @param thin: Fraction at which to thin chain

        """

        self._chainfile = open(fname, 'a+')
        for jj in range((iter-isave), iter, thin):
            ind = int(jj/thin)
            self._chainfile.write('%.17e\t %.17e\t %e\t'%(self._lnprob[ind], self._lnlike[ind],\
                                                  self.naccepted/iter))
            self._chainfile.write('\t'.join(["%.17e"%(self._chain[ind,kk]) \
                                            for kk in range(self.ndim)]))
            self._chainfile.write('\n')
        self._chainfile.close()


    # function to update covariance matrix for jump proposals
    def _updateRecursive(self, iter, mem):

        """ 
        Function to recursively update sample covariance matrix.

        @param iter: Iteration of sampler
        @param mem: Number of steps between updates

        """

        it = iter - mem

        if it == 0:
            self.M2 = np.zeros((self.ndim, self.ndim))
            self.mu = np.zeros(self.ndim)

        for ii in range(mem):
            diff = np.zeros(self.ndim)
            it += 1
            for jj in range(self.ndim):
                
                diff[jj] = self._AMbuffer[iter-mem+ii,jj] - self.mu[jj]
                self.mu[jj] += diff[jj]/it

            self.M2 += np.outer(diff, (self._AMbuffer[iter-mem+ii,:]-self.mu))

        self.cov = self.M2/(it-1)  

        # do svd
        self.U, self.S, v = np.linalg.svd(self.cov)

    # update DE buffer samples
    def _updateDEbuffer(self, iter, burn):
        """
        Update Differential Evolution with last burn
        values in the total chain

        @param iter: Iteration of sampler
        @param burn: Total number of samples in DE buffer

        """

        self._DEbuffer = self._AMbuffer[iter-burn:iter]

        
    # SCAM jump
    def covarianceJumpProposalSCAM(self, x, iter, beta):

        """
        Single Component Adaptive Jump Proposal. This function will occasionally
        jump in more than 1 parameter. It will also occasionally use different
        jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        q = x.copy()
        qxy = 0

        # number of parameters to update at once 
        prob = np.random.rand()
        if prob > (1 - 1/self.ndim):
            block = self.ndim

        elif prob > (1 - 2/self.ndim):
            block = np.ceil(self.ndim/2)

        elif prob > 0.8:
            block = 5

        else:
            block = 1

        # adjust step size
        prob = np.random.rand()

        # small jump
        if prob > 0.9:
            scale = 0.2

        # large jump
        elif prob > 0.97:
            scale = 10
        
        # small-medium jump
        elif prob > 0.6:
            scale = 0.5

        # standard medium jump
        else:
            scale = 1.0

        # adjust scale based on temperature
        if self.temp <= 100:
            scale *= np.sqrt(self.temp)

        # get parmeters in new diagonalized basis
        y = np.dot(self.U.T, x)

        # make correlated componentwise adaptive jump
        ind = np.unique(np.random.randint(0, self.ndim, block))
        neff = len(ind)
        cd = 2.4  / np.sqrt(2*neff) * scale 

        y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(self.S[ind])
        q = np.dot(self.U, y)

        return q, qxy
    
    # AM jump
    def covarianceJumpProposalAM(self, x, iter, beta):

        """
        Adaptive Jump Proposal. This function will occasionally 
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        q = x.copy()
        qxy = 0

        # adjust step size
        prob = np.random.rand()

        # small jump
        if prob > 0.9:
            scale = 0.2

        # large jump
        elif prob > 0.97:
            scale = 10
        
        # small-medium jump
        elif prob > 0.6:
            scale = 0.5

        # standard medium jump
        else:
            scale = 1.0

        # adjust scale based on temperature
        if self.temp <= 100:
            scale *= np.sqrt(self.temp)

        cd = 2.4/np.sqrt(2*self.ndim) * np.sqrt(scale)
        q = np.random.multivariate_normal(x, cd**2*self.cov)

        return q, qxy


    # Differential evolution jump
    def DEJump(self, x, iter, beta):

        """
        Differential Evolution Jump. This function will  occasionally 
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        # get old parameters
        q = x.copy()
        qxy = 0

        bufsize = np.alen(self._DEbuffer)

        # draw a random integer from 0 - iter
        mm = np.random.randint(0, bufsize)
        nn = np.random.randint(0, bufsize)

        # make sure mm and nn are not the same iteration
        while mm == nn: nn = np.random.randint(0, bufsize)

        # get jump scale size
        prob = np.random.rand()

        # mode jump
        if prob > 0.5:
            scale = 1.0
            
        else:
            scale = np.random.rand() * 2.4/np.sqrt(2*self.ndim) * np.sqrt(1/beta)

        for ii in range(self.ndim):
            
            # jump size
            sigma = self._DEbuffer[mm, ii] - self._DEbuffer[nn, ii]

            # jump
            q[ii] += scale * sigma
        
        return q, qxy


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


    # add auxilary jump proposal distribution functions
    def addAuxilaryJump(self, func):
        """
        Add auxilary jump proposal distribution. This will be called after every
        standard jump proposal. Examples include cyclic boundary conditions and 
        pulsar phase fixes

        @param func: jump proposal function

        """
        
        # set auxilary jump
        self.aux = func

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
    def _jump(self, x, iter):
        """
        Call Jump proposals

        """

        # get length of cycle
        length = len(self.propCycle)

        # call function
        q, qxy = self.randomizedPropCycle[np.mod(iter, length)](x, iter, 1/self.temp)

        # axuilary jump
        if self.aux is not None:
            q, qxy_aux = self.aux(x, q, iter, 1/self.temp)
            qxy += qxy_aux

        # increment proposal cycle counter and re-randomize if at end of cycle
        if iter % length == 0: self.randomizeProposalCycle()

        return q, qxy

    # TODO: jump statistics











