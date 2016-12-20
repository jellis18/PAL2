from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import jdcal, time
import sys
from scipy.stats import mode
import scipy.linalg as sl
from collections import OrderedDict
import json

from PAL2 import PALutils
from PAL2 import PALmodels
from PAL2 import bayesutils as bu
from scipy.interpolate import interp1d 

def convert_mjd_to_greg(toas):
    """
    Converts MJD array to year array

    :param toas: Array of TOAs in days

    :return: Array of TOAs in years
    """
 
    dates = np.zeros(len(toas))
    for ct, toa in enumerate(toas):
        x = jdcal.jd2gcal(2400000.5, toa)
        dates[ct] = x[0] + x[1]/12 + x[2]/365.25 + x[3]/365.25

    return dates


def compute_daily_ave(times, res, err, ecorr=None, dt=10, flags=None):
    """
    Computes daily averaged residuals 

     :param times: TOAs in seconds
     :param res: Residuals in seconds
     :param err: Scaled (by EFAC and EQUAD) error bars in seconds
     :param ecorr: (optional) ECORR value for each point in s^2 [default None]
     :param dt: (optional) Time bins for averaging [default 10 s]
     :param flags: (optional) Array of flags [default None]

     :return: Average TOAs in seconds
     :return: Average error bars in seconds
     :return: Average residuals in seconds
     :return: (optional) Average flags
     """

    isort = np.argsort(times)

    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]

    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])

    avetoas = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    if flags is not None:
        aveflags = np.array([flags[l[0]] for l in bucket_ind])

    aveerr = np.zeros(len(bucket_ind))
    averes = np.zeros(len(bucket_ind))

    for i,l in enumerate(bucket_ind):
        M = np.ones(len(l))
        C = np.diag(err[l]**2) 
        if ecorr is not None:
            C += np.ones((len(l), len(l))) * ecorr[l[0]]

        avr = 1/np.dot(M, np.dot(np.linalg.inv(C), M))
        aveerr[i] = np.sqrt(avr)
        averes[i] = avr * np.dot(M, np.dot(np.linalg.inv(C), res[l]))


    if flags is not None:
        return avetoas, aveerr, averes, aveflags
    else:
        return avetoas, aveerr, averes


def get_ave_res(model, pars, flags, indicator=None, DM=False):

    """
    Get whitened and full modeled averaged residuals.

    :param model: PAL2 Model class
    :param pars: ML parameters
    :param pars: TOA flags
    :param indicator: (optional) indicator array for RJMCMC runs [Default=None]
    :param DM: (optional) Include DM in model [Default=False]

    :return: Averaged TOAs in seconds
    :return: Averaged residuals in seconds
    :return: Averaged whitened residuals in seconds
    :return: Averaged error bars in seconds
    :return: Full residuals in seconds
    :return: Full whitened residuals
    :return: Normalized whitened residuals
    :return: Flags per epoch
    """

    model.setPsrNoise(pars, twoComponent=False)
    model.updateDetSources(pars, selection=indicator)

    res_red, red_err = model.create_realization(pars, signal='red', 
                                                incJitter=True, selection=indicator)
    res_tm, tm_err = model.create_realization(pars, signal='tm', 
                                              incJitter=True, selection=indicator)
    if DM:
        res_dm, dm_err = model.create_realization(pars, signal='dm', 
                                                  incJitter=True, selection=indicator)
    res_jitter, red_err = model.create_realization(pars, signal='jitter', 
                                                   incJitter=True, selection=indicator)

    whiteres = model.psr[0].detresiduals - res_tm[0] - res_red[0] 
    whiteres_full = whiteres - res_jitter[0]
    tmres = model.psr[0].residuals - res_tm[0] 
    
    if DM:
        whiteres -= res_dm[0]
        whiteres_full -= res_dm[0]
        tmres -= res_dm[0]
    
    norm1 = whiteres_full / np.sqrt(model.psr[0].Nvec)

    njitter = len(model.psr[0].avetoas)
    ecorr = np.dot(model.psr[0].Ttmat[:,-njitter:], model.psr[0].Qamp)
    avetoas, aveerr, averes, aveflags = compute_daily_ave(model.psr[0].toas, tmres, 
                                       np.sqrt(model.psr[0].Nvec), 
                                       ecorr=ecorr,dt=1.0, flags=flags)
    avetoas, aveerr, avewhiteres = compute_daily_ave(model.psr[0].toas, whiteres, 
                                            np.sqrt(model.psr[0].Nvec), 
                                            ecorr=ecorr, dt=1.0)

    return avetoas, averes, avewhiteres, aveerr, tmres, whiteres, norm1, aveflags
    


def get_quad_posteriors(model, chain, selection=None,
                        N=1000, add_det=False, idd=None, 
                        fixmask=None, maxpars=None, 
                        maxsel=None):

    """
    Get posteriors on quadratic parameters and waveform
    realizations.

    :param model: PAL2 Model class
    :param chain: MCMC chain file
    :param selection: (optional) Indicator array for RJMCMC runs [default=None]
    :param N: (optional) Number of posterior draws [default=1000]
    :param add_det: (optional) Add deterministic sources into waveform realization
    :param idd: 
        (optional) Parameter indices of quadratic parameters to make waveform
        realizations. Defaults to red noise signal.
    :param fixmask: 
        (optional) Boolean mask for fixed parameters (useful for DMX)
    :param maxpars: 
        (optional) Include ML params to compute conditional posterior
        on quadratid parameters
    :param maxsel: 
        (optional) Include ML indicator array to compute conditional posterior
        on quadratid parameters

    :return: Posterior samples from quadratic parameters [N x npar]
    :return: Posterior samples of wavform realization [N x ntoa]
    :return: Posterior samples of whitened normalized residuals [N x ntoa]
    """
    
    M = chain.shape[0]
    p = model.psr[0]
    nt = len(p.toas)
    ret = np.zeros((N, p.Tmat.shape[1]))
    real = np.zeros((N, nt))
    white = np.zeros((N, nt))

    if add_det:
        det = np.zeros((N, nt))
        
    if selection is None:
        sel = np.array([1]*model.dimensions, dtype=np.bool)
    
    for ii in range(N):
        
        if maxpars is None:
            indx = np.random.randint(0, M)
            pars = chain[indx,:]
        else:
            pars = maxpars
        
        # index array
        if maxsel is None:
            if selection is not None:
                sel = selection[indx,:]
        else:
            sel = maxsel

        if model.likfunc == 'mark9':
            model.mark9LogLikelihood(pars, selection=sel)
        elif model.likfunc == 'mark6':
            model.mark6LogLikelihood(pars, incJitter=True, selection=sel)
        
        if add_det:
            det[ii,:] = p.residuals - p.detresiduals
        
        # whitened residuals
        d = np.dot(p.Ttmat.T, p.detresiduals / p.Nvec)
        
        # sigma matrix
        Sigma = model.TNT + model.Phiinv
        
        # QR decomp      
        try:
            Q, R = sl.qr(Sigma)
            Sigi = sl.solve(R, Q.T)
            mn = np.dot(Sigi, d)

            # conditional pdf
            if fixmask is not None:
                mnx = mn[fixmask]
                mny = mn[~fixmask]
                Sxx = Sigi[fixmask,:][:,fixmask]
                Syy = Sigi[~fixmask,:][:,~fixmask]
                Sxy = Sigi[fixmask,:][:,~fixmask]
                Syx = Sigi[~fixmask,:][:,fixmask]

                cf = sl.cho_factor(Syy)
                SiyySyx = sl.cho_solve(cf, Syx)
                Sxxy = Sxx - np.dot(Sxy, SiyySyx)

            # SVD
            U, s, V = sl.svd(Sigi)
            Li = U * np.sqrt(s)

            ret[ii,:] = mn + np.dot(Li, np.random.randn(Li.shape[0]))
            if fixmask is not None:
                U, s, V = sl.svd(Sxxy)
                Li = U * np.sqrt(s)
                mnx -= np.dot(Sxy, sl.cho_solve(cf, ret[ii,~fixmask]-mny))
                tmp = mnx + np.dot(Li, np.random.randn(Li.shape[0]))
        except np.linalg.LinAlgError:
            ret[ii,:] = np.zeros(len(mn))
            if fixmask is not None:
                tmp = np.zeros(len(mnx))

        if idd is None:
            nf = len(p.Ffreqs) // 2
            #ntmpars = len(p.ptmdescription)
            ntmpars = p.Mmat_reduced.shape[1]
            idd = np.arange(ntmpars, ntmpars+2*nf)


        real[ii,:] = np.dot(p.Ttmat[:,idd], ret[ii,idd])
        white[ii,:] = (p.detresiduals - np.dot(p.Ttmat, ret[ii,:])) / np.sqrt(p.Nvec)
        if add_det:
            real[ii,:] += det[ii,:]

        if fixmask is not None:
            ret[ii,fixmask] = tmp

        sys.stdout.write('\r')
        sys.stdout.write('%g'%(ii/N * 100))
        sys.stdout.flush()
        
    return ret, real, white


def get_spectrum(model, chain, selection=None, N=1000):
    """
    Get posteriors red noise spectrum

    :param model: PAL2 Model class
    :param chain: MCMC chain file
    :param selection: (optional) Indicator array for RJMCMC runs [default=None]
    :param N: (optional) Number of posterior draws [default=1000]


    :return: Frequency array
    :return: Posterior samples of spectrum [N x nfreq]
    """
    
    M = chain.shape[0]
    p = model.psr[0]
    T = (p.toas.max() - p.toas.min())
    nf = len(p.Ffreqs)
    if model.haveExt:
        nf += len(p.Fextfreqs)
    psd = np.zeros((N, nf/2))

    freqs = p.Ffreqs[::2]
    if model.haveExt:
        freqs = np.concatenate((p.Ffreqs, p.Fextfreqs))[::2]

    if selection is None:
        sel = np.array([1]*model.dimensions, dtype=np.bool)
    
    for ii in range(N):
        
        indx = np.random.randint(0, M)
        pars = chain[indx,:]
        
        # index array
        if selection is not None:
            sel = selection[indx,:]
                
        # update phi matrix
        model.constructPhiMatrix(pars, incTM=True, incJitter=True, 
                                 incCorrelations=False, selection=sel)
        
        #ntmpars = len(p.ptmdescription)
        ntmpars = p.Mmat_reduced.shape[1]
        Phi = 1 / np.diag(model.Phiinv)
        psd[ii,:] = Phi[ntmpars:ntmpars+nf][::2] * T / 2 * 1e12

        sys.stdout.write('\r')
        sys.stdout.write('%g'%(ii/N * 100))
        sys.stdout.flush()
        
    
    return freqs, np.log10(psd) 

def get_fourier_spectrum(model, qreal, N=1000):
    """
    Get posteriors red noise fourier spectrum spectrum

    :param model: PAL2 Model class
    :param qreal: Nreal x npar array containing individual quadratic posteriors
    :param N: (optional) Number of posterior draws [default=1000]

    :return: Frequency array
    :return: Posterior samples of spectrum [N x nfreq]
    """
    
    p = model.psr[0]
    T = (p.toas.max() - p.toas.min())
    nf = len(p.Ffreqs)
    psd = np.zeros((N, nf/2))
    
    for ii in range(N):

        ixx = np.random.randint(0, qreal.shape[0])
        
        ntmpars = p.Mmat_reduced.shape[1]
        idd = np.arange(ntmpars, ntmpars+nf)
        a = qreal[ixx,idd]
        psd[ii,:] = (a[::2]**2 + a[1::2]**2)*T/2*1e12

        sys.stdout.write('\r')
        sys.stdout.write('%g'%(ii/N * 100))
        sys.stdout.flush()
        
    
    return p.Ffreqs[::2], np.log10(psd) 

def get_mcmc_files(chaindir):
    """
    Load (RJ)MCMC files.
    :param chaindir: Full path to chain directory

    :return: Chain array [N x npar]
    :return: parameter array [npar]
    :return: Log probability array [N x 2]
    :return: (optional) Indicator array [N x npar]
    """
    
    if os.path.exists(chaindir + 'run_0'):
        chain = np.loadtxt(chaindir + '/run_0/chain.txt')
        pars = np.loadtxt(chaindir + '/pars.txt', dtype='S42')
        burn = int(0.25 * chain.shape[0])
        ind = np.loadtxt(chaindir + '/run_0/indicator.txt', dtype=bool)
        logp = np.loadtxt(chaindir + '/run_0/prob.txt')

        return (chain[burn:,:], pars, logp[burn:,], ind[burn:,])
    
    else:
        try:
            chain = np.loadtxt(chaindir + '/chain_1.txt')
        except IOError:
            chain = np.loadtxt(chaindir + '/chain_1.0.txt')
        pars = np.loadtxt(chaindir + '/pars.txt', dtype='S42')
        burn = int(0.25 * chain.shape[0])
        
        return (chain[burn:,:-4], pars, chain[burn:,-4:-2])


def make_waveform_realization_plot(ax, psr, real, sigma=0.68, *args, **kwargs):
    """
    Make a waveform realization plot of signal vs time with
    uncertantiy region.
    
    :param ax: axes object instance
    :param psr: Pulsar object instance
    :param real: Nreal x ntoa array containing individual waveform realizations.
    :param sigma: Uncertainty level on waveform
    """
    
    nt = real.shape[1]
    xmed, xlow, xhigh = np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for ii in range(nt):
        rlind = np.flatnonzero(real[:,ii])
        tmp, xmed[ii] = bu.confinterval(real[rlind,ii], onesided=True, sigma=0.5)
        xlow[ii], xhigh[ii] = bu.confinterval(real[rlind,ii], sigma=sigma, 
                                              type='minArea')
    
    idx = np.argsort(psr.toas)
    ax.fill_between(convert_mjd_to_greg(psr.toas[idx]/86400), xlow[idx]*1e6, 
                     xhigh[idx]*1e6, **kwargs)
    ax.plot(convert_mjd_to_greg(psr.toas[idx]/86400), xmed[idx]*1e6, ls='--', 
             lw=1.5, **kwargs)


def make_dm_waveform_realization_plot(ax, psr, qreal, incDM=True, *args, **kwargs):
    """
    Make a waveform realization plot of DM vs time with
    uncertantiy region.
    
    :param ax: axes object instance
    :param psr: Pulsar object instance
    :param qreal: Nreal x npar array containing individual quadratic posteriors
    :param incDM: Boolean whether or not DM is in timing model or in noise model
    """
    dmconst = 2.41e-4
    #ntmpars = len(psr.ptmdescription)
    ntmpars = psr.Mmat_reduced.shape[1]
    nf = psr.Fmat.shape[1]
    nfdm = len(psr.Fdmfreqs)
    idd = np.arange(ntmpars+nf,ntmpars+nf+nfdm)
    idx = np.argsort(psr.toas)
    nt = len(idx)
    dmsig = np.zeros((qreal.shape[0], nt))
    for ii in range(qreal.shape[0]):
        if incDM:
            id1 = list(psr.newdes).index('DM1')
            dmsig[ii,:] = psr.Ttmat[:,id1] * qreal[ii,id1] * dmconst * psr.freqs**2
            id2 = list(psr.newdes).index('DM2')
            dmsig[ii,:] += psr.Ttmat[:,id2] * qreal[ii,id2] * dmconst * psr.freqs**2
            dmsig[ii,:] += np.dot(psr.Fdmmat, qreal[ii,idd])
            dmsig[ii,:] -= dmsig[ii,:].mean()
        else:
            idd = np.array([ct for ct,pp in enumerate(psr.ptmdescription) if 'DMX' in pp])
            dmsig[ii,:] = np.dot(psr.Ttmat[:,idd], qreal[ii,idd]) * dmconst * psr.freqs**2
    
    xmed, xlow, xhigh = np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for ii in range(nt):
        rlind = np.flatnonzero(dmsig[:,ii])
        tmp, xmed[ii] = bu.confinterval(dmsig[rlind,ii], onesided=True, sigma=0.5)
        xlow[ii], xhigh[ii] = bu.confinterval(dmsig[rlind,ii], sigma=0.68, 
                                              type='minArea')

    ax.fill_between(convert_mjd_to_greg(psr.toas[idx]/86400), xlow[idx]*1e3, 
                     xhigh[idx]*1e3, **kwargs)
    ax.plot(convert_mjd_to_greg(psr.toas[idx]/86400), (xmed[idx]-xmed[idx].mean())*1e3, ls='--', 
             lw=1.5, color='k')


def make_spectrum_realization_plot(ax, f, psd, sigma=0.68, 
                                   plot_median=True, use_bars=False, 
                                   *args, **kwargs):
    """
    Make a waveform realization plot of signal vs time with
    uncertantiy region.
    
    :param ax: axes object instance
    :param model: PAL2 Model class
    :param real: Nreal x nf array containing individual spectrum realizations.
    :param sigma: Uncertainty level on waveform
    :param plot_median: Plot the median spectrum
    :param use_bars: Use error bars instead of fill_between
    """
    
    nt = len(f) 
    xmed, xlow, xhigh = np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for ii in range(nt):
        tmp, xmed[ii] = bu.confinterval(psd[:,ii], onesided=True, sigma=0.5)
        xlow[ii], xhigh[ii] = bu.confinterval(psd[:,ii], sigma=sigma, 
                                              type='minArea')
    
    #print xlow, xhigh
    if use_bars:
        ymean = psd.mean(axis=0)
        ylow = np.log10(10**ymean - 10**xlow)
        yhigh = np.log10(10**xhigh - 10**ymean)
        yerr = np.vstack((ylow, yhigh))
        ax.errorbar(f, 10**ymean, yerr=10**yerr, fmt='o', capsize=0, **kwargs)
    else:
        ax.fill_between(f, 10**xlow, 10**xhigh, **kwargs)
    if plot_median:
        ax.plot(f, 10**xmed, ls='--', lw=1.5, color='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(10**xlow.min()/2, 10**xhigh.max()*2)
    df = f[1] - f[0]
    ax.set_xlim(f.min()-df, f.max()+df)
    ax.grid(which='both')

class ChainPP(object):
    
    def __init__(self, chaindir, h5file=None,
                 jsonfile=None, outdir='noise_output',
                 save=True, nreal=1000):

        ret = get_mcmc_files(chaindir)
        self.chain, self.pars, self.logp = ret[0], ret[1], ret[2]
        if len(ret) == 4:
            self.indicator = ret[3]
        else:
            self.indicator = None
        
        self.ndim = len(self.pars)
        
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass
        
        self.jsonfile = jsonfile
        self.h5file = h5file
        if self.jsonfile is None:
            self.model = None
        else:
            # open JSON file
            with open(self.jsonfile) as f:
                pmodel = OrderedDict(json.load(f))
            pmodel['likfunc'] = 'mark6'
            # load model
            pulsars = map(str, pmodel['pulsarnames'])
            self.model = PALmodels.PTAmodels(self.h5file, pulsars=pulsars)
            self.model.initModel(pmodel, memsave=True, verbose=True)
            p0 = self.model.initParameters()
            self.model.mark6LogLikelihood(
                p0, incCorrelations=False, 
                incJitter=True, varyNoise=True)
            
        self.outdir = outdir
        self.save = save

    def get_quadratic_parameters(self, nreal=1000, maxpars=None):

        ret, real, white = get_quad_posteriors(self.model, self.chain, 
                                               selection=self.indicator, N=nreal, 
                                               add_det=True, fixmask=None, 
                                               maxpars=maxpars)
        return ret, real, white


        
    def get_ml_values(self, ind=None, mtype='full'):
        
        if ind is None:
            ind = np.arange(0, self.ndim)
            
        ind = np.atleast_1d(ind)
            
        x = OrderedDict()
        for i in ind:
            if mtype == 'full':
                x[self.pars[i]] = self.chain[np.argmax(self.logp[:,-1]), i]
            elif mtype == 'marg':
                x[self.pars[i]] = bu.getMax(self.chain[:,i])
            elif mtype == 'mean':
                x[self.pars[i]] = self.chain[:,i].mean()

            
        return x
        
    def plot_lnlike(self):
        
        plt.figure()
        plt.plot(self.logp[:,-2])
        plt.ylabel(r'lnlike')
        
        if self.save:
            plt.savefig(self.outdir + '/lnlike.png',
                        dpi=300, bbox_inches='tight')
        
    def plot_lnpost(self):
        
        plt.figure()
        plt.plot(self.logp[:,-1])
        plt.ylabel(r'lnpost')
        
        if self.save:
            plt.savefig(self.outdir + '/lnpost.png',
                        dpi=300, bbox_inches='tight')
        
    def plot_trace(self, ind=None):
        
        if ind is None:
            ind = np.arange(0, self.ndim)
            
        ind = np.atleast_1d(ind)
        
        ndim = len(ind)
        if len(ind) > 1:
            ncols = 3
            nrows = int(np.ceil(ndim/ncols))
        else:
            ncols, nrows = 1,1
            
        plt.figure(figsize=(15, 2*nrows))
        for ii in range(ndim):
            plt.subplot(nrows, ncols, ii+1)
            plt.plot(self.chain[:,ind[ii]][self.chain[:,ind[ii]]!=0])
            plt.ylabel(self.pars[ind[ii]])
            
        plt.tight_layout()
        
        if self.save:
            plt.savefig(self.outdir + '/trace.png',
                        dpi=300, bbox_inches='tight')
        
    def plot_hist(self, ind=None):
        
        if ind is None:
            ind = np.arange(0, self.ndim)
            
        ind = np.atleast_1d(ind)
        
        ndim = len(ind)
        if len(ind) > 1:
            ncols = 3
            nrows = int(np.ceil(ndim/ncols))
        else:
            ncols, nrows = 1,1
            
        plt.figure(figsize=(15, 2*nrows))
        for ii in range(ndim):
            plt.subplot(nrows, ncols, ii+1)
            plt.hist(self.chain[:,ind[ii]][self.chain[:,ind[ii]]!=0], 
                     50, normed=True, histtype='step')
            plt.xlabel(self.pars[ind[ii]])
            
        plt.tight_layout()
        
        if self.save:
            plt.savefig(self.outdir + '/hist.png',
                        dpi=300, bbox_inches='tight')
        
    def plot_tri(self, ind=None, npar=8, *args, **kwargs):
        
        if ind is None:
            ind = np.arange(0, self.ndim)
            
        ind = np.atleast_1d(ind)
        nplots = int(np.ceil(len(ind) / npar))
        plt.rcParams['font.size'] = 8
        
        start = 0
        for ii in range(nplots):
            ax = bu.triplot(self.chain[:,ind[start:npar+start]], interpolate=True, 
                            labels=list(self.pars[ind[start:npar+start]]), tex=False, 
                            figsize=(24,18), *args, **kwargs)
            start += npar
            
            if self.save:
                plt.savefig(self.outdir + '/tri_{0}.png'.format(ii),
                            dpi=300, bbox_inches='tight')
            
    def plot_spectrum(self, nreal=1000, use_bars=False, plot_median=True, 
                      color='gray', ax=None):
        
        f, psd = get_spectrum(self.model, self.chain, selection=self.indicator, N=nreal)
        if ax is None:
            ax = plt.subplot(111)
        make_spectrum_realization_plot(ax, f, psd, sigma=0.68, use_bars=use_bars, 
                                       plot_median=plot_median, color=color, alpha=0.5)
        ax.set_xlabel(r'Frequency [Hz]')
        ax.set_ylabel(r'Power Spectral Density [s$^2$]')
        if self.save:
            plt.savefig(self.outdir + '/spectrum_{0}.png'.format(self.model.psr[0].name),
                        dpi=300, bbox_inches='tight')

    def plot_fourier_spectrum(self, nreal=1000, maxpars=None, quad=None, 
                              ax=None, use_bars=False, color='gray'):

        plt.figure()
        if quad is None:
            quad, real, white = self.get_quadratic_parameters(
                nreal=nreal, maxpars=maxpars)
        f, psd = get_fourier_spectrum(self.model, quad, N=nreal)
        if ax is None:
            ax = plt.subplot(111)
        make_spectrum_realization_plot(ax, f, psd, sigma=0.68, color=color, 
                                       alpha=0.5, use_bars=use_bars)
        ax.set_xlabel(r'Frequency [Hz]')
        ax.set_ylabel(r'Power Spectral Density [s$^2$]')
        if self.save:
            plt.savefig(self.outdir + '/fourier_spectrum_{0}.png'.format(
                self.model.psr[0].name), dpi=300, bbox_inches='tight')


    def plot_residuals(self, mtype='full', nreal=1000):
        
        if self.model is None:
            raise NotImplementedError('Must input model to make residuals')
        
        x = self.get_ml_values(mtype=mtype)
        p0 = x.values()

        for ct, p in enumerate(self.model.psr):
            #ntmpars = len(p.ptmdescription)
            ntmpars = p.Mmat_reduced.shape[1]
            nf = self.model.npf[ct]
            nfdm = self.model.npfdm[ct]
            idd = np.arange(ntmpars,ntmpars+nf)

            # check for DM
            if nfdm > 0:
                incDM = True
            else:
                incDM = False

            dmmask = None
            if not incDM:
                dmmask = np.zeros(p.Tmat.shape[1], dtype=bool)
                for ii, pp in enumerate(p.ptmdescription):
                    if 'DMX' in pp:
                        dmmask[ii] = True

            ret, real, white = get_quad_posteriors(self.model, self.chain, 
                                                   selection=self.indicator, N=nreal, 
                                                   add_det=True, idd=idd, 
                                                   fixmask=None)

            avetoas, averes, avewhiteres, aveerr, res, whiteres, norm1, aveflags = \
                    get_ave_res(self.model, p0, indicator=self.indicator, 
                                DM=incDM, flags=p.flags)
                
            plt.figure(figsize=(8,7))

            uflags = np.unique(p.flags)
            uflags2 = np.unique(aveflags)
            ax = plt.subplot(311)
            for flag in uflags:
                if flag in uflags2:
                    ind = aveflags == flag
                    ax.errorbar(convert_mjd_to_greg(avetoas[ind]/86400), averes[ind]*1e6, 
                                aveerr[ind]*1e6, fmt='.', 
                                capsize=0, alpha=0.8, label=flag)
                else:
                    ind = p.flags == flag
                    ax.errorbar(convert_mjd_to_greg(p.toas[ind]/86400), res[ind]*1e6, 
                                np.sqrt(p.Nvec)[ind]*1e6, fmt='.', 
                                capsize=0, alpha=0.8, label=flag)

            make_waveform_realization_plot(ax, p, real, sigma=0.68, 
                                           color='gray', alpha=0.5)
            ax.legend(loc='center' ,bbox_to_anchor=(0.5, 1.2), 
                      fontsize=10, numpoints=1, ncol=2)
            ax.grid()
            ax.set_ylabel('Residuals [$\mu s$]')
            
            ax = plt.subplot(312)
            for flag in uflags:
                if flag in uflags2:
                    ind = aveflags == flag
                    ax.errorbar(convert_mjd_to_greg(avetoas[ind]/86400), avewhiteres[ind]*1e6, 
                                aveerr[ind]*1e6, fmt='.', 
                                capsize=0, alpha=0.8, label=flag)
                else:
                    ind = p.flags == flag
                    ax.errorbar(convert_mjd_to_greg(p.toas[ind]/86400), whiteres[ind]*1e6, 
                                np.sqrt(p.Nvec)[ind]*1e6, fmt='.', 
                                capsize=0, alpha=0.8, label=flag)
            ax.grid()
            ax.set_ylabel('Whitened Residuals [$\mu s$]')

            ax = plt.subplot(313)
            make_dm_waveform_realization_plot(ax, p, ret, incDM=incDM, 
                                              color='gray', alpha=0.5)
            ax.set_ylabel('$DM(t)$ [$10^{-3}$ pc cm$^-3$]')
            ax.grid()
             
            if self.save:
                plt.savefig(self.outdir + '/res_{0}.png'.format(p.name),
                            dpi=300, bbox_inches='tight')
                
    def get_t2_noise_pars(self, mtype='full'):
        x = self.get_ml_values(mtype=mtype)
        fout = open(self.outdir + '/t2_noise.txt', 'w')
        for key, val in x.items():
            if 'efac' in key:
                fout.write('TNEF -f {0} {1}\n'.format(
                    key.split('efac-')[-1], val))
            elif 'equad' in key:
                fout.write('TNEQ -f {0} {1}\n'.format(
                    key.split('equad-')[-1], val))
            elif 'jitter_q' in key:
                fout.write('TNECORR -f {0} {1}\n'.format(
                    key.split('jitter_q-')[-1], 10**val*1e6))
            elif 'RN-Amplitude' in key:
                fout.write('TNRedAmp {0}\n'.format(val))
            elif 'RN-spectral-index' in key:
                fout.write('TNRedGam {0}\n'.format(val))
                fout.write('TNRedRedC 50')
                
    def get_t1_noise_pars(self, mtype='full'):
        x = self.get_ml_values(mtype=mtype)
        efacs = {} 
        fout = open(self.outdir + '/t1_noise.txt', 'w')
        for key, val in x.items():
            if 'efac' in key:
                efacs[key.split('efac-')[-1]] = val
                
        for key, val in x.items():
            if 'efac' in key:
                flag = key.split('efac-')[-1]
                fout.write('T2EFAC -f %s %1.4f\n'%(flag, val))
            elif 'equad' in key:
                flag = key.split('equad-')[-1]
                fout.write('T2EQUAD -f %s %1.4f\n'%(
                        flag, (10**val*1e6)/efacs[flag]))
            elif 'jitter_q' in key:
                flag = key.split('jitter_q-')[-1]
                fout.write('ECORR -f  %s %1.4f\n'%(
                    flag, 10**val*1e6))
            elif 'RN-Amplitude' in key:
                fac = (86400.*365.24*1e6)/(2.0*np.pi*np.sqrt(3.0))
                sval = str('%0.4e'%(fac*10**val))
                sval = sval.replace('e', 'D')
                fout.write('RNAMP {0}\n'.format(sval))
            elif 'RN-spectral-index' in key:
                fout.write('RNIDX %2.4f\n'%(-val))
            
        fout.close()


