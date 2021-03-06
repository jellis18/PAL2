from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.ndimage.filters as filter
from PAL2 import PALutils
try:
    import healpy as hp
except ImportError:
    hp = None
try:
    from PAL2 import plot
except ImportError:
    plot = None
import matplotlib.mlab as ml
from matplotlib.ticker import FormatStrFormatter, LinearLocator, NullFormatter, NullLocator
import matplotlib.ticker
import matplotlib.colors
from scipy.stats import gaussian_kde
from optparse import OptionParser
from statsmodels.distributions.empirical_distribution import ECDF
import os

import matplotlib
import distutils.version
mpl_version = distutils.version.LooseVersion(matplotlib.__version__)


"""
Given a 2D matrix of (marginalised) likelihood levels, this function returns
the 1, 2, 3- sigma levels. The 2D matrix is usually either a 2D histogram or a
likelihood scan

"""
def getsigmalevels(hist2d, sig_levels=[0.68, 0.95, 0.997]):
  # We will draw contours with these levels
  sigma1, sigma2, sigma3 = sig_levels
  level1 = 0
  level2 = 0
  level3 = 0

  #
  lik = hist2d.reshape(hist2d.size)
  sortlik = np.sort(lik)

  # Figure out the 1sigma level
  dTotal = np.sum(sortlik)
  nIndex = sortlik.size
  dSum = 0
  while (dSum < dTotal * sigma1):
    nIndex -= 1
    dSum += sortlik[nIndex]
  level1 = sortlik[nIndex]

  # 2 sigma level
  nIndex = sortlik.size
  dSum = 0
  while (dSum < dTotal * sigma2):
    nIndex -= 1
    dSum += sortlik[nIndex]
  level2 = sortlik[nIndex]

  # 3 sigma level
  nIndex = sortlik.size
  dSum = 0
  while (dSum < dTotal * sigma3):
    nIndex -= 1
    dSum += sortlik[nIndex]
  level3 = sortlik[nIndex]

  return level1, level2, level3


def confinterval(samples, sigma=0.68, onesided=False, weights=None,
                 bins=40, type='equalArea'):
    """

    Given a list of samples, return the desired cofidence intervals.
    Returns the minimum and maximum confidence levels

    @param samples: Samples that we wish to get confidence intervals

    @param sigmalevel: Sigma level 1, 2, or 3 sigma, will return
                       corresponding confidence limits

    @param onesided: Boolean to use onesided or twosided confidence
                     limits.

    @param weights: Histogram Weights.

    @param bins: Number of histogram bins

    @param type: equalArea: Integrates from sides of posterior
                 minArea: Brute force search for confidence interval with smallest
                          paramter range
                 equalProb: Integrates from MAP downwards


    """

    ecdf = ECDF(samples)

    # Create the binning
    x = np.linspace(min(samples), max(samples), 1000)
    ecdf = ECDF(samples)
    y = ecdf(x)

    # Find the intervals
    if type == 'equalArea' or onesided:
        if onesided:
            x2max = x[np.flatnonzero(y<=sigma)[-1]]
            x2min = x2max
        else:
            x2min = x[np.flatnonzero(y<=0.5*(1-sigma))[-1]]
            x2max = x[np.flatnonzero(y>=1-0.5*(1-sigma))[0]]

    if type == 'minArea':
        delta, xmin, xmax = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
        start = 0
        for ii in range(len(y)):
            ind = np.flatnonzero((y-y[ii])>=sigma)
            if len(ind) == 0:
                delta[ii] = np.inf
            else:
                delta[ii] = x[ind[0]] - x[ii]
                xmin[ii] = x[ii]
                xmax[ii] = x[ind[0]]

        minind = np.argmin(delta)
        x2min = xmin[minind]
        x2max = xmax[minind]


    if type == 'equalProb' and not(onesided):
        hist, xedges = np.histogram(samples, bins=bins, weights=weights)
        xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])
        x = np.linspace(xedges.min(), xedges.max(), 10000)
        ifunc = interp.interp1d(xedges, hist, kind='linear')
        sortlik = np.sort(ifunc(x))
        sortlik /= sortlik.sum()
        ind = np.argsort(ifunc(x))
        idx = np.flatnonzero(np.cumsum(sortlik) > 1-sigma)
        x2min = x[ind][idx].min()
        x2max = x[ind][idx].max()

    return x2min, x2max

def makesubplot2d(ax, samples1, samples2, cmap=None, color='k', weights=None,
                  smooth=True, bins=[40, 40], contours=True, x_range=None,
                  y_range=None, logx=False, logy=False, logz=False, lw=1.5,
                  conf_levels=[0.68, 0.95, 0.99]):

    if x_range is None:
        xmin = np.min(samples1)
        xmax = np.max(samples1)
    else:
        xmin = x_range[0]
        xmax = x_range[1]

    if y_range is None:
        ymin = np.min(samples2)
        ymax = np.max(samples2)
    else:
        ymin = y_range[0]
        ymax = y_range[1]

    if logx:
        bins[0] = np.logspace(np.log10(xmin), np.log10(xmax), bins[0])

    if logy:
        bins[1] = np.logspace(np.log10(ymin), np.log10(ymax), bins[1])

    hist2d,xedges,yedges = np.histogram2d(samples1, samples2, weights=weights, \
            bins=bins,range=[[xmin,xmax],[ymin,ymax]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]

    if logz:
        for ii in range(hist2d.shape[0]):
            for jj in range(hist2d.shape[1]):
                if hist2d[ii,jj] <= 0:
                    hist2d[ii,jj] = 1


    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])
    yedges = np.delete(yedges, -1) + 0.5*(yedges[1] - yedges[0])

    # gaussian smoothing
    if smooth:
        hist2d = filter.gaussian_filter(hist2d, sigma=0.75)

    if contours:

        level1, level2, level3 = getsigmalevels(hist2d, conf_levels)

        contourlevels = (level1, level2)

        contourcolors = (color, color)
        contourlinestyles = ('-', '--')
        contourlinewidths = (lw, lw)

        # patch to fix new level ordering in mpl v 1.5.1
        if mpl_version >= '1.5.1':
            contourlevels = contourlevels[::-1]
            contourcolors = contourcolors[::-1]
            contourlinestyles = contourlinestyles[::-1]
            contourlinewidths = contourlinewidths[::-1]


        c1 = ax.contour(xedges,yedges,hist2d.T,contourlevels[:2], \
                        colors=contourcolors[:2], linestyles=contourlinestyles[:2], \
                        linewidths=contourlinewidths[:2], zorder=2)
    if cmap:
        if logz:
            c2 = ax.imshow(np.flipud(hist2d.T), extent=extent, aspect=ax.get_aspect(), \
                      interpolation='gaussian', norm=matplotlib.colors.LogNorm(), cmap=cmap)
        else:
            c2 = ax.imshow(np.flipud(hist2d.T), extent=extent, aspect=ax.get_aspect(), \
                      interpolation='gaussian', cmap=cmap)

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

def getMeanAndStd(samples, weights=None, bins=50):
    """
    Get mean and standard deviation. Only really useful when weights != None
    """

    hist, xedges = np.histogram(samples, bins, normed=True, weights=weights)
    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])

    # pdf
    p = hist/np.sum(hist)

    # mean
    m = np.sum(xedges*p)

    # variance
    std = np.sqrt(np.sum(xedges**2*p) - m**2)


    return m, std



def makesubplot1d(ax, samples, weights=None, interpolate=False, smooth=True,\
                  label=None, bins=30, range=None, color='k',
                  orientation='vertical', logbin=False, **kwargs):
    """
    Make histogram of samples

    """

    if range is None:
        hist, xedges = np.histogram(samples, bins, normed=True, weights=weights)
    else:
        hist, xedges = np.histogram(samples, bins, normed=True, range=range, weights=weights)

    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])

    # gaussian smoothing
    if smooth:
        hist = filter.gaussian_filter(hist, sigma=0.75)
        if interpolate:
            f = interp.interp1d(xedges, hist, kind='cubic')
            if logbin:
                xedges = np.logspace(np.log10(xedges.min()),
                                     np.log10(xedges.max()),
                                     10000)
            else:
                xedges = np.linspace(xedges.min(), xedges.max(), 10000)
            hist = f(xedges)

    # make plot
    if label is not None:
        if orientation == 'horizontal':
            ax.plot(hist, xedges, color=color, label=label, **kwargs)
        else:
            ax.plot(xedges, hist, color=color, label=label, **kwargs)
    else:
        if orientation == 'horizontal':
            ax.plot(hist, xedges, color=color, **kwargs)
        else:
            ax.plot(xedges, hist, color=color, **kwargs)

def getMax(samples, weights=None, range=None, bins=50):
    """
    Make histogram of samples

    """

    if range is None:
        hist, xedges = np.histogram(samples, bins, normed=True, weights=weights)
    else:
        hist, xedges = np.histogram(samples, bins, normed=True, range=range,\
                                   weights=weights)

    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])

    # gaussian smoothing
    hist = filter.gaussian_filter(hist, sigma=0.75)

    # interpolation
    f = interp.interp1d(xedges, hist, kind='cubic')
    xedges = np.linspace(xedges.min(), xedges.max(), 10000)
    hist = f(xedges)

    return xedges[np.argmax(hist)]


# make triangle plot of marginalized posterior distribution
def triplot(chain, color='k', weights=None, interpolate=False, smooth=True, \
           labels=None, figsize=(11,8.5), title=None, inj=None, tex=True, \
            incMaxPost=True, cmap='YlOrBr', lw=1.5, ranges=False, axarr=None):

    """

    Make Triangle plot

    """

    # rcParams settings
    if chain.shape[1] < 10:
        ticksize = 10
        #plt.rcParams['ytick.labelsize'] = 10.0
        #plt.rcParams['xtick.labelsize'] = 10.0
    else:
        ticksize = 8
        #plt.rcParams['ytick.labelsize'] = 8.0
        #plt.rcParams['xtick.labelsize'] = 8.0
    if tex:
        plt.rcParams['text.usetex'] = True


    # get number of parameters
    ndim = chain.shape[1]
    parameters = np.arange(ndim, dtype=np.int)

    if axarr is not None:
        f = plt.gcf()
        #fig, axarr = plt.subplots(nrows=len(parameters), ncols=len(parameters),figsize=figsize)
    else:
        f, axarr = plt.subplots(nrows=len(parameters), ncols=len(parameters),figsize=figsize)

    for i in range(len(parameters)):
        # for j in len(parameters[np.where(i <= parameters)]:
        for j in range(len(parameters)):
            ii = i
            jj = len(parameters) - j - 1

            # get ranges
            if ranges:
                xmin, xmax = confinterval(chain[:, parameters[ii]], sigma=0.95,
                                          type='equalProb')
                x_range = [xmin, xmax]
                xmin, xmax = confinterval(chain[:, parameters[jj]], sigma=0.95,
                                          type='equalProb')
                y_range = [xmin, xmax]

            else:
                x_range = [chain[:, parameters[ii]].min(), chain[:, parameters[ii]].max()]
                y_range = [chain[:, parameters[jj]].min(), chain[:, parameters[jj]].max()]


            axarr[ii, jj].tick_params(axis='both', which='major', labelsize=10)

            xmajorLocator = matplotlib.ticker.MaxNLocator(nbins=4,prune='both')
            ymajorLocator = matplotlib.ticker.MaxNLocator(nbins=4,prune='both')

            if j <= len(parameters)-i-1:
                axarr[jj][ii].xaxis.set_minor_locator(NullLocator())
                axarr[jj][ii].yaxis.set_minor_locator(NullLocator())
                axarr[jj][ii].xaxis.set_major_locator(NullLocator())
                axarr[jj][ii].yaxis.set_major_locator(NullLocator())

                axarr[jj][ii].xaxis.set_minor_formatter(NullFormatter())
                axarr[jj][ii].yaxis.set_minor_formatter(NullFormatter())
                axarr[jj][ii].xaxis.set_major_formatter(NullFormatter())
                axarr[jj][ii].yaxis.set_major_formatter(NullFormatter())
                xmajorFormatter = FormatStrFormatter('%g')
                ymajorFormatter = FormatStrFormatter('%g')

                if ii == jj:
                    # Make a 1D plot
                    makesubplot1d(axarr[ii][ii], chain[:,parameters[ii]], \
                                  weights=weights, interpolate=interpolate, \
                                  smooth=smooth, color=color, lw=lw, range=x_range)
                    axarr[ii][jj].set_ylim(ymin=0)
                    if incMaxPost:
                        mx = getMax(chain[:,parameters[ii]], weights=weights)
                        axarr[ii][jj].set_title('%5.4g'%(mx), fontsize=10)

                    if inj is not None:
                        axarr[ii][ii].axvline(inj[ii], lw=2, color='k')
                else:
                    # Make a 2D plot
                    makesubplot2d(axarr[jj][ii], chain[:,parameters[ii]],
                                  chain[:,parameters[jj]], cmap=cmap,
                                  color=color, weights=weights,
                                  smooth=smooth, lw=lw, x_range=x_range,
                                  y_range=y_range)

                    if inj is not None:
                        axarr[jj][ii].plot(inj[ii], inj[jj], 'x', color='k', markersize=12, \
                                           mew=2, mec='k')

                axarr[jj][ii].xaxis.set_major_locator(xmajorLocator)
                axarr[jj][ii].yaxis.set_major_locator(ymajorLocator)
            else:
                axarr[jj][ii].set_visible(False)
                #axarr[jj][ii].axis('off')

            if jj == len(parameters)-1:
                axarr[jj][ii].xaxis.set_major_formatter(xmajorFormatter)
                if labels:
                    axarr[jj][ii].set_xlabel(labels[ii])

            if ii == 0:
                if jj == 0:
                    axarr[jj][ii].yaxis.set_major_locator(NullLocator())
                    #axarr[jj][ii].set_ylabel('Post.')
                else:
                    axarr[jj][ii].yaxis.set_major_formatter(ymajorFormatter)
                    if labels:
                        axarr[jj][ii].set_ylabel(labels[jj])

    # overall plot title
    if title:
        f.suptitle(title, fontsize=14, y=0.90)

    # make plots closer together
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0.1)

    return axarr


def pol2cart(lon, lat):
    """
    Utility function to convert longitude,latitude on a unit sphere to
    cartesian co-ordinates.

    """

    x = np.cos(lat)*np.cos(lon)
    y = np.cos(lat)*np.sin(lon)
    z = np.sin(lat)

    return np.array([x,y,z])


def greedy_bin_sky(skypos, skycarts):
    """

    Greedy binning algorithm

    """

    N = len(skycarts)
    skycarts = np.array(skycarts)
    bins = np.zeros(N)
    for raSample, decSample in skypos:
        sampcart = pol2cart(raSample, decSample)
        dx = np.dot(skycarts, sampcart)
        maxdx = np.argmax(dx)
        bins[maxdx] += 1

    # fill in skymap
    histIndices = np.argsort(bins)[::-1]    # in decreasing order
    NSamples = len(skypos)

    frac = 0.0
    skymap = np.zeros(N)
    for i in histIndices:
        frac = float(bins[i])/float(NSamples)
        skymap[i] = frac

    return skymap


def plotSkyMap(raSample, decSample, nside=16, contours=None, colorbar=True, \
              inj=None, psrs=None, smooth=True, smoothsigma=0.1, cmap='YlOrBr',
               outfile='skymap.pdf', color='k'):
    """

    Plot Skymap of chain samples on Mollwiede projection.

    @param raSample: Array of right ascension samples
    @param decSample: Array of declination  samples
    @param nside: Number of pixels across equator [default = 64]
    @param contours: Confidence contours to draw eg. 68%, 95% etc
                     By default this is set to none and no contours
                     will be drawn.
    @param colorbar: Boolean option to draw colorbar [default = True]
    @param inj: list of injected values [ra, dec] in radians to plot
                [default = None]
    @param psrs: Stacked array of pulsar sky locations [ra, dec] in radians
                 [default=None] Will plot as white diamonds

    """

    # clear figures
    #plt.clf()

    # create stacked array of ra and dec
    skypos = np.column_stack([raSample, decSample])

    npix = hp.nside2npix(nside)    # number of pixels total


    # initialize theta and phi map coordinantes
    skycarts=[]
    for ii in range(npix):
        skycarts.append(np.array(hp.pix2vec(nside,ii)))

    # get skymap values from greedy binning algorithm
    skymap = greedy_bin_sky(skypos, skycarts)

    # smooth skymap
    if smooth:
        skymap = hp.smoothing(skymap, sigma=smoothsigma)

    # make plot
    ax = plt.subplot(111, projection='astro mollweide')

    # Add contours
    if contours is not None:
        for percent in contours:
            indices = np.argsort(-skymap)
            sky = skymap[indices]
            region = np.zeros(skymap.shape)
            ind = np.min(ml.find(np.cumsum(sky) >= 0.01*percent))
            region[indices[0:ind]] = 1.0
            cs = plot.contour(lambda lon, lat: region[hp.ang2pix(nside, 0.5*np.pi - lat, lon)], \
                          colors=color, linewidths=1.0, levels=[0.5])
            #plt.clabel(cs, [0.5], fmt={0.5: '$\mathbf{%d\%%}$' % percent}, fontsize=8, inline=True)

    # plot map
    ax.grid()
    plot.outline_text(ax)
    if cmap is not None:
        plot.healpix_heatmap(skymap, cmap=cmap)

    # add injection
    if inj:
        ax.plot(inj[0], inj[1], 'x', color='k', markersize=8, mew=2, mec='k')

    # add pulsars
    if np.all(psrs):
        ax.plot(psrs[:,0], psrs[:,1], '*', color='lime', markersize=8, mew=1, mec='k')

    # add colorbar and title
    if colorbar:
        plt.colorbar(orientation='horizontal')
        plt.suptitle(r'$p(\alpha,\delta|d)$', y=0.1)

    # save skymap
    plt.savefig(outfile, bbox_inches='tight')



def upperlimitplot2d(x, y, sigma=0.95, ymin=None, ymax=None, bins=40, log=False, \
                     savename=None, labels=None, hold=False, **kwargs):

    """

    Make upper limits of a parameter as a function of another.

    @param x: Parameter we are making upper limits for
    @param y: Parameter which we will bin
    @param sigma: Sigma level of upper limit
    @param ymin: Minimum value of binning parameter [default=None]
    @param ymax: Maximum value of binning parameter [default=None]
    @param bins: Number of bins
    @param log: If True, plot on log-log scale
    @param savename: Output filename for saved figure
    @param labels: List of labels for axes [xlabel, ylabel]
    @param hold: Hold current figure?

    """

    # clear current figure
    if hold == False:
        plt.clf()

    if ymin is None:
        ymin = y.min()
    if ymax is None:
        ymax = y.max()

    yedges = np.linspace(ymin, ymax, bins+1)
    deltay = yedges[1] - yedges[0]
    yvals = np.linspace(ymin+0.5*deltay, ymax-0.5*deltay, bins)
    bin_index = []
    upper = []

    for i in range(bins):
        # Obtain the indices in the range of the bin
        indices = np.flatnonzero(np.logical_and(y>yedges[i], y<yedges[i+1]))

        # Obtain the 1-sided x-sigma upper limit
        if len(indices) > 0:
            bin_index.append(i)
            a, sigma1 = confinterval(x[indices], sigma=sigma, onesided=True)
            upper.append(sigma1)

    # make bin_indes and upper into arrays
    bin_index = np.array(bin_index)
    upper = np.array(upper)

    # make plot
    if log:
        plt.loglog(10**yvals[bin_index], 10**upper, **kwargs)
        plt.grid(which='major')
        plt.grid(which='minor')
    else:
        plt.plot(yvals[bin_index], upper, **kwargs)
        plt.grid()

    # labels
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    if savename:
        plt.savefig(savename, bbox_inches='tight')
    else:
        plt.savefig('2dUpperLimit.pdf', bbox_inches='tight')

        """
Given an mcmc chain, plot the log-spectrum

"""
def makespectrumplot(ax, chain, parstart=1, numfreqs=10, freqs=None, \
        Apl=None, gpl=None, Asm=None, asm=None, fcsm=0.1, plotlog=False, \
        lcolor='black', Tmax=None, Aref=None, title=None, \
        values=False):

    if freqs is None:
        ufreqs = np.log10(np.arange(1, 1+numfreqs))
    else:
        ufreqs = np.log10(np.sort(np.array(list(set(freqs)))))

    #ufreqs = np.array(list(set(freqs)))
    yval = np.zeros(len(ufreqs))
    yerr = np.zeros(len(ufreqs))

    if len(ufreqs) != (numfreqs):
        print "WARNING: parameter range does not correspond to #frequencies"

    for ii in range(numfreqs):
        fmin, fmax = confinterval(chain[:, parstart+ii], sigma=0.68)
        yval[ii] = (fmax + fmin) * 0.5
        yerr[ii] = (fmax - fmin) * 0.5

    retvals = []
    if values:
        retvals.append(yval)
        retvals.append(yerr)


    # For plotting reference spectra
    pfreqs = 10 ** ufreqs
    ypl = None
    ysm = None

    if plotlog:
        pic_spy = 3.16e7
        ax.errorbar(ufreqs, yval, yerr=yerr, fmt='.', c=lcolor)
        # outmatrix = np.array([ufreqs, yval, yerr]).T
        # np.savetxt('spectrumplot.txt', outmatrix)

        if Apl is not None and gpl is not None and Tmax is not None:
            Apl = 10**Apl
            ypl = (Apl**2 * pic_spy**3 / (12*np.pi*np.pi * (Tmax))) * ((pfreqs * pic_spy) ** (-gpl))
            ax.plot(np.log10(pfreqs), np.log10(ypl), 'g--', linewidth=2.0)

        if Asm is not None and asm is not None and Tmax is not None:
            Asm = 10**Asm
            fcsm = fcsm / pic_spy
            ysm = (Asm * pic_spy**3 / Tmax) * ((1 + (pfreqs/fcsm)**2)**(-0.5*asm))
            ax.plot(np.log10(pfreqs), np.log10(ysm), 'r--', linewidth=2.0)


        #plt.axis([np.min(ufreqs)-0.1, np.max(ufreqs)+0.1, np.min(yval-yerr)-1, np.max(yval+yerr)+1])
        ax.set_xlabel("Frequency [log(f/Hz)]")
        #if True:
        #    #freqs = likobhy.ptapsrs[0].Ffreqs
        #    Tmax = 156038571.88061461
        #    Apl = 10**-13.3 ; Asm = 10**-24
        #    apl = 4.33 ; asm = 4.33
        #    fc = (10**-1.0)/pic_spy

        #    pcsm = (Asm * pic_spy**3 / Tmax) * ((1 + (freqs/fc)**2)**(-0.5*asm))
        #    pcpl = (Apl**2 * pic_spy**3 / (12*np.pi*np.pi * Tmax)) * \
        #    (freqs*pic_spy) ** (-apl)
        #    plt.plot(np.log10(freqs), np.log10(pcsm), 'r--', linewidth=2.0)
        #    plt.plot(np.log10(freqs), np.log10(pcpl), 'g--', linewidth=2.0)

    else:
        ax.errorbar(10**ufreqs, yval, yerr=yerr, fmt='.', c='black')
        if Aref is not None:
            ax.plot(10**ufreqs, np.log10(yinj), 'k--')
        plt.axis([np.min(10**ufreqs)*0.9, np.max(10**ufreqs)*1.01, np.min(yval-yerr)-1, np.max(yval+yerr)+1])
        plt.xlabel("Frequency [Hz]")

    #plt.title("Power spectrum")
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel("Power Spectrum [s^2]")
    plt.grid(True)

    return retvals

def makePostPlots(chain, labels, outDir='./postplots'):

    import acor

    if not os.path.exists(outDir):
        try:
            os.makedirs(outDir)
        except OSError:
            pass


    ndim = chain.shape[1]
    for ii in range(ndim):

        xmajorLocator = matplotlib.ticker.MaxNLocator(nbins=6,prune='both')
        ymajorLocator = matplotlib.ticker.MaxNLocator(nbins=6,prune='both')

        fig = plt.figure(figsize=(10,4))

        ax = fig.add_subplot(121)
        acl = acor.acor(chain[:,ii])[0]
        neff = len(chain[:,ii]) / acl * 10
        ax.plot(chain[:,ii])
        plt.title('Neff = {0}'.format(int(neff)))
        plt.ylabel(labels[ii])

        ax = fig.add_subplot(122)
        if 'equad' in labels[ii] or 'jitter' in labels[ii] or \
           'Amplitude' in labels[ii]:
            ax.hist(10**chain[:,ii], 50, lw=2, color='b', \
                    weights=10**chain[:,ii], normed=True)
        else:
            ax.hist(chain[:,ii], 50, lw=2, color='b', normed=True)
        plt.xlabel(labels[ii])
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)

        plt.savefig(outDir + '/' + labels[ii] + '_post.png', bbox_inches='tight', \
                   dpi=200)


def makeSkyMap(samples, lmax, nside=16, psrs=None, cmap='YlOrBr'):

    # number of pixels total
    npix = hp.nside2npix(nside)

    # initialize theta and phi map coordinantes
    skypos=[]
    for ii in range(npix):
        skypos.append(np.array(hp.pix2ang(nside,ii)))

    skypos = np.array(skypos)
    harmvals = PALutils.SetupSkymapPlottingGrid(lmax,skypos)

    pwr = []
    for ii in range(len(samples)):
        samples_tot = np.append(2.*np.sqrt(np.pi), samples[ii])
        gwp = PALutils.GWpower(samples_tot, harmvals)
        pwr.append(gwp)

    pwr = np.array(pwr)
    pwr = np.mean(pwr, axis=0)

    ax = plt.subplot(111, projection='astro mollweide')
    ax.grid()
    plot.outline_text(ax)
    plot.healpix_heatmap(pwr, cmap=cmap)
    plt.colorbar(orientation='horizontal')

    # add pulsars locations
    if np.all(psrs):
        ax.plot(psrs[:,0], psrs[:,1], '*', color='w', markersize=6, mew=1, mec='w')

    return pwr
