from __future__ import division
import numpy as np
import scipy.linalg as sl
import scipy.special as ss
from scipy import integrate
from scipy.optimize import minimize_scalar
import PALutils
import time
import sys,os

########################## DETECTION STATISTICS ##################################

# compute f_p statistic
def fpStat(psr, f0):
    """ 
    Computes the Fp-statistic as defined in Ellis, Siemens, Creighton (2012)
    
    @param psr: List of pulsar object instances
    @param f0: Gravitational wave frequency

    @return: Value of the Fp statistic evaluated at f0

    """

    fstat=0.
    npsr = len(psr)

    # define N vectors from Ellis et al, 2012 N_i=(x|A_i) for each pulsar
    N = np.zeros(2)
    M = np.zeros((2, 2))
    for ii,p in enumerate(psr):

        # Define A vector
        A = np.zeros((2, p.ntoa))
        A[0,:] = 1./f0**(1./3.) * np.sin(2*np.pi*f0*p.toas)
        A[1,:] = 1./f0**(1./3.) * np.cos(2*np.pi*f0*p.toas)

        N = np.array([np.dot(A[0,:], np.dot(p.invCov, p.res)), \
                      np.dot(A[1,:], np.dot(p.invCov, p.res))]) 
        
        # define M matrix M_ij=(A_i|A_j)
        for jj in range(2):
            for kk in range(2):
                M[jj,kk] = np.dot(A[jj,:], np.dot(p.invCov, A[kk,:]))
        
        # take inverse of M
        Minv = np.linalg.inv(M)
        fstat += 0.5 * np.dot(N, np.dot(Minv, N))

    # return F-statistic
    return fstat

def fpstatAmbiguity(psr, f1, f2):
    """
    
    """

    npsr = len(psr)

    # define N vectors from Ellis et al, 2012 N_i=(x|A_i) for each pulsar
    M1 = np.zeros((2, 2))
    M2 = np.zeros((2, 2))
    M_mixed = np.zeros((2, 2))
    tmp1 = 0
    tmp2 = 0
    for ii,p in enumerate(psr):

        # Define A vector
        A1 = np.zeros((2, p.ntoa))
        A1[0,:] = 1./f1**(1./3.) * np.sin(2*np.pi*f1*p.toas)
        A1[1,:] = 1./f1**(1./3.) * np.cos(2*np.pi*f1*p.toas)
        
        A2 = np.zeros((2, p.ntoa))
        A2[0,:] = 1./f2**(1./3.) * np.sin(2*np.pi*f2*p.toas)
        A2[1,:] = 1./f2**(1./3.) * np.cos(2*np.pi*f2*p.toas)

        # define M matrix M_ij=(A_i|A_j)
        for jj in range(2):
            for kk in range(2):
                M1[jj,kk] = np.dot(A1[jj,:], np.dot(p.invCov, A1[kk,:]))
                M2[jj,kk] = np.dot(A2[jj,:], np.dot(p.invCov, A2[kk,:]))
                M_mixed[jj,kk] = np.dot(A1[jj,:], np.dot(p.invCov, A2[kk,:]))
        
        # take inverse of M
        Minv1 = np.linalg.inv(M1)
        Minv2 = np.linalg.inv(M2)

        tmp1 += np.dot(Minv1, M_mixed)
        tmp2 += np.dot(Minv2, M_mixed)

    amb = np.trace(np.dot(tmp1, tmp2))
    
    return amb - (2*npsr)**2 




def marginalizedPulsarPhaseLike(psr, theta, phi, phase, inc, psi, freq, h, maximize=False):
    """ 
    Compute the log-likelihood marginalized over pulsar phases

    @param psr: List of pulsar object instances
    @param theta: GW polar angle [radian]
    @param phi: GW azimuthal angle [radian]
    @param phase: Initial GW phase [radian]
    @param inc: GW inclination angle [radian]
    @param psi: GW polarization angle [radian]
    @param freq: GW initial frequency [Hz]
    @param h: GW strain
    @param maximize: Option to maximize over pulsar phases instead of marginalize

    """

    # get number of pulsars
    npsr = len(psr)
       
    # get c and d
    c = np.cos(phase)
    d = np.sin(phase)

    # construct xi = M**5/3/D and omega
    xi = 0.25 * np.sqrt(5/2) * (np.pi*freq)**(-2/3) * h
    omega = np.pi*freq

    lnlike = 0
    for ct, pp in enumerate(psr):

        # compute relevant inner products
        cip = np.dot(np.cos(2*omega*pp.toas), np.dot(pp.invCov, pp.res)) 
        sip = np.dot(np.sin(2*omega*pp.toas), np.dot(pp.invCov, pp.res))
        N = np.dot(np.cos(2*omega*pp.toas), np.dot(pp.invCov, np.cos(2*omega*pp.toas)))

        # compute fplus and fcross
        fplus, fcross, cosMu = PALutils.createAntennaPatternFuncs(pp, theta, phi)

        # mind your p's and q's
        p = (1+np.cos(inc)**2) * (fplus*np.cos(2*psi) + fcross*np.sin(2*psi))
        q = 2*np.cos(inc) * (fplus*np.sin(2*psi) - fcross*np.cos(2*psi))

        # construct X Y and Z
        X = -xi/omega**(1/3) * (p*sip + q*cip - 0.5*xi/omega**(1/3)*N*c*(p**2+q**2))
        Y = -xi/omega**(1/3) * (q*sip - p*cip - 0.5*xi/omega**(1/3)*N*d*(p**2+q**2))
        Z = xi/omega**(1/3) * ((p*c+q*d)*sip - (p*d-q*c)*cip \
                        -0.5*xi/omega**(1/3)*N*(p**2+q**2))

        # add to log-likelihood
        #print X, Y
        if maximize:
            lnlike += Z + np.sqrt(X**2 + Y**2)
        else:
            lnlike += Z + np.log(ss.iv(0, np.sqrt(X**2 + Y**2)))

    return lnlike

def marginalizedPulsarPhaseLikeNumerical(psr, theta, phi, phase, inc, psi, freq, h,\
                                         maximize=False):
    """ 
    Compute the log-likelihood marginalized over pulsar phases

    @param psr: List of pulsar object instances
    @param theta: GW polar angle [radian]
    @param phi: GW azimuthal angle [radian]
    @param phase: Initial GW phase [radian]
    @param inc: GW inclination angle [radian]
    @param psi: GW polarization angle [radian]
    @param freq: GW initial frequency [Hz]
    @param h: GW strain
    @param maximize: Option to maximize over pulsar phases instead of marginalize

    """

    tstart = time.time()

    # get number of pulsars
    npsr = len(psr)
       
    # construct xi = M**5/3/D and omega
    xi = 0.25 * np.sqrt(5/2) * (np.pi*freq)**(-2/3) * h
    omega = np.pi*freq
    
    # get a values from Ellis et al 2012
    a1 = xi * ((1+np.cos(inc)**2)*np.cos(phase)*np.cos(2*psi) + \
               2*np.cos(inc)*np.sin(phase)*np.sin(2*psi))
    a2 = -xi * ((1+np.cos(inc)**2)*np.sin(phase)*np.cos(2*psi) - \
                2*np.cos(inc)*np.cos(phase)*np.sin(2*psi))
    a3 = xi * ((1+np.cos(inc)**2)*np.cos(phase)*np.sin(2*psi) - \
               2*np.cos(inc)*np.sin(phase)*np.cos(2*psi))
    a4 = -xi * ((1+np.cos(inc)**2)*np.sin(phase)*np.sin(2*psi) + \
                2*np.cos(inc)*np.cos(phase)*np.cos(2*psi))

    lnlike = 0
    tip = 0
    tint = 0
    tmax = 0
    for ct, pp in enumerate(psr):

        tstartip = time.time()

        # compute relevant inner products
        N1 = np.dot(np.cos(2*omega*pp.toas), np.dot(pp.invCov, pp.res)) 
        N2 = np.dot(np.sin(2*omega*pp.toas), np.dot(pp.invCov, pp.res))
        M11 = np.dot(np.sin(2*omega*pp.toas), np.dot(pp.invCov, np.sin(2*omega*pp.toas)))
        M22 = np.dot(np.cos(2*omega*pp.toas), np.dot(pp.invCov, np.cos(2*omega*pp.toas)))
        M12 = np.dot(np.cos(2*omega*pp.toas), np.dot(pp.invCov, np.sin(2*omega*pp.toas)))

        # compute fplus and fcross
        fplus, fcross, cosMu = PALutils.createAntennaPatternFuncs(pp, theta, phi)

        # mind your p's and q's
        p = fplus*a1 + fcross*a3
        q = fplus*a2 + fcross*a4

        # constuct multipliers of pulsar phase terms
        X = p*N1 + q*N2 + p**2*M11 + q**2*M22 + 2*p*q*M12
        Y = p*N1 + q*N2 + 2*p**2*M11 + 2*q**2*M22 + 4*p*q*M12
        Z = p*N2 - q*N1 + 2*(p**2-q**2)*M12 - 2*p*q*(M11-M22)
        W = q**2*M11 + p**2*M22 -2*p*q*M12
        V = p*q*(M11-M22) - (p**2-q**2)*M12
        
        #print X, Y, Z, W, V
        tip += (time.time() - tstartip)

        tstartint = time.time()

        # find the maximum of argument of exponential function
        phip = np.linspace(0, 2*np.pi, 10000)
        arg = X - Y*np.cos(phip) + Z*np.sin(phip) + W*np.sin(phip)**2 + 2*V*np.cos(phip)*np.sin(phip)
        maxarg = np.max(arg)

        if maximize:
            tmax += maxarg

        else:

            # define integrand for numerical integration
            f = lambda phi: np.exp(X - Y*np.cos(phi) + Z*np.sin(phi) + \
                    W*np.sin(phi)**2 + 2*V*np.cos(phi)*np.sin(phi) - maxarg)

            # do numerical integration
            integral = integrate.quad(f, 0, 2*np.pi)[0]
            lnlike += maxarg + np.log(integral)

            tint += (time.time() - tstartint)

    print 'Loglike = {0}'.format(lnlike)
    print 'Total Evaluation Time = {0} s'.format(time.time() - tstart)
    print 'Total inner product evaluation Time = {0} s'.format(tip)
    print 'Total Integration Time = {0} s\n'.format(tint)

    if maximize:
        lnlike = tmax

    return lnlike


def optStat(psr, ORF, gam=4.33333):
    """
    Computes the Optimal statistic as defined in Chamberlin, Creighton, Demorest et al (2013)

    @param psr: List of pulsar object instances
    @param ORF: Vector of pairwise overlap reduction values
    @param gam: Power Spectral index of GBW (default = 13/3, ie SMBMBs)

    @return: Opt: Optimal statistic value (A_gw^2)
    @return: sigma: 1-sigma uncertanty on Optimal statistic
    @return: snr: signal-to-noise ratio of cross correlations

    """

    #TODO: maybe compute ORF in code instead of reading it in. Would be less
    # of a risk but a bit slower...

    k = 0
    npsr = len(psr)
    top = 0
    bot = 0
    for ll in xrange(0, npsr):
        for kk in xrange(ll+1, npsr):

            # form matrix of toa residuals and compute SigmaIJ
            tm = PALutils.createTimeLags(psr[ll].toas, psr[kk].toas)

            # create cross covariance matrix without overall amplitude A^2
            SIJ = ORF[k]/2 * PALutils.createRedNoiseCovarianceMatrix(tm, 1, gam)
            
            # construct numerator and denominator of optimal statistic
            bot += np.trace(np.dot(psr[ll].invCov, np.dot(SIJ, np.dot(psr[kk].invCov, SIJ.T))))
            top += np.dot(psr[ll].res, np.dot(psr[ll].invCov, np.dot(SIJ, \
                        np.dot(psr[kk].invCov, psr[kk].res))))
            k+=1

    # compute optimal statistic
    Opt = top/bot
    
    # compute uncertainty
    sigma = 1/np.sqrt(bot)

    # compute SNR
    snr = top/np.sqrt(bot)

    # return optimal statistic and snr
    return Opt, sigma, snr

def crossPower(psr, gam=13/3):
    """

    Compute the cross power as defined in Eq 9 and uncertainty of Eq 10 in 
    Demorest et al (2012).

    @param psr: List of pulsar object instances
    @param gam: Power spectral index of GWB

    @return: vector of cross power for each pulsar pair
    @return: vector of cross power uncertainties for each pulsar pair

    """

    # initialization
    npsr = len(psr) 

    # now compute cross power
    rho = []
    sig = []
    xi = []
    for ll in range(npsr):
        for kk in range(ll+1, npsr):
            
            # matrix of time lags
            tm = PALutils.createTimeLags(psr[ll].toas, psr[kk].toas)

            # create cross covariance matrix without overall amplitude A^2
            SIJ = PALutils.createRedNoiseCovarianceMatrix(tm, 1, gam)
            
            # construct numerator and denominator of optimal statistic
            bot = np.trace(np.dot(psr[ll].invCov, np.dot(SIJ, np.dot(psr[kk].invCov, SIJ.T))))
            top = np.dot(psr[ll].res, np.dot(psr[ll].invCov, np.dot(SIJ, \
                        np.dot(psr[kk].invCov, psr[kk].res))))

            # cross correlation and uncertainty
            rho.append(top/bot)
            sig.append(1/np.sqrt(bot))


    return np.array(rho), np.array(sig)


#def crossPower(psr):
#    """
#
#    Compute the cross power as defined in Eq 9 and uncertainty of Eq 10 in 
#    Demorest et al (2012).
#
#    @param psr: List of pulsar object instances
#
#    @return: vector of cross power for each pulsar pair
#    @return: vector of cross power uncertainties for each pulsar pair
#
#    """
#
#    # initialization
#    npsr = len(psr) 
#
#
#    for ii in range(npsr):
#
#        # matrix of time lags
#        tm = PALutils.createTimeLags(psr[ii].toas, psr[ii].toas)
#
#        # red noise covariance matrix
#        Cgw = PALutils.createRedNoiseCovarianceMatrix(tm, 1, 13/3)
#        Cgw = np.dot(psr[ii].G.T, np.dot(Cgw, psr[ii].G))
#
#        # white noise covariance matrix
#        white = PALutils.createWhiteNoiseCovarianceMatrix(psr[ii].err, 1, 0)
#        white = np.dot(psr[ii].G.T, np.dot(white, psr[ii].G))
#
#        # chlolesky decomposition of white noise
#        L = sl.cholesky(white)
#        Linv = np.linalg.inv(L)
#
#        # sandwich with Linv
#        Cgwnew = np.dot(Linv, np.dot(Cgw, Linv.T))
#
#        # get svd of matrix
#        u, s, v = sl.svd(Cgwnew)
#
#        # data written in new basis
#        c = np.dot(u.T, np.dot(Linv, np.dot(psr[ii].G.T, psr[ii].res)))
#
#        # obtain the maximum likelihood value of Agw
#        f = lambda x: -PALutils.twoComponentNoiseLike(x, s, c)
#        fbounded = minimize_scalar(f, bounds=(0, 1e-14, 3.0e-13), method='Golden')
#
#        # maximum likelihood value
#        hc_ml = np.abs(fbounded.x)
#        print 'Max like Amp = {0}'.format(hc_ml)
#
#        # create inverse covariance matrix from svd decomposition
#        tmp = hc_ml**2 * Cgw + white
#        #psr[ii].invCov = np.dot(psr[ii].G, np.dot(sl.inv(tmp), psr[ii].G.T))
#
#    # now compute cross power
#    rho = []
#    sig = []
#    xi = []
#    for ll in range(npsr):
#        for kk in range(ll+1, npsr):
#            
#            # matrix of time lags
#            tm = PALutils.createTimeLags(psr[ll].toas, psr[kk].toas)
#
#            # create cross covariance matrix without overall amplitude A^2
#            SIJ = PALutils.createRedNoiseCovarianceMatrix(tm, 1, 13/3)
#            
#            # construct numerator and denominator of optimal statistic
#            bot = np.trace(np.dot(psr[ll].invCov, np.dot(SIJ, np.dot(psr[kk].invCov, SIJ.T))))
#            top = np.dot(psr[ll].res, np.dot(psr[ll].invCov, np.dot(SIJ, \
#                        np.dot(psr[kk].invCov, psr[kk].res))))
#
#            # cross correlation and uncertainty
#            rho.append(top/bot)
#            sig.append(1/np.sqrt(bot))
#
#            # angular separation
#            xi.append(PALutils.angularSeparation(psr[ll].theta, psr[ll].phi, \
#                                                psr[kk].theta, psr[kk].phi))
#    
#
#    return np.array(rho), np.array(sig), np.array(xi)
#

            
    

######################### BAYESIAN LIKELIHOOD FUNCTIONS ####################################


def firstOrderLikelihood(psr, ORF, Agw, gamgw, Ared, gred, efac, equad, \
                        interpolate=False):
    """
    Compute the value of the first-order likelihood as defined in 
    Ellis, Siemens, van Haasteren (2013).

    @param psr: List of pulsar object instances
    @param ORF: Vector of pairwise overlap reduction values
    @param Agw: Amplitude of GWB in standard strain amplitude units
    @param gamgw: Power spectral index of GWB
    @param Ared: Vector of amplitudes of intrinsic red noise in GWB strain units
    @param gamgw: Vector of power spectral index of red noise
    @param efac: Vector of efacs 
    @param equad: Vector of equads
    @param interpolate: Boolean to perform interpolation only with compressed
                        data. (default = False)

    @return: Log-likelihood value

    """
    npsr = len(psr)
    loglike = 0
    tmp = []

    # start loop to evaluate auto-terms
    for ll in range(npsr):

       r1 = np.dot(psr[ll].G.T, psr[ll].res)

       # create time lags
       tm = PALutils.createTimeLags(psr[ll].toas, psr[ll].toas)

       #TODO: finish option to do interpolation when using compression

       # calculate auto GW covariance matrix
       SC = PALutils.createRedNoiseCovarianceMatrix(tm, Agw, gamgw)

       # calculate auto red noise covariance matrix
       SA = PALutils.createRedNoiseCovarianceMatrix(tm, Ared[ll], gred[ll])

       # create white noise covariance matrix
       #TODO: add ability to use multiple efacs for different backends
       white = PALutils.createWhiteNoiseCovarianceMatrix(psr[ll].err, efac[ll], equad[ll])

       # total auto-covariance matrix
       P = SC + SA + white

       # sandwich with G matrices
       Ppost = np.dot(psr[ll].G.T, np.dot(P, psr[ll].G))

       # do cholesky solve
       cf = sl.cho_factor(Ppost)

       # solution vector P^_1 r
       rr = sl.cho_solve(cf, r1)

       # temporarily store P^-1 r
       tmp.append(np.dot(psr[ll].G, rr))

       # add to log-likelihood
       loglike  += -0.5 * (np.sum(np.log(2*np.pi*np.diag(cf[0])**2)) + np.dot(r1, rr))

 
    # now compute cross terms
    k = 0
    for ll in range(npsr):
        for kk in range(ll+1, npsr):

            # create time lags
            tm = PALutils.createTimeLags(psr[ll].toas, psr[kk].toas)

            # create cross covariance matrix
            SIJ = PALutils.createRedNoiseCovarianceMatrix(tm, 1, gamgw)

            # carry out matrix-vetor operations
            tmp1 = np.dot(SIJ, tmp[kk])

            # add to likelihood
            loglike += ORF[k]/2 * Agw**2 * np.dot(tmp[ll], tmp1)
            
            # increment ORF counter
            k += 1

    return loglike

def lentatiMarginalizedLikeCoarse(psr, F, s, U, rho, efac, equad, cequad):
    """
    Lentati marginalized likelihood function only including efac and equad

    @param psr: Pulsar class
    @param F: Fourier design matrix constructed in PALutils
    @param s: diagonalized white noise matrix
    @param U: exploder matrix
    @param rho: Power spectrum coefficients
    @param efac: constant multipier on error bar covaraince matrix term
    @param equad: Additional white noise added in quadrature to efac
    @param equad: coarse grained equad

    @return: LogLike: loglikelihood

    """

    # compute d
    d = np.dot(U.T, psr.res/(efac**2*s + equad**2))

    # compute X
    N = 1/(efac**2*s + equad**2)
    right = (N*U.T).T
    X = np.dot(U.T, right)

    if np.any(rho == -np.inf):
          
        logdet_N = np.sum(np.log(2*np.pi*(efac**2*s + equad**2)))
        logdet_Q = np.sum(np.log(np.ones(len(d))*cequad**2))
        Sigma = np.eye(len(d))/cequad**2 + X 
        cf = sl.cho_factor(Sigma)
        expval2 = sl.cho_solve(cf, d)
        logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))
        dtNdt = np.sum(psr.res**2/(efac**2*s + equad**2))
        logLike = -0.5 * (logdet_N + logdet_Q + logdet_Sigma + dtNdt - np.dot(d, expval2))

    else:

        # compute Fq
        Fq = F/cequad**2

        # compute Sigma
        FtQF = np.dot(F.T, F) / cequad**2

        arr = np.zeros(2*len(rho))
        arr[0::2] = rho
        arr[1::2] = rho
      
        Sigma = FtQF + np.diag(1/10**arr)

        # sigma inverse
        cf = sl.cho_factor(Sigma)
        Sigma_inv = sl.cho_solve(cf, np.eye(Sigma.shape[0]))
        logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

        # total array to invert
        tot = np.diag(np.ones(len(d))/cequad**2) - np.dot(Fq, np.dot(Sigma_inv, Fq.T)) + X

        # first exponential term
        expval1 = np.sum(psr.res**2/(efac**2*s + equad**2))

        # cholesky decomp for second term in exponential
        try:
            cf = sl.cho_factor(tot)
            expval2 = sl.cho_solve(cf, d)
            logdet_tot = np.sum(2*np.log(np.diag(cf[0])))

        except np.linalg.LinAlgError:
            #return -np.inf
            print 'Cholesky Decomposition Failed!! Using SVD instead'
            u,ss,v = sl.svd(tot)
            expval2 = np.dot(u, 1/ss*np.dot(u.T, d))
            logdet_tot = np.sum(np.log(ss))

        logdet_Phi = np.sum(np.log(10**arr))

        logdet_N = np.sum(np.log(2*np.pi*(efac**2*s + equad**2)))

        logdet_Q = np.sum(np.log(np.ones(len(d))*cequad**2))

        dtNdt = np.sum(psr.res**2/(efac**2*s + equad**2))

        logLike = -0.5 * (logdet_N + logdet_Phi + logdet_Sigma + logdet_tot + logdet_Q)\
                        - 0.5 * (expval1 - np.dot(d, expval2))

    #print logdet_Sigma, logdet_Phi, W**2*np.dot(d, expval2)
  

    return logLike


def lentatiMarginalizedLikeCoarse2(psr, F, s, U, rho, efac, equad, cequad):
    """
    Lentati marginalized likelihood function only including efac and equad

    @param psr: Pulsar class
    @param F: Fourier design matrix constructed in PALutils
    @param s: diagonalized white noise matrix
    @param U: exploder matrix
    @param rho: Power spectrum coefficients
    @param efac: constant multipier on error bar covaraince matrix term
    @param equad: Additional white noise added in quadrature to efac
    @param equad: coarse grained equad

    @return: LogLike: loglikelihood

    """

    # compute d
    d = np.dot(U.T, psr.res/(efac**2*s + equad**2))

    # compute X
    N = 1/(efac**2*s + equad**2)
    right = (N*U.T).T
    X = np.dot(U.T, right)

    if np.any(rho == -np.inf):
          
        logdet_N = np.sum(np.log(2*np.pi*(efac**2*s + equad**2)))
        logdet_Q = np.sum(np.log(np.ones(len(d))*cequad**2))
        Sigma = np.eye(len(d))/cequad**2 + X 
        cf = sl.cho_factor(Sigma)
        expval2 = sl.cho_solve(cf, d)
        logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))
        dtNdt = np.sum(psr.res**2/(efac**2*s + equad**2))
        logLike = -0.5 * (logdet_N + logdet_Q + logdet_Sigma + dtNdt - np.dot(d, expval2))

    else:

        arr = np.zeros(2*len(rho))
        arr[0::2] = rho
        arr[1::2] = rho

        # compute Y
        right = (10**arr*F).T
        Y = np.dot(F, right) + np.diag(np.ones(len(d))*cequad**2)

        # get determinant and inverse of Y
        try:
            cf = sl.cho_factor(Y)
            Yinv = sl.cho_solve(cf, np.eye(Y.shape[0]))
            logdet_Y = np.sum(2*np.log(np.diag(cf[0])))
        except np.linalg.LinAlgError:
            #return -np.inf
            print 'Cholesky Decomposition Failed!! Using SVD instead'
            u,ss,v = sl.svd(Y)
            right = (1/ss*u).T
            Yinv = np.dot(u, right)
            logdet_Y = np.sum(np.log(ss))


        # compute Sigma
        Sigma = X + Yinv

        # first exponential term
        expval1 = np.sum(psr.res**2/(efac**2*s + equad**2))

        # cholesky decomp for second term in exponential
        try:
            cf = sl.cho_factor(Sigma)
            expval2 = sl.cho_solve(cf, d)
            logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

        except np.linalg.LinAlgError:
            #return -np.inf
            print 'Cholesky Decomposition Failed!! Using SVD instead'
            u,ss,v = sl.svd(Sigma)
            right = (1/ss*u).T
            expval2 = np.dot(u, np.dot(right, d))
            logdet_Sigma = np.sum(np.log(ss))

        logdet_N = np.sum(np.log(2*np.pi*(efac**2*s + equad**2)))

        logLike = -0.5 * (logdet_N + logdet_Y + logdet_Sigma)\
                        - 0.5 * (expval1 - np.dot(d, expval2))

    #print logdet_Sigma, logdet_Phi, W**2*np.dot(d, expval2)
  

    return logLike




def lentatiMarginalizedLike(psr, F, s, rho, efac, equad):
    """
    Lentati marginalized likelihood function only including efac and equad

    @param psr: Pulsar class
    @param F: Fourier design matrix constructed in PALutils
    @param s: diagonalized white noise matrix
    @param rho: Power spectrum coefficients
    @param efac: constant multipier on error bar covaraince matrix term
    @param equad: Additional white noise added in quadrature to efac

    @return: LogLike: loglikelihood

    """
    
    if np.any(rho == -np.inf):
          
        logdet_N = np.sum(np.log(2*np.pi*(efac**2*s + equad**2)))
        dtNdt = np.sum(psr.res**2/(efac**2*s + equad**2))
        logLike = -0.5 * (logdet_N + dtNdt)

    else:
    
        # compute d
        d = np.dot(F.T, psr.res/(efac**2*s + equad**2))

        # compute Sigma
        N = 1/(efac**2*s + equad**2)
        right = (N*F.T).T
        FNF = np.dot(F.T, right)

        arr = np.zeros(2*len(rho))
        ct = 0
        for ii in range(0, 2*len(rho), 2):
            arr[ii] = rho[ct]
            arr[ii+1] = rho[ct]
            ct += 1

        Sigma = FNF + np.diag(1/10**arr)

        # cholesky decomp for second term in exponential
        cf = sl.cho_factor(Sigma)
        expval2 = sl.cho_solve(cf, d)
        logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

        logdet_Phi = np.sum(np.log(10**arr))

        logdet_N = np.sum(np.log(2*np.pi*(efac**2*s + equad**2)))

        dtNdt = np.sum(psr.res**2/(efac**2*s + equad**2))

        logLike = -0.5 * (logdet_N + logdet_Phi + logdet_Sigma)\
                        - 0.5 * (dtNdt - np.dot(d, expval2))

      

    return logLike

def lentatiMarginalizedLikePL(psr, F, s, A, f, gam, efac, equad, fc=None, beta=None):
    """
    Lentati marginalized likelihood function only including efac and equad
    and power law coefficients

    @param psr: Pulsar class
    @param F: Fourier design matrix constructed in PALutils
    @param s: diagonalized white noise matrix
    @param A: Power spectrum Amplitude
    @param gam: Power spectrum index
    @param f: Frequencies at which to parameterize power spectrum (Hz)
    @param efac: constant multipier on error bar covaraince matrix term
    @param equad: Additional white noise added in quadrature to efac
    @param fc: Optional cross over frequency in powerlaw:

                P(f) = A/(1+(f/fc)^2)^-gamma/2

    @param beta: Optional secondary spectral index in powerlaw:

                P(f) = A f^-gamma/(1+(f/fc)^2)^beta/2

    @return: LogLike: loglikelihood

    """

    # compute total time span of data
    Tspan = psr.toas.max() - psr.toas.min()

    # get power spectrum coefficients
    f1yr = 1/3.16e7

    if fc is not None and beta is None:
        rho = A**2/12/np.pi**2 * f1yr**(gam-3) /(fc**2 + f**2)**(gam/2)/Tspan

    elif fc is not None and beta is not None:
        rho = A**2/12/np.pi**2 * f1yr**(gam-3) * f**(-gam) * (1+(fc/f)**2)**(-beta/2)/Tspan
        
    elif fc is None and beta is None:
        rho = A**2/12/np.pi**2 * f1yr**(gam-3) * f**(-gam)/Tspan

    # compute d
    d = np.dot(F.T, psr.res/(efac*s + equad**2))

    # compute Sigma
    N = 1/(efac*s + equad**2)
    right = (N*F.T).T
    FNF = np.dot(F.T, right)

    arr = np.zeros(2*len(rho))
    ct = 0
    for ii in range(0, 2*len(rho), 2):
        arr[ii] = rho[ct]
        arr[ii+1] = rho[ct]
        ct += 1

    Phi = np.diag(10**arr)
    Sigma = FNF + np.diag(1/arr)

    # cholesky decomp for second term in exponential
    cf = sl.cho_factor(Sigma)
    expval2 = sl.cho_solve(cf, d)
    logdet_Sigma = np.sum(2*np.log(np.diag(cf[0]))) #+ psr.G.shape[0]*np.log(2*np.pi)

    logdet_Phi = np.sum(np.log(2*np.pi*arr))

    logdet_N = np.sum(np.log(2*np.pi*(efac*s + equad**2)))

    dtNdt = np.sum(psr.res**2/(efac*s + equad**2))

    logLike = -0.5 * (logdet_N + logdet_Phi + logdet_Sigma)\
                    - 0.5 * (dtNdt - np.dot(d, expval2))


    return logLike

def modelIndependentFullPTA(psr, F, s, rho, kappa, efac, equad, ORF):
    """
    Model Independent stochastic background likelihood function

    """
    tstart = time.time()

    # get the number of modes, should be the same for all pulsars
    nmode = len(rho)
    npsr = len(psr)

    loglike1 = 0
    FtNF = []
    for ct,p in enumerate(psr):
    
        # compute d
        if ct == 0:
            d = np.dot(F[ct].T, p.res/(efac[ct]*s[ct] + equad[ct]**2))
        else:
            d = np.append(d, np.dot(F[ct].T, p.res/(efac[ct]*s[ct] + equad[ct]**2)))

        # compute FT N F
        N = 1/(efac[ct]*s[ct] + equad[ct]**2)
        right = (N*F[ct].T).T
        FtNF.append(np.dot(F[ct].T, right))
        
        # log determinant of N
        logdet_N = np.sum(np.log(efac[ct]*s[ct] + equad[ct]**2))

        # triple produce in likelihood function
        dtNdt = np.sum(p.res**2/(efac[ct]*s[ct] + equad[ct]**2))

        loglike1 += -0.5 * (logdet_N + dtNdt)

    # construct elements of sigma array
    sigdiag = []
    sigoffdiag = []
    for ii in range(npsr):
        tot = np.zeros(2*nmode)
        offdiag = np.zeros(2*nmode)

        # off diagonal terms
        offdiag[0::2] = 10**rho
        offdiag[1::2] = 10**rho

        # diagonal terms
        tot[0::2] = 10**rho
        tot[1::2] = 10**rho

        # add in individual red noise
        if len(kappa[ii]) > 0:
            tot[0::2][0:len(kappa[ii])] = 10**kappa[ii]
            tot[1::2][0:len(kappa[ii])] = 10**kappa[ii]
        
        # fill in lists of arrays
        sigdiag.append(tot)
        sigoffdiag.append(offdiag)

    tstart2 = time.time()

    # compute Phi inverse from Lindley's code
    smallMatrix = np.zeros((2*nmode, npsr, npsr))
    for ii in range(npsr):
        for jj in range(ii,npsr):

            if ii == jj:
                smallMatrix[:,ii,jj] = ORF[ii,jj] * sigdiag[jj]
            else:
                smallMatrix[:,ii,jj] = ORF[ii,jj] * sigoffdiag[jj]
                smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]


    # invert them
    logdet_Phi = 0
    for ii in range(2*nmode):
        L = sl.cho_factor(smallMatrix[ii,:,:])
        smallMatrix[ii,:,:] = sl.cho_solve(L, np.eye(npsr))
        logdet_Phi += np.sum(2*np.log(np.diag(L[0])))

    # now fill in real covariance matrix
    Phi = np.zeros((2*npsr*nmode, 2*npsr*nmode))
    for ii in range(npsr):
        for jj in range(ii,npsr):
            for kk in range(0,2*nmode):
                Phi[kk+ii*2*nmode,kk+jj*2*nmode] = smallMatrix[kk,ii,jj]
    
    # symmeterize Phi
    Phi = Phi + Phi.T - np.diag(np.diag(Phi))
            
    # compute sigma
    Sigma = sl.block_diag(*FtNF) + Phi

    tmatrix = time.time() - tstart2

    tstart3 = time.time()
            
    # cholesky decomp for second term in exponential
    cf = sl.cho_factor(Sigma)
    expval2 = sl.cho_solve(cf, d)
    logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

    tinverse = time.time() - tstart3

    logLike = -0.5 * (logdet_Phi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) + loglike1

    #print 'Total time: {0}'.format(time.time() - tstart)
    #print 'Matrix construction time: {0}'.format(tmatrix)
    #print 'Inversion time: {0}\n'.format(tinverse)


    return logLike

def modelIndependentFullPTANoisePL(psr, F, s, f, rho, Ared, gred, efac, equad, ORF):
    """
    Model Independent stochastic background likelihood function

    """
    tstart = time.time()

    # get the number of modes, should be the same for all pulsars
    nmode = len(rho)
    npsr = len(psr)

    # parameterize intrinsic red noise as power law
    kappa = [] 
    Tspan = 1/f[0]
    f1yr = 1/3.16e7
    for ii in range(npsr):
        if Ared[ii] == 0:
            kappa.append([])
        else:
            kappa.append(np.log10(Ared[ii]**2/12/np.pi**2 * f1yr**(gred[ii]-3) * f**(-gred[ii])/Tspan))

    loglike1 = 0
    FtNF = []
    for ct,p in enumerate(psr):
    
        # compute d
        if ct == 0:
            d = np.dot(F[ct].T, p.res/(efac[ct]*s[ct] + equad[ct]**2))
        else:
            d = np.append(d, np.dot(F[ct].T, p.res/(efac[ct]*s[ct] + equad[ct]**2)))


        # compute FT N F
        N = 1/(efac[ct]*s[ct] + equad[ct]**2)
        right = (N*F[ct].T).T
        FtNF.append(np.dot(F[ct].T, right))
        
        # log determinant of N
        logdet_N = np.sum(np.log(efac[ct]*s[ct] + equad[ct]**2))

        # triple produce in likelihood function
        dtNdt = np.sum(p.res**2/(efac[ct]*s[ct] + equad[ct]**2))

        loglike1 += -0.5 * (logdet_N + dtNdt)

    # construct elements of sigma array
    sigdiag = []
    sigoffdiag = []
    for ii in range(npsr):
        tot = np.zeros(2*nmode)
        offdiag = np.zeros(2*nmode)

        # off diagonal terms
        offdiag[0::2] = 10**rho
        offdiag[1::2] = 10**rho

        # diagonal terms
        tot[0::2] = 10**rho
        tot[1::2] = 10**rho
        
        # add in individual red noise
        if len(kappa[ii]) > 0:
            tot[0::2][0:len(kappa[ii])] += 10**kappa[ii]
            tot[1::2][0:len(kappa[ii])] += 10**kappa[ii]
        
        # fill in lists of arrays
        sigdiag.append(tot)
        sigoffdiag.append(offdiag)

    tstart2 = time.time()

    # compute Phi inverse from Lindley's code
    smallMatrix = np.zeros((2*nmode, npsr, npsr))
    for ii in range(npsr):
        for jj in range(ii,npsr):

            if ii == jj:
                smallMatrix[:,ii,jj] = ORF[ii,jj] * sigdiag[jj]
            else:
                smallMatrix[:,ii,jj] = ORF[ii,jj] * sigoffdiag[jj]
                smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]


    # invert them
    logdet_Phi = 0
    for ii in range(2*nmode):
        L = sl.cho_factor(smallMatrix[ii,:,:])
        smallMatrix[ii,:,:] = sl.cho_solve(L, np.eye(npsr))
        logdet_Phi += np.sum(2*np.log(np.diag(L[0])))

    # now fill in real covariance matrix
    Phi = np.zeros((2*npsr*nmode, 2*npsr*nmode))
    for ii in range(npsr):
        for jj in range(ii,npsr):
            for kk in range(0,2*nmode):
                Phi[kk+ii*2*nmode,kk+jj*2*nmode] = smallMatrix[kk,ii,jj]
    
    # symmeterize Phi
    Phi = Phi + Phi.T - np.diag(np.diag(Phi))
            
    # compute sigma
    Sigma = sl.block_diag(*FtNF) + Phi

    tmatrix = time.time() - tstart2

    tstart3 = time.time()
            
    # cholesky decomp for second term in exponential
    try:
        cf = sl.cho_factor(Sigma)
        expval2 = sl.cho_solve(cf, d)
        logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

    except np.linalg.LinAlgError:
        print 'Cholesky Decomposition Failed!! Using SVD instead'
        u,s,v = sl.svd(Sigma)
        expval2 = np.dot(u, 1/s*np.dot(u.T, d))
        logdet_Sigma = np.sum(np.log(s))

    tinverse = time.time() - tstart3

    logLike = -0.5 * (logdet_Phi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) + loglike1

    #print 'Total time: {0}'.format(time.time() - tstart)
    #print 'Matrix construction time: {0}'.format(tmatrix)
    #print 'Inversion time: {0}\n'.format(tinverse)


    return logLike

def modelIndependentFullPTAPL(psr, F, s, f, Agw, gamgw, Ared, gred, efac, equad, ORF):
    """
    Model Independent stochastic background likelihood function

    """
    tstart = time.time()

    # parameterize GW as power law
    Tspan = 1/f[0]
    f1yr = 1/3.16e7
    rho = np.log10(Agw**2/12/np.pi**2 * f1yr**(gamgw-3) * f**(-gamgw)/Tspan)

    # get the number of modes, should be the same for all pulsars
    nmode = len(rho)
    npsr = len(psr)

    # parameterize intrinsic red noise as power law
    kappa = [] 
    for ii in range(npsr):
        if Ared[ii] == 0:
            kappa.append([])
        else:
            kappa.append(np.log10(Ared[ii]**2/12/np.pi**2 * f1yr**(gred[ii]-3) * f**(-gred[ii])/Tspan))

    loglike1 = 0
    FtNF = []
    for ct,p in enumerate(psr):
    
        # compute d
        if ct == 0:
            d = np.dot(F[ct].T, p.res/(efac[ct]*s[ct] + equad[ct]**2))
        else:
            d = np.append(d, np.dot(F[ct].T, p.res/(efac[ct]*s[ct] + equad[ct]**2)))

        # compute FT N F
        N = 1/(efac[ct]*s[ct] + equad[ct]**2)
        right = (N*F[ct].T).T
        FtNF.append(np.dot(F[ct].T, right))
        
        # log determinant of N
        logdet_N = np.sum(np.log(efac[ct]*s[ct] + equad[ct]**2))

        # triple produce in likelihood function
        dtNdt = np.sum(p.res**2/(efac[ct]*s[ct] + equad[ct]**2))

        loglike1 += -0.5 * (logdet_N + dtNdt)

    tF = time.time() - tstart
    
    tstart2 = time.time()

    # construct elements of sigma array
    sigdiag = []
    sigoffdiag = []
    for ii in range(npsr):
        tot = np.zeros(2*nmode)
        offdiag = np.zeros(2*nmode)

        # off diagonal terms
        offdiag[0::2] = 10**rho
        offdiag[1::2] = 10**rho

        # diagonal terms
        tot[0::2] = 10**rho
        tot[1::2] = 10**rho

        # add in individual red noise
        if len(kappa[ii]) > 0:
            tot[0::2][0:len(kappa[ii])] += 10**kappa[ii]
            tot[1::2][0:len(kappa[ii])] += 10**kappa[ii]
        
        # fill in lists of arrays
        sigdiag.append(tot)
        sigoffdiag.append(offdiag)


    # compute Phi inverse from Lindley's code
    smallMatrix = np.zeros((2*nmode, npsr, npsr))
    for ii in range(npsr):
        for jj in range(ii,npsr):

            if ii == jj:
                smallMatrix[:,ii,jj] = ORF[ii,jj] * sigdiag[jj]
            else:
                smallMatrix[:,ii,jj] = ORF[ii,jj] * sigoffdiag[jj]
                smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]


    # invert them
    logdet_Phi = 0
    for ii in range(2*nmode):
        L = sl.cho_factor(smallMatrix[ii,:,:])
        smallMatrix[ii,:,:] = sl.cho_solve(L, np.eye(npsr))
        logdet_Phi += np.sum(2*np.log(np.diag(L[0])))

    # now fill in real covariance matrix
    Phi = np.zeros((2*npsr*nmode, 2*npsr*nmode))
    for ii in range(npsr):
        for jj in range(ii,npsr):
            for kk in range(0,2*nmode):
                Phi[kk+ii*2*nmode,kk+jj*2*nmode] = smallMatrix[kk,ii,jj]
    
    # symmeterize Phi
    Phi = Phi + Phi.T - np.diag(np.diag(Phi))
            
    # compute sigma
    Sigma = sl.block_diag(*FtNF) + Phi

    tmatrix = time.time() - tstart2

    tstart3 = time.time()
            
    # cholesky decomp for second term in exponential
    cf = sl.cho_factor(Sigma)
    expval2 = sl.cho_solve(cf, d)
    logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

    tinverse = time.time() - tstart3

    logLike = -0.5 * (logdet_Phi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) + loglike1

    #print 'Total time: {0}'.format(time.time() - tstart)
    #print 'FtF time: {0}'.format(tF)
    #print 'Matrix construction time: {0}'.format(tmatrix)
    #print 'Inversion time: {0}\n'.format(tinverse)

    return logLike

def modelIndependentFullPTASinglSource(psr, proj, s, f, theta, phi, rho, kappa, efac, equad, ORF):
    """
    Model Independent single source testing function

    """
    tstart = time.time()
    
    # get the number of modes, should be the same for all pulsars
    nmode = len(rho)
    npsr = len(psr)

    # get F matrices for all pulsars at given frequency
    F = [np.array([np.sin(2*np.pi*f*p.toas), np.cos(2*np.pi*f*p.toas)]).T for p in psr]

    F = [np.dot(proj[ii], F[ii]) for ii in range(len(proj))]

    loglike1 = 0
    FtNF = []
    for ct,p in enumerate(psr):
    
        # compute d
        if ct == 0:
            d = np.dot(F[ct].T, p.res/(efac[ct]*s[ct] + equad[ct]**2))
        else:
            d = np.append(d, np.dot(F[ct].T, p.res/(efac[ct]*s[ct] + equad[ct]**2)))

        # compute FT N F
        N = 1/(efac[ct]*s[ct] + equad[ct]**2)
        right = (N*F[ct].T).T
        FtNF.append(np.dot(F[ct].T, right))
        
        # log determinant of N
        logdet_N = np.sum(np.log(efac[ct]*s[ct] + equad[ct]**2))

        # triple produce in likelihood function
        dtNdt = np.sum(p.res**2/(efac[ct]*s[ct] + equad[ct]**2))

        loglike1 += -0.5 * (logdet_N + dtNdt)

    # construct elements of sigma array
    sigdiag = []
    sigoffdiag = []
    fplus = np.zeros(npsr)
    fcross = np.zeros(npsr)
    for ii in range(npsr):
        fplus[ii], fcross[ii], cosMu = PALutils.createAntennaPatternFuncs(psr[ii], theta, phi)
        tot = np.zeros(2*nmode)
        offdiag = np.zeros(2*nmode)

        # off diagonal terms
        offdiag[0::2] = 10**rho 
        offdiag[1::2] = 10**rho

        # diagonal terms
        tot[0::2] = 10**rho
        tot[1::2] = 10**rho

        # add in individual red noise
        if len(kappa[ii]) > 0:
            tot[0::2][0:len(kappa[ii])] += 10**kappa[ii]
            tot[1::2][0:len(kappa[ii])] += 10**kappa[ii]
        
        # fill in lists of arrays
        sigdiag.append(tot)
        sigoffdiag.append(offdiag)

    tstart2 = time.time()

    # compute Phi inverse from Lindley's code
    smallMatrix = np.zeros((2*nmode, npsr, npsr))
    for ii in range(npsr):
        for jj in range(ii,npsr):

            if ii == jj:
                smallMatrix[:,ii,jj] = ORF[ii,jj] * sigdiag[jj] * (fplus[ii]**2 + fcross[ii]**2)
            else:
                smallMatrix[:,ii,jj] = ORF[ii,jj] * sigoffdiag[jj] * (fplus[ii]*fplus[jj] + fcross[ii]*fcross[jj])
                smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]


    # invert them
    logdet_Phi = 0
    for ii in range(2*nmode):
        L = sl.cho_factor(smallMatrix[ii,:,:])
        smallMatrix[ii,:,:] = sl.cho_solve(L, np.eye(npsr))
        logdet_Phi += np.sum(2*np.log(np.diag(L[0])))

    # now fill in real covariance matrix
    Phi = np.zeros((2*npsr*nmode, 2*npsr*nmode))
    for ii in range(npsr):
        for jj in range(ii,npsr):
            for kk in range(0,2*nmode):
                Phi[kk+ii*2*nmode,kk+jj*2*nmode] = smallMatrix[kk,ii,jj]
    
    # symmeterize Phi
    Phi = Phi + Phi.T - np.diag(np.diag(Phi))
            
    # compute sigma
    Sigma = sl.block_diag(*FtNF) + Phi

    tmatrix = time.time() - tstart2

    tstart3 = time.time()
            
    # cholesky decomp for second term in exponential
    cf = sl.cho_factor(Sigma)
    expval2 = sl.cho_solve(cf, d)
    logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

    tinverse = time.time() - tstart3

    logLike = -0.5 * (logdet_Phi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) + loglike1

    #print 'Total time: {0}'.format(time.time() - tstart)
    #print 'Matrix construction time: {0}'.format(tmatrix)
    #print 'Inversion time: {0}\n'.format(tinverse)

    return logLike


def modelIndependentFirstOrder(psr, F, s, rho, kappa, efac, equad, ORF):
    """
    Model Independent stochastic background first order likelihood function

    """
    tstart = time.time()

    # get the number of modes, should be the same for all pulsars
    npsr = len(psr)
   
    logLike = 0
    P = []
    phi = []
    for ct, p in enumerate(psr):
    
        nmode = len(rho[ct])
   
        # compute d
        d = np.dot(F[ct].T, p.res/(efac[ct]*s[ct] + equad[ct]**2))

        # compute Sigma
        N = 1/(efac[ct]*s[ct] + equad[ct]**2)
        right = (N*F[ct].T).T
        FNF = np.dot(F[ct].T, right)

        Phi = np.zeros(2*nmode)
        Phi[0::2] = 10**rho[ct]
        Phi[1::2] = 10**rho[ct]

        phi.append(Phi)
                
        # add in individual red noise
        if len(kappa[ct]) > 0:
            Phi[0::2][0:len(kappa[ct])] += 10**kappa[ct]
            Phi[1::2][0:len(kappa[ct])] += 10**kappa[ct]


        # compute Sigma matrix
        Sigma = FNF + np.diag(1/Phi)

        tmp1 = N*p.res
        tmp2 = np.dot(right, np.dot(Sigma, np.dot(right.T, p.res)))
        P.append(np.dot(F[ct].T, tmp1+tmp2))

        # cholesky decomp for second term in exponential
        cf = sl.cho_factor(Sigma)
        expval2 = sl.cho_solve(cf, d)
        logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

        logdet_Phi = np.sum(np.log(Phi))

        logdet_N = np.sum(np.log(efac[ct]*s[ct] + equad[ct]**2))

        dtNdt = np.sum(p.res**2/(efac[ct]*s[ct] + equad[ct]**2))

        logLike += -0.5 * (logdet_N + logdet_Phi + logdet_Sigma)\
                        - 0.5 * (dtNdt - np.dot(d, expval2))

    # auto terms
    for ii in range(npsr):
        for jj in range(ii+1, npsr):
            logLike += 0.5 * ORF[ii,jj] * np.dot(P[ii], phi[ii] * P[jj])

    
    print 'Evaluation time = {0} s'.format(time.time() - tstart)

    return logLike



