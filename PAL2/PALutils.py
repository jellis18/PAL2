from __future__ import division
import numpy as np
import scipy.special as ss
import scipy.linalg as sl
import scipy.integrate as si
import scipy.interpolate as interp
#import numexpr as ne
import sys,os

def createAntennaPatternFuncs(psr, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).

    @param psr: pulsar object for single pulsar
    @param gwtheta: GW polar angle in radians
    @param gwphi: GW azimuthal angle in radians

    @return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the 
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = np.array([-np.sin(gwphi), np.cos(gwphi), 0.0])
    n = np.array([-np.cos(gwtheta)*np.cos(gwphi), -np.cos(gwtheta)*np.sin(gwphi),\
                  np.sin(gwtheta)])
    omhat = np.array([-np.sin(gwtheta)*np.cos(gwphi), -np.sin(gwtheta)*np.sin(gwphi),\
                      -np.cos(gwtheta)])

    phat = np.array([np.sin(psr.theta)*np.cos(psr.phi), np.sin(psr.theta)*np.sin(psr.phi), \
                     np.cos(psr.theta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    return fplus, fcross, cosMu

def createResiduals(psr, gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc, pdist=None, \
                        pphase=None, psrTerm=True, evolve=True, phase_approx=False):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013.

    @param psr: pulsar object for single pulsar
    @param gwtheta: Polar angle of GW source in celestial coords [radians]
    @param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    @param mc: Chirp mass of SMBMB [solar masses]
    @param dist: Luminosity distance to SMBMB [Mpc]
    @param fgw: Frequency of GW (twice the orbital frequency) [Hz]
    @param phase0: Initial Phase of GW source [radians]
    @param psi: Polarization of GW source [radians]
    @param inc: Inclination of GW source [radians]
    @param pdist: Pulsar distance to use other than those in psr [kpc]
    @param pphase: Use pulsar phase to determine distance [radian]
    @param psrTerm: Option to include pulsar term [boolean] 
    @param evolve: Option to exclude evolution [boolean]

    @return: Vector of induced residuals

    """

    # get antenna pattern funcs and cosMu
    fplus, fcross, cosMu = createAntennaPatternFuncs(psr, gwtheta, gwphi)
    
    # get values from pulsar object
    toas = psr.toas
    if pdist is None and pphase is None:
        pdist = psr.dist
    elif pdist is None and pphase is not None:
        pdist = pphase/(2*np.pi*fgw*(1-cosMu)) / 1.0267e11
   

    # convert units
    mc *= 4.9e-6         # convert from solar masses to seconds
    dist *= 1.0267e14    # convert from Mpc to seconds
    pdist *= 1.0267e11   # convert from kpc to seconds
    
    # get pulsar time
    tp = toas-pdist*(1-cosMu)

    # orbital frequency
    w0 = np.pi * fgw
    phase0 /= 2 # orbital phase
    omegadot = 96/5 * mc**(5/3) * w0**(11/3)

    # evolution
    if evolve:

        # calculate time dependent frequency at earth and pulsar
        omega = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * toas)**(-3/8)
        omega_p = w0 * (1 - 256/5 * mc**(5/3) * w0**(8/3) * tp)**(-3/8)

        # calculate time dependent phase
        phase = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega**(-5/3))
        phase_p = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3))
    

    elif phase_approx:
        
        # monochromatic
        omega = np.pi*fgw
        omega_p = w0 * (1 + 256/5 * mc**(5/3) * w0**(8/3) * pdist*(1-cosMu))**(-3/8)
        
        # phases
        phase = phase0 + omega * toas
        phase_p = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3)) + omega_p*toas
          
    # no evolution
    else: 
        
        # monochromatic
        omega = np.pi*fgw
        omega_p = omega
        
        # phases
        phase = phase0 + omega * toas
        phase_p = phase0 + omega * tp
        

    # define time dependent coefficients
    At = -0.5*np.sin(2*phase)*(3+np.cos(2*inc))
    Bt = 2*np.cos(2*phase)*np.cos(inc)
    At_p = -0.5*np.sin(2*phase_p)*(3+np.cos(2*inc))
    Bt_p = 2*np.cos(2*phase_p)*np.cos(inc)

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*omega**(1./3.))
    alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))


    # define rplus and rcross
    rplus = alpha*(At*np.cos(2*psi)-Bt*np.sin(2*psi))
    rcross = alpha*(At*np.sin(2*psi)+Bt*np.cos(2*psi))
    rplus_p = alpha_p*(At_p*np.cos(2*psi)-Bt_p*np.sin(2*psi))
    rcross_p = alpha_p*(At_p*np.sin(2*psi)+Bt_p*np.cos(2*psi))

    # residuals
    if psrTerm:
        res = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
    else:
        res = -fplus*rplus - fcross*rcross

    return res

def createResidualsFast(psr, gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc, pdist=None, \
                        pphase=None, psrTerm=True, evolve=True, phase_approx=False):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013. Trys to be smart about it

    @param psr: list of pulsar objects for all pulsars
    @param gwtheta: Polar angle of GW source in celestial coords [radians]
    @param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    @param mc: Chirp mass of SMBMB [solar masses]
    @param dist: Luminosity distance to SMBMB [Mpc]
    @param fgw: Frequency of GW (twice the orbital frequency) [Hz]
    @param phase0: Initial Phase of GW source [radians]
    @param psi: Polarization of GW source [radians]
    @param inc: Inclination of GW source [radians]
    @param pdist: Pulsar distance to use other than those in psr [kpc]
    @param pphase: Use pulsar phase to determine distance [radian]
    @param psrTerm: Option to include pulsar term [boolean] 
    @param evolve: Option to exclude evolution [boolean]

    @return: Vector of induced residuals

    """

    # convert units
    mc *= 4.9e-6         # convert from solar masses to seconds
    dist *= 1.0267e14    # convert from Mpc to seconds

    # define initial orbital frequency 
    w0 = np.pi * fgw
    phase0 /= 2 # orbital phase
    w053 = w0**(-5/3)

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)
    incfac1, incfac2 = -0.5*(3+np.cos(2*inc)), 2*np.cos(inc)

    # unit vectors to GW source
    m = np.array([-singwphi, cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

    # various factors invloving GW parameters
    fac1 = 256/5 * mc**(5/3) * w0**(8/3) 
    fac2 = 1/32/mc**(5/3)
    fac3 = mc**(5/3)/dist

    res = []
    for ct, p in enumerate(psr):

        # use definition from Sesana et al 2010 and Ellis et al 2012
        phat = np.array([np.sin(p.theta)*np.cos(p.phi), np.sin(p.theta)*np.sin(p.phi),\
                np.cos(p.theta)])

        fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
        fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
        cosMu = -np.dot(omhat, phat)

    
        # get values from pulsar object
        toas = p.toas
        if pdist is None and pphase is None:
            pd = p.dist
        elif pdist is None and pphase is not None:
            pd = pphase[ct]/(2*np.pi*fgw*(1-cosMu)) / 1.0267e11
        else:
            pd = pdist[ct]
        

        # convert units
        pd *= 1.0267e11   # convert from kpc to seconds
        
        # get pulsar time
        tp = toas-pd*(1-cosMu)

        # evolution
        if evolve:

            # calculate time dependent frequency at earth and pulsar
            omega = w0 * (1 - fac1 * toas)**(-3/8)
            omega_p = w0 * (1 - fac1 * tp)**(-3/8)

            # calculate time dependent phase
            phase = phase0 + fac2 * (w053 - omega**(-5/3))
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3))
        
        # use approximation that frequency does not evlolve over observation time
        elif phase_approx:
            
            # frequencies
            omega = w0
            omega_p = w0 * (1 + fac1 * pd*(1-cosMu))**(-3/8)
            
            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3)) + omega_p*toas
              
        # no evolution
        else: 
            
            # monochromatic
            omega = w0
            omega_p = omega
            
            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + omega * tp
            

        # define time dependent coefficients
        At = np.sin(2*phase) * incfac1
        Bt = np.cos(2*phase) * incfac2
        At_p = np.sin(2*phase_p) * incfac1
        Bt_p = np.cos(2*phase_p) * incfac2

        # now define time dependent amplitudes
        alpha = fac3 / omega**(1/3)
        alpha_p = fac3 / omega_p**(1/3)

        # define rplus and rcross
        rplus = alpha * (At*cos2psi - Bt*sin2psi)
        rcross = alpha * (At*sin2psi + Bt*cos2psi)
        rplus_p = alpha_p * (At_p*cos2psi - Bt_p*sin2psi)
        rcross_p = alpha_p * (At_p*sin2psi + Bt_p*cos2psi)

        # residuals
        if psrTerm:
            res.append(fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross))
        else:
            res.append(-fplus*rplus - fcross*rcross)

    return res



def computeLuminosityDistance(z):
    """

    Compute luminosity distance via gaussian quadrature.

    @param z: Redshift at which to calculate luminosity distance

    @return: Luminosity distance in Mpc

    """

    # constants
    H0 = 71 * 1000      # Hubble constant in (m/s)/Mpc
    Ol = 0.73           # Omega lambda
    Om = 0.27           # Omega matter
    G = 6.67e-11        # Gravitational constant in SI units
    c = 3.0e8           # Speed of light in SI units

    # proper distance function
    properDistance = lambda z: c/H0/np.sqrt(Ol+Om*(1+z)**3)
    
    # carry out numerical integration
    Dp = si.quad(properDistance, 0 ,z)[0]
    Dl = (1+z) * Dp

    return Dl

def calculateMatchedFilterSNR(psr, data, temp):
    """

    Compute the SNR from a single continuous source for a single puslar

    @param psr: Pulsar class containing all info on pulsar
    @param data: The residual data (or the template if we want to optimal SNR)
    @param temp: The template model of the signal

    @return: SNR

    """

    return np.dot(data, np.dot(psr.invCov, temp))/np.sqrt(np.dot(temp, np.dot(psr.invCov, temp)))

def createRmatrix(designmatrix, err):
    """
    Create R matrix as defined in Ellis et al (2013) and Demorest et al (2012)

    @param designmatrix: Design matrix as returned by tempo2

    @return: R matrix 
   
    """

    W = np.diag(1/err)

    u, s, v = sl.svd(np.dot(W, designmatrix),full_matrices=False)

    return np.eye(len(err)) - np.dot(np.linalg.inv(W), np.dot(u, np.dot(u.T, W))) 


def createGmatrix(designmatrix):
    """
    Return G matrix as defined in van Haasteren et al 2013

    @param designmatrix: Design matrix as returned by tempo2

    @return: G matrix as defined in van Haasteren et al 2013

    """

    nfit = designmatrix.shape[1]
    npts = designmatrix.shape[0]

    # take singular value decomp
    u, s, v = sl.svd(designmatrix, full_matrices=True)

    return u[:,-(npts-nfit):]

def createQSDdesignmatrix(toas):
    """
    Return designmatrix for QSD model

    @param toas: vector of TOAs in seconds

    @return: M design matrix for QSD

   """

    designmatrix = np.zeros((len(toas), 3))

    for ii in range(3):
        designmatrix[:,ii] = toas**(ii)

    return designmatrix

def createDesignmatrix(toas, freqs, RADEC=False, PX=False, DMX=False):
    """
    Return designmatrix for QSD model

    @param psr: Pulsar class

    @return: M design matrix for QSD

   """
    model = ['QSD', 'QSD', 'QSD']
    if RADEC:
        model.append('RA')
        model.append('DEC')
    if PX:
        model.append('PX')
    if DMX:
        if len(toas) % 2 != 0:
            print "ERROR: len(toas) must be a factor of 2 for DMX!!"
            sys.exit()

        model += ['DMX' for ii in range(int(len(toas)/2))]
    
    ndim = len(model)
    designmatrix = np.zeros((len(toas), ndim))
    
    for ii in range(ndim):
        if model[ii] == 'QSD':
            designmatrix[:,ii] = toas**(ii)
        if model[ii] == 'RA':
            designmatrix[:,ii] = np.sin(2*np.pi/3.16e7*toas)
        if model[ii] == 'DEC':
            designmatrix[:,ii] = np.cos(2*np.pi/3.16e7*toas)
        if model[ii] == 'PX': 
            designmatrix[:,ii] = np.cos(4*np.pi/3.16e7*toas)
        if model[ii] == 'DMX':
            designmatrix[:,ii:] = DMXDesignMatrix(toas, freqs, dt=43200)
            break

    return designmatrix


def createTimeLags(toa1, toa2, round=True):
    """
    Create matrix of time lags tm = |t_i - t_j|

    @param toa1: times-of-arrival in seconds for psr 1
    @param toa2: times-of-arrival in seconds for psr 2
    @param round: option to round time difference to 0 if less than 1 hr

    @return: matrix of time lags tm = |t_i - t_j|

    """

    t1, t2 = np.meshgrid(toa2, toa1)

    tm = np.abs(t1-t2)

    if round:
        hr = 3600. # hour in seconds
        tm = np.where(tm<hr, 0.0, tm)
        
    return tm

def exploderMatrix(toas, freqs=None, dt=1200, flags=None):
    """
    Compute exploder matrix for daily averaging

    @param toas: array of toas
    @param dt: time offset (seconds)

    @return: exploder matrix and daily averaged toas

    """


    processed = np.array([0]*len(toas), dtype=np.bool)  # No toas processed yet
    U = np.zeros((len(toas), 0))
    avetoas = np.empty(0)
    avefreqs = np.empty(0)
    aveflags = []

    while not np.all(processed):
        npindex = np.where(processed == False)[0]
        ind = npindex[0]
        satmin = toas[ind] - dt
        satmax = toas[ind] + dt

        dailyind = np.where(np.logical_and(toas > satmin, toas < satmax))[0]

        newcol = np.zeros((len(toas)))
        newcol[dailyind] = 1.0

        U = np.append(U, np.array([newcol]).T, axis=1)
        avetoas = np.append(avetoas, np.mean(toas[dailyind]))
        
        if freqs is not None:
            avefreqs = np.append(avefreqs, np.mean(freqs[dailyind]))
        
        # TODO: what if we have different backends overlapping
        if flags is not None:
            aveflags.append(flags[dailyind][0])

        processed[dailyind] = True
    
    if freqs is not None and flags is not None:
        return avetoas, avefreqs, aveflags, U
    elif freqs is not None and flags is None:
        return avetoas, avefreqs, U
    elif freqs is None and flags is not None:
        return avetoas, aveflags, U
    else:
        return avetoas, U

def DMXDesignMatrix(toas, freqs, dt=1200):
    """
    Compute DMX Design matrix

    @param toas: array of toas
    @param freqs: array of frequencies in Hz
    @param dt: time offset (seconds)

    @return: Design matrix for DMX

    """


    processed = np.array([0]*len(toas), dtype=np.bool)  # No toas processed yet
    M = np.zeros((len(toas), 0))

    while not np.all(processed):
        npindex = np.where(processed == False)[0]
        ind = npindex[0]
        satmin = toas[ind] - dt
        satmax = toas[ind] + dt

        dailyind = np.where(np.logical_and(toas > satmin, toas < satmax))[0]

        newcol = np.zeros((len(toas)))
        newcol[dailyind] = 1.0/freqs[dailyind]**2

        M = np.append(M, np.array([newcol]).T, axis=1)
        
        processed[dailyind] = True
    
    return M


def sumTermCovarianceMatrix(tm, fL, gam, nsteps):
    """
    Calculate the power series expansion for the Hypergeometric
    function in the standard power law covariance matrix.

    @param tm: Matrix of time lags in years
    @param fL: Low frequency cutoff
    @param gam: Power Law spectral index
    @param nsteps: Number of terms in the power series expansion
    """

    sum=0
    for i in range(nsteps):

     sum += ((-1)**i)*((2*np.pi*fL*tm)**(2*i))/(ss.gamma(2*i+1)*(2.*i+1-gam))

    return sum

def sumTermCovarianceMatrix_fast(tm, fL, gam):
    """
    Calculate the power series expansion for the Hypergeometric
    function in the standard power law covariance matrix. This
    version uses the Python package numexpr and is much faster
    that using numpy. For now it is hardcoded to use only the 
    first 3 terms.

    @param tm: Matrix of time lags in years
    @param fL: Low frequency cutoff
    @param gam: Power Law spectral index
    """

    x = 2*np.pi*fL*tm

    sum = ne.evaluate("1/(1-gam) - x**2/(2*(3-gam)) + x**4/(24*(5-gam))")

    return sum


def createGHmatrix(toa, err, res, G, fidelity):
    """
    Create "H" compression matrix as defined in van Haasteren 2013(b).
    Multiplies with "G" matrix to create the "GH" matrix, which can simply replace
    the "G" matrix in all likelihoods which are marginalised over the timing-model


    @param toa: times-of-arrival (in days) for psr
    @param err: error bars on toas (in seconds)
    @param res: residuals (in seconds) of psr
    @param G: G matrix as defined in van Haasteren et al 2013(a)
    @param fidelity: fraction of total sensitivity retained in compressed data

    @return: GH matrix, which can simply replace "G" matrix in likelihood

    """

    # forming the error-bar covariance matrix, sandwiched with G matrices
    GCnoiseG = np.dot(G.T,np.dot(np.diag(err**2.0)*np.eye(len(err)),G))
    
    # forming the unscaled (Agwb=1) covariance matrix of GWB-induced residuals
    tm = createTimeLags(toa, toa)
    Cgwb = createRedNoiseCovarianceMatrix(tm, 1, 13/3)
    GCgwbG = np.dot(G.T, np.dot(Cgwb, G))
    
    # approximate the whitening matrix with the inverse root of the marginalised error-bar matrix
    CgwbMargWhite = np.dot(sl.sqrtm(sl.inv(GCnoiseG)).T, \
                    np.dot(GCgwbG, sl.sqrtm(sl.inv(GCnoiseG))))

    # compute the eigendecomposition of the 'whitened' GWB covariance matrix; 
    # order the eigenvalues largest first
    eigVal,eigVec = sl.eigh(CgwbMargWhite)
    idx = eigVal.argsort()[::-1] 
    eigVal = eigVal[idx]
    eigVec = eigVec[:,idx]
    
    # computing a rough estimate of the GWB amplitude for a strain-spectrum slope of -2/3
    Tspan = toa.max() - toa.min()
    sigma_gwb = np.std(res) * 1e-15
    Amp = (sigma_gwb/(1.37*(10**(-9)))) / (Tspan**(5/3))
    
    # looping over eigenvalues until the fidelity criterion of van Haasteren 2013(b) 
    # is satisfied; only the 'principal' eigenvectors are retained
    index = np.amax(np.where(np.cumsum((eigVal/(1+(Amp**2.0)*eigVal))**2.0)/ \
                             np.sum((eigVal/(1.0+(Amp**2.0)*eigVal))**2.0).real \
                             <= fidelity)[0]) 
    
    # forming the data-compression matrix
    H = np.dot(sl.sqrtm(sl.inv(GCnoiseG)).real,eigVec.T[:index+1].T.real)
    
    return np.dot(G,H)



def createRedNoiseCovarianceMatrix(tmcopy, Amp, gam, fH=None, fast=False):
    """
    Create red noise covariance matrix. If fH is None, then
    return standard power law covariance matrix. If fH is not
    none, return power law covariance matrix with high frequency 
    cutoff.

    @param tm: Matrix of time lags in seconds
    @param Amp: Amplitude of red noise in GW units
    @param gam: Red noise power law spectral index
    @param fH: Optional high frequency cutoff in yr^-1
    @param fast: Option to use Python numexpr to speed 
                    up calculation (default = True)

    @return: Red noise covariance matrix in seconds^2

    """

    # conversion from seconds to years
    s2yr = 1/3.16e7
    
    # convert tm to yr
    tm = tmcopy.copy()
    tm *= s2yr

    # compute high frequency cutoff
    Tspan = tm.max()
    fL = 1/(10*Tspan)


    if fH is None:

        # convert amplitude to correct units
        A = Amp**2/24/np.pi**2
        if fast:
            x = 2*np.pi*fL*tm
            corr = (2*A/(fL**(gam-1)))*((ss.gamma(1-gam)*np.sin(np.pi*gam/2)*ne.evaluate("x**(gam-1)")) \
                            -sumTermCovarianceMatrix_fast(tm, fL, gam))
        else:
            corr = (2*A/(fL**(gam-1)))*((ss.gamma(1-gam)*np.sin(np.pi*gam/2)*(2*np.pi*fL*tm)**(gam-1)) \
                            -sumTermCovarianceMatrix(tm, fL, gam, 5))

    elif fH is not None:

        alpha=(3-gam)/2.0

        # convert amplitude to correct units
        A = Amp**2
 
        EulerGamma=0.577

        x = 2*np.pi*fL*tm

        norm = (fL**(2*alpha - 2)) * 2**(alpha - 3) / (3 * np.pi**1.5 * ss.gamma(1.5 - alpha))

        # introduce a high-frequency cutoff
        xi = fH/fL
        
        # avoid the gamma singularity at alpha = 1
        if np.abs(alpha - 1) < 1e-6:
            zero = np.log(xi) + (EulerGamma + np.log(0.5 * xi)) * np.log(xi) * (alpha - 1)
        else:
            zero = norm * 2**(-alpha) * ss.gamma(1 - alpha) * (1 - xi**(2*alpha - 2))

        corr = A * np.where(x==0,zero,norm * x**(1 - alpha) * (ss.kv(1 - alpha,x) - xi**(alpha - 1) \
                                                           * ss.kv(1 - alpha,xi * x)))

    # return in s^2
    return corr / (s2yr**2)

def createWhiteNoiseCovarianceMatrix(err, efac, equad, tau=None, tm=None):
    """
    Return standard white noise covariance matrix with
    efac and equad parameters

    @param err: Error bars on toas in seconds
    @param efac: Multiplier on error bar component of covariance matrix
    @param equad: Extra toa independent white noise in seconds
    @param tau: Extra time scale of correlation if appropriate. If this
                parameter is specified must also read in matrix of time lags
    @param tm: Matrix of time lags.

    @return: White noise covariance matrix in seconds^2

    """
    
    if tau is None and tm is None:
        corr = efac * np.diag(err**2) + equad**2 * np.eye(np.alen(err)) 

    elif tau is not None and tm is not None:
        sincFunc = np.sinc(2*np.pi*tm/tau)
        corr = efac * np.diag(err**2) + equad**2 * sincFunc

    return corr

# return the false alarm probability for the fp-statistic
def ptSum(N, fp0):
    """
    Compute False alarm rate for Fp-Statistic. We calculate
    the log of the FAP and then exponentiate it in order
    to avoid numerical precision problems

    @param N: number of pulsars in the search
    @param fp0: The measured value of the Fp-statistic

    @returns: False alarm probability ad defined in Eq (64)
              of Ellis, Seiemens, Creighton (2012)

    """

    n = np.arange(0,N)

    return np.sum(np.exp(n*np.log(fp0)-fp0-np.log(ss.gamma(n+1))))

def dailyAverage(pulsar):
    """

    Function to compute daily averaged residuals such that we
    have one residual per day per frequency band.

    @param pulsar: pulsar class from Michele Vallisneri's 
                     libstempo library.

    @return: mtoas: Average TOA of a single epoch
    @return: qmatrix: Linear operator that transforms residuals to
                      daily averaged residuals
    @return: merr: Daily averaged error bar
    @return: mfreqs: Daily averaged frequency value
    @return: mbands: Frequency band for daily averaged residual

    """

    toas = pulsar.toas()        # days 
    res = pulsar.residuals()    # seconds
    err = pulsar.toaerrs * 1e-6 # seconds
    freqs = pulsar.freqs        # MHz

    # timescale to do averaging (1 day)
    t_ave = 86400    # s/day
    
    # set up array with one day spacing
    yedges = np.longdouble(np.arange(toas.min(),toas.max()+1,1))

    # unique frequency bands
    bands = list(np.unique(pulsar.flags['B']))
    flags = list(pulsar.flags['B'])

    qmatrix = []
    mtoas = []
    merr = []
    mres = []
    mfreqs = []
    mbands = []
    for ii in range(len(yedges)-1):

        # find toa indices that are in bin 
        indices = np.flatnonzero(np.logical_and(toas>=yedges[ii], toas<yedges[ii+1]))

        # loop over different frequency bands
        for band in bands:
            array = np.zeros(len(toas))

            # find indices in that band
            toainds = [ct for ct,flag in enumerate(flags) if flag == band]

            # find all indices that are within 1 day and are in frequency band
            ind = [indices[jj] for jj in range(len(indices)) if np.any(np.equal(indices[jj],toainds))]

            # construct average quantities
            if len(ind) > 0:
                weight = (np.sum(1/err[ind]**2))
                array[ind] = 1/err[ind]**2 / weight
                qmatrix.append(array)
                mtoas.append(np.mean(toas[ind]))
                merr.append(np.sqrt(1/np.sum(1/err[ind]**2)))
                mfreqs.append(np.mean(pulsar.freqs[ind]))
                mbands.append(band)

            
    # turn lists into arrays with double precision
    qmatrix = np.double(np.array(qmatrix))
    mtoas = np.double(np.array(mtoas))
    merr = np.double(np.array(merr))
    mfreqs = np.double(np.array(mfreqs))
    mbands = np.array(mbands)
    
    # construct new design matrix without inter band frequency jumps
    dmatrix = np.double(pulsar.designmatrix())[:,0:-pulsar.nJumps]

    return mtoas, qmatrix, merr, dmatrix, mfreqs, mbands

def computeORF(psr):
    """
    Compute pairwise overlap reduction function values.

    @param psr: List of pulsar object instances

    @return: Numpy array of pairwise ORF values for every pulsar
             in pulsar class

    """

    # begin loop over all pulsar pairs and calculate ORF
    k = 0
    npsr = len(psr)
    ORF = np.zeros(npsr*(npsr-1)/2.)
    phati = np.zeros(3)
    phatj = np.zeros(3)
    for ll in xrange(0, npsr):
        phati[0] = np.cos(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[1] = np.sin(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[2] = np.cos(psr[ll].theta)

        for kk in xrange(ll+1, npsr):
            phatj[0] = np.cos(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[1] = np.sin(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[2] = np.cos(psr[kk].theta)

            xip = (1.-np.sum(phati*phatj)) / 2.
            ORF[k] = 3.*( 1./3. + xip * ( np.log(xip) -1./6.) )
            k += 1

    return ORF

def computeORFMatrix(psr):
    """
    Compute ORF matrix.

    @param psr: List of pulsar object instances

    @return: Matrix that has the ORF values for every pulsar
             pair with 2 on the diagonals to account for the 
             pulsar term.

    """

    # begin loop over all pulsar pairs and calculate ORF
    npsr = len(psr)
    ORF = np.zeros((npsr, npsr))
    phati = np.zeros(3)
    phatj = np.zeros(3)
    for ll in xrange(0, npsr):
        phati[0] = np.cos(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[1] = np.sin(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[2] = np.cos(psr[ll].theta)

        for kk in xrange(0, npsr):
            phatj[0] = np.cos(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[1] = np.sin(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[2] = np.cos(psr[kk].theta)
            
            if ll != kk:
                xip = (1.-np.sum(phati*phatj)) / 2.
                ORF[ll, kk] = 3.*( 1./3. + xip * ( np.log(xip) -1./6.) )
            else:
                ORF[ll, kk] = 2.0

    return ORF


def twoComponentNoiseLike(Amp, D, c, b=1):
    """

    Likelihood function for two component noise model

    @param Amp: trial amplitude in GW units
    @param D: Vector of eigenvalues from diagonalized red noise
              covariance matrix
    @param c: Residuals written new diagonalized basis
    @param b: constant factor multiplying B matrix in total covariance C = aA + bB

    @return: loglike: The log-likelihood for this pulsar

    """

    loglike = -0.5 * np.sum(np.log(2*np.pi*(Amp**2*D + 1)) + c**2/(Amp**2*D + 1)) 

    return loglike

def angularSeparation(theta1, phi1, theta2, phi2):
    """
    Calculate the angular separation of two points on the sky.

    @param theta1: Polar angle of point 1 [radian]
    @param phi1: Azimuthal angle of point 1 [radian]
    @param theta2: Polar angle of point 2 [radian]
    @param phi2: Azimuthal angle of point 2 [radian]

    @return: Angular separation in radians

    """
    
    # unit vectors
    rhat1 = phat = [np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1)]
    rhat2 = phat = [np.sin(theta2)*np.cos(phi2), np.sin(theta2)*np.sin(phi2), np.cos(theta2)]

    cosMu = np.dot(rhat1, rhat2)

    return np.arccos(cosMu)

def weighted_values(values, probabilities, size):
    """
    Draw a weighted value based on its probability

    @param values: The values from which to choose
    @param probabilities: The probability of choosing each value
    @param size: The number of values to return

    @return: size values based on their probabilities

    """

    bins = np.add.accumulate(probabilities)
    return values[np.digitize(np.random.random_sample(size), bins)]

def computeNormalizedCovarianceMatrix(cov):
    """
    Compute the normalized covariance matrix from the true covariance matrix

    @param cov: covariance matrix

    @return: cnorm: normalized covaraince matrix

    """

    # get size of array
    ndim = cov.shape[0]
    cnorm = np.zeros((ndim, ndim))

    # compute normalized covariance matrix
    for ii in range(ndim):
        for jj in range(ndim):
            cnorm[ii,jj] = cov[ii,jj]/np.sqrt(cov[ii,ii]*cov[jj,jj])

    return cnorm 


def createfourierdesignmatrix(t, nmodes, freq=False, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    @param t: vector of time series in seconds
    @param nmodes: number of fourier coefficients to use
    @param freq: option to output frequencies
    @param Tspan: option to some other Tspan

    @return: F: fourier design matrix
    @return: f: Sampling frequencies (if freq=True)

    """

    N = len(t)
    F = np.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    f = np.linspace(1/T, nmodes/T, nmodes)
    Ffreqs = np.zeros(2*nmodes)
    Ffreqs[0::2] = f
    Ffreqs[1::2] = f

    # The sine/cosine modes
    ct = 0
    for ii in range(0, 2*nmodes-1, 2):
        
        F[:,ii] = np.cos(2*np.pi*f[ct]*t)
        F[:,ii+1] = np.sin(2*np.pi*f[ct]*t)
        ct += 1
    
    if freq:
        return F, Ffreqs
    else:
        return F

def createGWB(psr, Amp, gam, DM=False, noCorr=False, seed=None):
    """
    Function to create GW incuced residuals from a stochastic GWB as defined
    in Chamberlin, Creighton, Demorest et al. (2013)
    
    @param psr: pulsar object for single pulsar
    @param Amp: Amplitude of red noise in GW units
    @param gam: Red noise power law spectral index
    @param DM: Add time varying DM as a power law (only valid for single pulsars)
    @param noCorr: Add red noise with no spatial correlations
    
    @return: Vector of induced residuals
    
    """

    if seed is not None:
        np.random.seed(seed)

    # get maximum number of points
    npts = np.max([p.ntoa for p in psr])
    
    # get maximum number of epochs
    #npts = np.max([exploderMatrix(p.toas)[1].shape[1] for p in psr])

    Npulsars = len(psr)

    # current Hubble scale in units of 1/day
    H0=(2.27e-18)*(60.*60.*24.)

    # create simulated GW time span (start and end times). Will be slightly larger than real data span

    #gw start and end times for entire data set
    start = np.min([p.toas.min() for p in psr]) - 86400
    stop = np.max([p.toas.max() for p in psr]) + 86400
        
    # define "how much longer" or howml variable, needed because IFFT cannot quite match the value of the integral of < |r(t)|^2 > 
    howml = 10.

    # duration of the signal, spanning total time data taken in days
    dur = stop - start

    # make a vector of evenly sampled data points
    ut = np.linspace(start, stop, npts)

    # time resolution in days
    dt = dur/npts

    # compute the overlap reduction function
    if noCorr:
        ORF = np.diag(np.ones(Npulsars)*2)
    else:
        ORF = computeORFMatrix(psr)

    # define frequencies spanning from DC to Nyquist. This is a vector spanning these frequencies in increments of 1/(dur*howml).
    f=np.arange(0, 1./(2.*dt), 1./(dur*howml))
    f[0] = f[1] # avoid divide by 0 warning

    Nf=len(f)

    # Use Cholesky transform to take 'square root' of ORF
    M=np.linalg.cholesky(ORF)

    # Create random frequency series from zero mean, unit variance, Gaussian distributions
    w = np.zeros((Npulsars, Nf), complex)
    for ll in range(Npulsars):
        w[ll,:] = np.random.randn(Nf) + 1j*np.random.randn(Nf)

    # Calculate strain spectral index alpha, beta
    alpha_f = -1./2.*(gam-3)

    # Value of secondary spectral index beta (note: beta = 2+2*alpha)
    beta_f=2.*alpha_f+2.

    # convert Amp to Omega
    f1yr_sec = 1./3.16e7
    Omega_beta = (2./3.)*(np.pi**2.)/(H0**2.)*float(Amp)**2*(1/f1yr_sec)**(2*alpha_f)

    # calculate GW amplitude Omega 
    Omega=Omega_beta*f**(beta_f)

    # Calculate frequency dependent pre-factor C(f)
    # could rewrite in terms of A instead of Omega for convenience.
    C=H0**2./(16.*np.pi**2)/(2.*np.pi)**2 * f**(-5.) * Omega * (dur * howml)

    ### injection residuals in the frequency domain
    Res_f=np.dot(M,w)
    for ll in range(Npulsars):
        Res_f[ll] = Res_f[ll] * C**(0.5)    #rescale by frequency dependent factor
        Res_f[ll,0] = 0						#set DC bin to zero to avoid infinities
        Res_f[ll,-1] = 0					#set Nyquist bin to zero also

    # Now fill in bins after Nyquist (for fft data packing) and take inverse FT
    Res_f2 = np.zeros((Npulsars, 2*Nf-2), complex)    # make larger array for residuals
    Res_t = np.zeros((Npulsars, 2*Nf-2))
    Res_f2[:,0:Nf] = Res_f[:,0:Nf]
    Res_f2[:, Nf:(2*Nf-2)] = np.conj(Res_f[:,(Nf-2):0:-1])
    Res_t = np.real(np.fft.ifft(Res_f2)/dt)

    #for ll in range(Npulsars):
    #    for kk in range(Nf):					# copies values into the new array up to Nyquist        
    #        Res_f2[ll,kk] = Res_f[ll,kk]

    #    for jj in range(Nf-2):					# pads the values bigger than Nyquist with frequencies back down to 1. Biggest possible index of this array is 2*Nf-3.
    #        Res_f2[ll,Nf+jj] = np.conj(Res_f[ll,(Nf-2)-jj])

    #    ## rows: each row corresponds to a pulsar
    #    ## columns: each col corresponds to a value of the time series containing injection signal.
    #    Res_t[ll,:]=np.real(np.fft.ifft(Res_f2[ll,:])/dt)     #ifft includes a factor of 1/N, so divide by dt to effectively multiply by df=1/T

    # shorten data and interpolate onto TOAs
    Res = np.zeros((Npulsars, npts))
    res_gw = []
    for ll in range(Npulsars):
        
        Res[ll,:] = Res_t[ll, 10:(npts+10)]
        f = interp.interp1d(ut, Res[ll,:], kind='linear')

        if DM and len(psr) == 1:
            print 'adding DM to toas'
            res_gw.append(f(psr[ll].toas)/((2.3687e-16)*psr[ll].freqs**2))
        else:
            res_gw.append(f(psr[ll].toas))

    return res_gw






