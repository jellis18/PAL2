from __future__ import division
import numpy as np
import scipy.special as ss
import scipy.linalg as sl
import scipy.integrate as si
import scipy.constants as sc
import scipy.interpolate as interp
try:
    import healpy as hp
except ImportError:
    print 'WARNING: No healpy installed'
    hp = None
import math
import numpy.polynomial.hermite as herm
from scipy.integrate import odeint
import numexpr as ne
import sys,os

SOLAR2S = sc.G / sc.c**3 * 1.98855e30
KPC2S = sc.parsec / sc.c * 1e3
MPC2S = sc.parsec / sc.c * 1e6


def compute_snr_mark6(x, Nvec, Tmat, cf):
    """
    Compute optimal SNR for mark6
    """

    # white noise term
    Nix = x / Nvec
    xNix = np.dot(x, Nix) 
    TNix = np.dot(Tmat.T, Nix)

    SigmaTNix = sl.cho_solve(cf, TNix)

    ret = xNix - np.dot(TNix, SigmaTNix)
    #print xNix,  np.dot(TNix, SigmaTNix)

    return np.sqrt(ret) 

def compute_snr_mark9(x, Nvec, Tmat, Qamp, Uinds, Sigma):
    """
    Compute optimal SNR for mark9
    """


    # white noise term
    Nx = python_block_shermor_0D(x, Nvec, Qamp, Uinds)
    xNx = np.dot(x, Nx)
    TNx = np.dot(Tmat.T, Nx)
    SigmaTNx = sl.cho_solve(cf, TNx)
    ret = xNx - np.dot(TNx, SigmaTNx)

    return np.sqrt(ret)

def get_snr_prior(model, snr0):

    snr = 0
    nfref = 0
    for ct, p in enumerate(model.psr):

        s = p.residuals - p.detresiduals
        if np.any(np.isnan(s)) or np.any(np.isinf(s)):
            return -np.inf

        nf = p.Ttmat.shape[1]

        if model.likfunc == 'mark9':
            snr += innerProduct_rr9(s, s, p.Nvec, p.Ttmat,
                                            model.Sigma[nfref:(nfref+nf),
                                                        nfref:(nfref+nf)], 
                                             p.Qamp, p.Uinds)
        else:
            try:
                snr += innerProduct_rr(s, s, p.Nvec, p.Ttmat,
                                       model.Sigma[nfref:(nfref+nf),
                                                   nfref:(nfref+nf)])
            except np.linalg.LinAlgError:
                return -np.inf

        nfref += nf

    snr = np.sqrt(snr)
    if np.isinf(snr) or np.isnan(snr):
        return -np.inf

    return np.log(3*snr/(4*snr0**2*(1+snr/(4*snr0))**5))


def innerProduct_rr9(x, y, Nvec, Tmat, Sigma, Qamp, Uinds):
    """
    Compute inner product using rank-reduced
    approximations for red noise/jitter 

    Compute: x^T N^{-1} y - x^T N^{-1} T \Sigma^{-1} T^T N^{-1} y

    :param x: vector timeseries 1
    :param y: vector timeseries 2
    :param Nvec: vector of white noise values
    :param Tmat: Modified design matrix including red noise/jitter
    :param Sigma: Sigma matrix (\varphi^{-1} + T^T N^{-1} T)

    :return: inner product (x|y)

    """
    Nx = python_block_shermor_0D(x, Nvec, Qamp, Uinds)
    Ny = python_block_shermor_0D(y, Nvec, Qamp, Uinds)
    xNy = np.dot(x, Ny)
    TNx = np.dot(Tmat.T, Nx)
    TNy = np.dot(Tmat.T, Ny)
    cf = sl.cho_factor(Sigma)
    SigmaTNy = sl.cho_solve(cf, TNy)
    ret = xNy - np.dot(TNx, SigmaTNy)

    return ret

def innerProduct_rr(x, y, Nvec, Tmat, Sigma, TNx=None, TNy=None):
    """
    Compute inner product using rank-reduced
    approximations for red noise/jitter 

    Compute: x^T N^{-1} y - x^T N^{-1} T \Sigma^{-1} T^T N^{-1} y

    :param x: vector timeseries 1
    :param y: vector timeseries 2
    :param Nvec: vector of white noise values
    :param Tmat: Modified design matrix including red noise/jitter
    :param Sigma: Sigma matrix (\varphi^{-1} + T^T N^{-1} T)
    :param TNx: T^T N^{-1} x precomputed
    :param TNy: T^T N^{-1} y precomputed

    :return: inner product (x|y)

    """

    # white noise term
    Ni = 1/Nvec
    xNy = np.dot(x*Ni, y)
    Nx, Ny = x*Ni, y*Ni
    
    if TNx == None and TNy == None: 
        TNx = np.dot(Tmat.T, Nx)
        TNy = np.dot(Tmat.T, Ny)


    cf = sl.cho_factor(Sigma)
    SigmaTNy = sl.cho_solve(cf, TNy)

    ret = xNy - np.dot(TNx, SigmaTNy)

    return ret

# compute f_p statistic
def fpStat(psr, f0):
    """ 
    Computes the Fp-statistic as defined in Ellis, Siemens, Creighton (2012)
    
    :param psr: List of pulsar object instances
    :param f0: Gravitational wave frequency

    :return: Value of the Fp statistic evaluated at f0

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

        N = np.array([np.dot(A[0,:], np.dot(p.invCov, p.residuals)), \
                      np.dot(A[1,:], np.dot(p.invCov, p.residuals))]) 
        
        # define M matrix M_ij=(A_i|A_j)
        for jj in range(2):
            for kk in range(2):
                M[jj,kk] = np.dot(A[jj,:], np.dot(p.invCov, A[kk,:]))
        
        # take inverse of M
        Minv = np.linalg.inv(M)
        fstat += 0.5 * np.dot(N, np.dot(Minv, N))

    # return F-statistic
    return fstat

# compute f_e-statistic
def feStat(psr, gwtheta, gwphi, f0):
    """ 
    Computes the F-statistic as defined in Ellis, Siemens, Creighton (2012)
    
    :param psr: List of pulsar object instances
    :param gwtheta: GW polar angle
    :param gwphi: GW azimuthal angle
    :param f0: Gravitational wave frequency

    :return: Value of the Fe statistic evaluated at gwtheta, phi, f0

    """
    
    npsr = len(psr)
    N = np.zeros(4)
    M = np.zeros((4,4))
    for ii, p in enumerate(psr):
        fplus, fcross, cosMu = createAntennaPatternFuncs(p, gwtheta, gwphi)

        # define A
        A = np.zeros((4, len(p.toas)))
        A[0,:] = fplus/f0**(1./3.) * np.sin(2*np.pi*f0*p.toas)
        A[1,:] = fplus/f0**(1./3.) * np.cos(2*np.pi*f0*p.toas)
        A[2,:] = fcross/f0**(1./3.) * np.sin(2*np.pi*f0*p.toas)
        A[3,:] = fcross/f0**(1./3.) * np.cos(2*np.pi*f0*p.toas)


        N += np.array([np.dot(A[0,:], np.dot(p.invCov, p.res)), \
                        np.dot(A[1,:], np.dot(p.invCov, p.res)), \
                        np.dot(A[2,:], np.dot(p.invCov, p.res)), \
                        np.dot(A[3,:], np.dot(p.invCov, p.res))]) 

        M += np.dot(A, np.dot(p.invCov, A.T))

    # inverse of M
    Minv = np.linalg.pinv(M)

    # Fe-statistic
    return 0.5 * np.dot(N, np.dot(Minv, N))



def createAntennaPatternFuncs(psr, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).

    :param psr: pulsar object for single pulsar
    :param gwtheta: GW polar angle in radians
    :param gwphi: GW azimuthal angle in radians

    :return: (fplus, fcross, cosMu), where fplus and fcross
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

def glitch_signal(gtime, gamp, gsign, t):
    """
    Glitch signal
    """
    # Define the heaviside function
    heaviside = lambda x: 0.5 * (np.sign(x) + 1)

    s = np.sign(gsign)
    amp = 10**gamp
    epoch = gtime * 86400

    # Return the time-series for the pulsar
    return amp * s * heaviside(t - epoch) * (t - epoch) 

def bwmsignal(parameters, raj, decj, t, corr='gr'):
    """
    Function that calculates the earth-term gravitational-wave burst-with-memory
    signal, as described in:
    Seto et al, van haasteren and Levin, phsirkov et al, Cordes and Jenet.
    This version uses the F+/Fx polarization modes, as verified with the
    Continuous Wave and Anisotropy papers. The rotation matrices were not very
    insightful anyway.
    parameter[0] = TOA time (MJD) the burst hits the earth
    parameter[1] = amplitude of the burst (strain h)
    parameter[2] = azimuthal angle (rad)    [0, 2pi]
    parameter[3] = polar angle (rad)        [0, pi]
    parameter[4] = polarisation angle (rad) [0, pi]
    raj = Right Ascension of the pulsar (rad)
    decj = Declination of the pulsar (rad)
    t = timestamps where the waveform should be returned
    returns the waveform as induced timing residuals (seconds)
    """

    if corr != 'mono':
        gwphi = parameters[2]
        gwdec = np.pi/2-parameters[3]
        gwpol = parameters[4]
    
    if corr == 'gr':
        pol = AntennaPattern(raj, decj, gwphi, gwdec, gwpol)
    elif corr == 'mono':
        pol = 1
    elif corr == 'dipole':
        pol = DipoleAntennaPattern(raj, decj, gwphi, gwdec, gwpol)
    elif corr == 'abs':
        pol = np.abs(AntennaPattern(raj, decj, gwphi, gwdec, gwpol))

    # Define the heaviside function
    heaviside = lambda x: 0.5 * (np.sign(x) + 1)

    # Return the time-series for the pulsar
    bwm = pol * (10**parameters[1]) * heaviside(t - parameters[0]) * \
            (t - parameters[0]) * 86400
    return bwm

def AntennaPattern(rajp, decjp, raj, decj, pol):
    """Return the antenna pattern for a given source position and
    pulsar position

    :param rajp:    Right ascension pulsar (rad) [0,2pi]
    :param decj:    Declination pulsar (rad) [-pi/2,pi/2]
    :param raj:     Right ascension source (rad) [0,2pi]
    :param dec:     Declination source (rad) [-pi/2,pi/2]
    :param pol:     Polarization angle (rad) [0,pi]
    """

    Omega = np.array([-np.cos(decj)*np.cos(raj), \
                      -np.cos(decj)*np.sin(raj), \
                      -np.sin(decj)]).flatten()

    mhat = np.array([-np.sin(raj), np.cos(raj), 0]).flatten()
    nhat = np.array([-np.cos(raj)*np.sin(decj), \
                     -np.sin(decj)*np.sin(raj), \
                     np.cos(decj)]).flatten()

    p = np.array([np.cos(rajp)*np.cos(decjp), \
                  np.sin(rajp)*np.cos(decjp), \
                  np.sin(decjp)]).flatten()

    Fp = 0.5 * (np.dot(nhat, p)**2 - np.dot(mhat, p)**2) / (1 + np.dot(Omega, p))
    Fc = np.dot(mhat, p) * np.dot(nhat, p) / (1 + np.dot(Omega, p))

    return np.cos(2*pol)*Fp + np.sin(2*pol)*Fc


def DipoleAntennaPattern(rajp, decjp, raj, decj, pol):
    """Return the dipole antenna pattern for a given source position and
    pulsar position

    :param rajp:    Right ascension pulsar (rad) [0,2pi]
    :param decj:    Declination pulsar (rad) [-pi/2,pi/2]
    :param raj:     Right ascension source (rad) [0,2pi]
    :param dec:     Declination source (rad) [-pi/2,pi/2]
    :param pol:     Polarization angle (rad) [0,2pi]
    """
    Omega = np.array([-np.cos(decj)*np.cos(raj), \
                      -np.cos(decj)*np.sin(raj), \
                      -np.sin(decj)])

    mhat = np.array([-np.sin(raj), np.cos(raj), 0])
    nhat = np.array([-np.cos(raj)*np.sin(decj), \
                     -np.sin(decj)*np.sin(raj), \
                     np.cos(decj)])

    p = np.array([np.cos(rajp)*np.cos(decjp), \
                  np.sin(rajp)*np.cos(decjp), \
                  np.sin(decjp)])

    return np.cos(pol) * np.dot(nhat, p) + \
            np.sin(pol) * np.dot(mhat, p)



def createResiduals(psr, gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc, pdist=None, \
                        pphase=None, psrTerm=True, evolve=False, phase_approx=True):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013.

    :param psr: pulsar object for single pulsar
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param mc: Chirp mass of SMBMB [solar masses]
    :param dist: Luminosity distance to SMBMB [Mpc]
    :param fgw: Frequency of GW (twice the orbital frequency) [Hz]
    :param phase0: Initial Phase of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param inc: Inclination of GW source [radians]
    :param pdist: Pulsar distance to use other than those in psr [kpc]
    :param pphase: Use pulsar phase to determine distance [radian]
    :param psrTerm: Option to include pulsar term [boolean] 
    :param evolve: Option to exclude evolution [boolean]

    :return: Vector of induced residuals

    """

    # get antenna pattern funcs and cosMu
    fplus, fcross, cosMu = createAntennaPatternFuncs(psr, gwtheta, gwphi)
    
    # get values from pulsar object
    toas = psr.toas
    if pdist is None and pphase is None:
        pdist = psr.dist
    elif pdist is None and pphase is not None:
        pdist = pphase/(2*np.pi*fgw*(1-cosMu)) / KPC2S
   

    # convert units
    mc *= SOLAR2S         # convert from solar masses to seconds
    dist *= MPC2S    # convert from Mpc to seconds
    pdist *= KPC2S   # convert from kpc to seconds
    
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
        if pphase is not None:
            print 'In here', pphase[ct]
            phase_p = pphase[ct] + omega_p * toas
        else:
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

def construct_wavelet(t, A, t0, f0, Q, phi0, idx=None):

    wave = np.zeros(len(t))

    # width of gaussian
    tau = Q / (2*np.pi*f0)

    # get time range
    tind = np.logical_and(t>=t0-4*tau, t<=t0+4*tau)

    wave[tind] = A * np.exp(-(2*np.pi*f0*(t[tind]-t0))**2/Q**2) * \
            np.cos(2*np.pi*f0*(t[tind]-t0)+phi0)

    # fileter
    if idx is not None:
        wave[~idx] = 0
    
    return wave

def construct_gw_wavelet(psr, gwtheta, gwphi, gwpsi, gweps, 
                         gwA, gwt0, gwf0, gwQ, gwphi0):
    
    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*gwpsi), np.cos(2*gwpsi)

    # unit vectors to GW source
    m = np.array([-singwphi, cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

    res = []
    for ct, p in enumerate(psr):

        # use definition from Sesana et al 2010 and Ellis et al 2012
        phat = np.array([np.sin(p.theta)*np.cos(p.phi), np.sin(p.theta)*np.sin(p.phi),\
                np.cos(p.theta)])

        fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
        fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
        cosMu = -np.dot(omhat, phat)

        rr = 0
        for A, t0, f0, Q, phi0 in zip(gwA, gwt0, gwf0, gwQ, gwphi0):

            wplus = construct_wavelet(p.toas, A, t0, f0, Q, phi0)
            wcross = gweps * construct_wavelet(p.toas, A, t0, f0, Q, phi0+3*np.pi/2)

            rr += fplus * (wplus*cos2psi - wcross*sin2psi) + \
                    fcross * (wplus*sin2psi + wcross*cos2psi)

        res.append(rr)

    return res

        

def createResidualsFast(psr, gwtheta, gwphi, mc, dist, fgw, phase0, 
                        psi, inc, pdist=None, pphase=None, psrTerm=True, 
                        evolve=False, phase_approx=True, tref=0, 
                        add_random_phase=False):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013. Trys to be smart about it

    :param psr: list of pulsar objects for all pulsars
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param mc: Chirp mass of SMBMB [solar masses]
    :param dist: Luminosity distance to SMBMB [Mpc]
    :param fgw: Frequency of GW (twice the orbital frequency) [Hz]
    :param phase0: Initial Phase of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param inc: Inclination of GW source [radians]
    :param pdist: Pulsar distance to use other than those in psr [kpc]
    :param pphase: Use pulsar phase to determine distance [radian]
    :param psrTerm: Option to include pulsar term [boolean] 
    :param evolve: Option to exclude evolution [boolean]
    :param add_random_phase: Option to include random phase in waveform to break coherence [boolean]

    :return: Vector of induced residuals

    """

    # convert units
    mc *= SOLAR2S         # convert from solar masses to seconds
    dist *= MPC2S    # convert from Mpc to seconds

    # define initial orbital frequency 
    w0 = np.pi * fgw
    phase0 /= 2 # orbital phase
    w053 = w0**(-5/3)

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)
    incfac1, incfac2 = 0.5*(3+np.cos(2*inc)), 2*np.cos(inc)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
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
        toas = p.toas - tref
        if pdist is None and pphase is None:
            pd = p.pdist
        elif pdist is None and pphase is not None:
            pd = pphase[ct]/(2*np.pi*fgw*(1-cosMu)) / KPC2S
        else:
            pd = pdist[ct]
        

        # convert units
        pd *= KPC2S   # convert from kpc to seconds
        
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
            if pphase is not None:
                phase_p = phase0 + pphase[ct] + omega_p * toas
            else:
                phase_p = phase0 + fac2 * (w053 - omega_p**(-5/3)) + omega_p*toas
              
        # no evolution
        else: 
            
            # monochromatic
            omega = w0
            omega_p = omega
            
            # phases
            phase = phase0 + omega * toas
            phase_p = phase0 + omega * tp
            
        # add random phase?
        if add_random_phase:
            phase += np.random.uniform(0, 2*np.pi)
            phase_p += np.random.uniform(0, 2*np.pi)

        # define time dependent coefficients
        At = np.sin(2*phase) * incfac1
        Bt = np.cos(2*phase) * incfac2
        At_p = np.sin(2*phase_p) * incfac1
        Bt_p = np.cos(2*phase_p) * incfac2

        # now define time dependent amplitudes
        alpha = fac3 / omega**(1/3)
        alpha_p = fac3 / omega_p**(1/3)

        # define rplus and rcross
        rplus = alpha * (At*cos2psi + Bt*sin2psi)
        rcross = alpha * (-At*sin2psi + Bt*cos2psi)
        rplus_p = alpha_p * (At_p*cos2psi + Bt_p*sin2psi)
        rcross_p = alpha_p * (-At_p*sin2psi + Bt_p*cos2psi)

        # residuals
        if psrTerm:
            res.append(fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross))
        else:
            res.append(-fplus*rplus - fcross*rcross)

    return res

def createResidualsFree(psr, gwtheta, gwphi, h, fgw, phase0, psi, inc,
                               pphase0, pfgw, psrTerm=True, tref=0):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013. Trys to be smart about it

    :param psr: list of pulsar objects for all pulsars
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param h: GW strain
    :param fgw: Frequency of GW (twice the orbital frequency) [Hz]
    :param phase0: Initial Phase of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param pphase0: pulsar phase
    :param pfgw: pulsar term frequency
    :param inc: Inclination of GW source [radians]
    :param psrTerm: Option to include pulsar term [boolean] 

    :return: Vector of induced residuals

    """

    # define initial orbital frequency 
    w0 = np.pi * fgw
    phase0 /= 2 # orbital phase
    wp = np.pi * pfgw
    pphase0 /= 2

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)
    incfac1, incfac2 = -0.5*(3+np.cos(2*inc)), 2*np.cos(inc)

    # unit vectors to GW source
    m = np.array([-singwphi, cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])

    res = []
    for ct, p in enumerate(psr):

        # use definition from Sesana et al 2010 and Ellis et al 2012
        phat = np.array([np.sin(p.theta)*np.cos(p.phi), np.sin(p.theta)*np.sin(p.phi),\
                np.cos(p.theta)])

        fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
        fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
        cosMu = -np.dot(omhat, phat)
    
        # phases
        phase = phase0 + w0 * (p.toas - tref)
        phase_p = phase0 + pphase0[ct] + wp[ct] * (p.toas - tref)
            
        # define time dependent coefficients
        At = np.sin(2*phase) * incfac1
        Bt = np.cos(2*phase) * incfac2
        At_p = np.sin(2*phase_p) * incfac1
        Bt_p = np.cos(2*phase_p) * incfac2

        # now define time dependent amplitudes
        alpha = 0.5 * h / w0
        alpha_p = 0.5 * h / w0**(2/3) / wp[ct]**(1/3)

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


def constructShapelet(times, t0, q, amps):
    """
    Construct shapelet.

    :param times: sample times
    :param t0: event time
    :param q: width of event
    :param amps: vector of amplitudes for different components
    """
   
    hermcoeff = []
    for ii in range(len(amps)):
        hermcoeff.append(amps[ii] / np.sqrt(2**ii*ss.gamma(ii+1)*np.sqrt(2*np.pi)))
        
    # evaluate hermite polynomial sums
    hermargs = (times-t0)/q
    hval = herm.hermval(hermargs, np.array(hermcoeff)) * np.exp(-hermargs**2/2)
    
    return hval



def computeLuminosityDistance(z):
    """

    Compute luminosity distance via gaussian quadrature.

    :param z: Redshift at which to calculate luminosity distance

    :return: Luminosity distance in Mpc

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

    :param psr: Pulsar class containing all info on pulsar
    :param data: The residual data (or the template if we want to optimal SNR)
    :param temp: The template model of the signal

    :return: SNR

    """

    return np.dot(data, np.dot(psr.invCov, temp))/np.sqrt(np.dot(temp, np.dot(psr.invCov, temp)))

def createRmatrix(designmatrix, err):
    """
    Create R matrix as defined in Ellis et al (2013) and Demorest et al (2012)

    :param designmatrix: Design matrix as returned by tempo2

    :return: R matrix 
   
    """

    W = np.diag(1/err)
    w = 1/err

    u, s, v = sl.svd((w * designmatrix.T).T,full_matrices=False)

    return np.eye(len(err)) - (1/w * np.dot(u, np.dot(u.T, W)).T).T 


def createGmatrix(designmatrix):
    """
    Return G matrix as defined in van Haasteren et al 2013

    :param designmatrix: Design matrix as returned by tempo2

    :return: G matrix as defined in van Haasteren et al 2013

    """

    nfit = designmatrix.shape[1]
    npts = designmatrix.shape[0]

    # take singular value decomp
    u, s, v = sl.svd(designmatrix, full_matrices=True)

    return u[:,-(npts-nfit):]

def createQSDdesignmatrix(toas, norm=True):
    """
    Return designmatrix for QSD model

    :param toas: vector of TOAs in seconds

    :return: M design matrix for QSD

   """

    designmatrix = np.zeros((len(toas), 3))

    for ii in range(3):
        designmatrix[:,ii] = (toas/toas.mean())**(ii)

    return designmatrix

def createDesignmatrix(toas, freqs=None, RADEC=False, PX=False, DMX=False):
    """
    Return designmatrix for QSD model

    :param psr: Pulsar class

    :return: M design matrix for QSD

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
        if model[ii] == 'DMX' and freqs is not None:
            designmatrix[:,ii:] = DMXDesignMatrix(toas, freqs, dt=43200)
            break

    return designmatrix


def createTimeLags(toa1, toa2, round=True):
    """
    Create matrix of time lags tm = |t_i - t_j|

    :param toa1: times-of-arrival in seconds for psr 1
    :param toa2: times-of-arrival in seconds for psr 2
    :param round: option to round time difference to 0 if less than 1 hr

    :return: matrix of time lags tm = |t_i - t_j|

    """

    t1, t2 = np.meshgrid(toa2, toa1)

    tm = np.abs(t1-t2)

    if round:
        hr = 3600. # hour in seconds
        tm = np.where(tm<hr, 0.0, tm)
        
    return tm

def exploderMatrix(times, freqs=None, dt=1.0, flags=None):
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
    if freqs is not None:
        avefreqs = np.array([np.mean(freqs[l]) for l in bucket_ind],'d')

    
    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
        
    if freqs is not None and flags is not None:
        return avetoas, avefreqs, aveflags, U
    elif freqs is not None and flags is None:
        return avetoas, avefreqs, U
    elif freqs is None and flags is not None:
        return avetoas, aveflags, U
    else:
        return avetoas, U


def constructSEkernel(times, lam, amp):

    tm = createTimeLags(times, times)
    K = amp**2 * np.exp(-tm**2/2/lam**2)

    return K    


def exploderMatrixNoSingles(times, flags, dt=10):
    isort = np.argsort(times)
    
    bucket_ref = [[times[isort[0]], flags[isort[0]]]]
    bucket_ind = [[isort[0]]]
        
    for i in isort[1:]:
        if times[i] - bucket_ref[-1][0] < dt and flags[i] == bucket_ref[-1][1]:
            if 'ABPP-L' in flags[i]:
                bucket_ref.append([times[i], flags[i]])
                bucket_ind.append([i])
            else:
                bucket_ind[-1].append(i)
        else:
            bucket_ref.append([times[i], flags[i]])
            bucket_ind.append([i])
        

    # find only epochs with more than 1 TOA
    bucket_ind2 = [ind for ind in bucket_ind if len(ind) > 2]
    
    avetoas = np.array([np.mean(times[l]) for l in bucket_ind2],'d')
    aveflags = np.array([flags[l[0]] for l in bucket_ind2])

    
    U = np.zeros((len(times),len(bucket_ind2)),'d')
    for i,l in enumerate(bucket_ind2):
        U[l,i] = 1
        
    return avetoas, aveflags, U

    

def exploderMatrix_slow(toas, freqs=None, dt=1200, flags=None):
    """
    Compute exploder matrix for daily averaging

    :param toas: array of toas
    :param dt: time offset (seconds)

    :return: exploder matrix and daily averaged toas

    """


    processed = np.array([0]*len(toas), dtype=np.bool)  # No toas processed yet
    U = np.zeros((len(toas), 0))
    avetoas = np.empty(0)
    avefreqs = np.empty(0)
    aveflags = []

    while not np.all(processed):
        npindex = np.where(processed == False)[0]
        ind = npindex[0]
        print ind
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

def dailyAveMatrix(times, err, dt=10, flags=None):
    """
    Compute matrix for daily averaging

    :param toas: array of toas
    :param toas: array of toa errors
    :param dt: time offset (seconds)

    :return: exploder matrix and daily averaged toas

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

    
    U = np.zeros((len(times),len(bucket_ind)))
    aveerr = np.zeros(len(bucket_ind))
    for i,l in enumerate(bucket_ind):
        w = 1/err[l]**2
        aveerr[i] = np.sqrt(1/np.sum(w))
        U[l,i] =  w/np.sum(w)
        
    if flags is not None:
        return avetoas, aveerr, aveflags, U
    else:
        return avetoas, aveerr, U



def DMXDesignMatrix(toas, freqs, dt=1200):
    """
    Compute DMX Design matrix

    :param toas: array of toas
    :param freqs: array of frequencies in Hz
    :param dt: time offset (seconds)

    :return: Design matrix for DMX

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

    :param tm: Matrix of time lags in years
    :param fL: Low frequency cutoff
    :param gam: Power Law spectral index
    :param nsteps: Number of terms in the power series expansion
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

    :param tm: Matrix of time lags in years
    :param fL: Low frequency cutoff
    :param gam: Power Law spectral index
    """

    x = 2*np.pi*fL*tm

    sum = ne.evaluate("1/(1-gam) - x**2/(2*(3-gam)) + x**4/(24*(5-gam))")

    return sum

def createGHmatrix(toa, err, res, G, fidelity, Amp = None):
    """
    Create "H" compression matrix as defined in van Haasteren 2013(b).
    Multiplies with "G" matrix to create the "GH" matrix, which can simply replace
    the "G" matrix in all likelihoods which are marginalised over the timing-model


    :param toa: times-of-arrival (in seconds) for psr
    :param err: error bars on toas (in seconds)
    :param res: residuals (in seconds) of psr
    :param G: G matrix as defined in van Haasteren et al 2013(a)
    :param fidelity: fraction of total sensitivity retained in compressed data

    :return: GH matrix, which can simply replace "G" matrix in likelihood

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
    if Amp is None:
        sigma_gwb = np.std(res) * 1e-15 * 1e9
        Amp = sigma_gwb * 0.89 * Tspan**(-5./3)
    #Amp = (sigma_gwb/(1.37*(10**(-9)))) / (Tspan**(5/3))
    
    # looping over eigenvalues until the fidelity criterion of van Haasteren 2013(b) 
    # is satisfied; only the 'principal' eigenvectors are retained
    fisherelements = eigVal**2 / (1 + Amp**2 * eigVal)**2
    cumev = np.cumsum(fisherelements)
    totrms = np.sum(fisherelements)

    l = int((np.flatnonzero( (cumev/totrms) >= fidelity )[0] + 1))
    #index = np.amax(np.where(np.cumsum((eigVal/(1+(Amp**2.0)*eigVal))**2.0)/ \
    #                         np.sum((eigVal/(1.0+(Amp**2.0)*eigVal))**2.0).real \
    #                         <= fidelity)[0]) 
    
    # forming the data-compression matrix
    H = np.dot(sl.sqrtm(sl.inv(GCnoiseG)).real,eigVec.T[:l].T.real)
    
    return np.dot(G,H)



def createRedNoiseCovarianceMatrix(tm, Amp, gam, fH=None, fast=False):
    """
    Create red noise covariance matrix. If fH is None, then
    return standard power law covariance matrix. If fH is not
    none, return power law covariance matrix with high frequency 
    cutoff.

    :param tm: Matrix of time lags in seconds
    :param Amp: Amplitude of red noise in GW units
    :param gam: Red noise power law spectral index
    :param fH: Optional high frequency cutoff in yr^-1
    :param fast: Option to use Python numexpr to speed 
                    up calculation (default = True)

    :return: Red noise covariance matrix in seconds^2

    """

    # conversion from seconds to years
    s2yr = 1/3.16e7
    
    # compute high frequency cutoff
    Tspan = tm.max() * s2yr
    fL = 1/(10*Tspan)


    if fH is None:

        # convert amplitude to correct units
        A = Amp**2/24/np.pi**2
        if fast:
            x = 2 *np.pi * fL * tm * s2yr
            corr = (2*A/(fL**(gam-1)))*((ss.gamma(1-gam)* \
                np.sin(np.pi*gam/2)*ne.evaluate("x**(gam-1)")) \
                -sumTermCovarianceMatrix_fast(tm*s2yr, fL, gam))
        else:
            x = 2 *np.pi * fL * tm * s2yr
            corr = (2*A/(fL**(gam-1)))*((ss.gamma(1-gam)* \
                np.sin(np.pi*gam/2)*(x)**(gam-1)) \
                -sumTermCovarianceMatrix(tm*s2yr, fL, gam, 5))

    elif fH is not None:

        alpha=(3-gam)/2.0

        # convert amplitude to correct units
        A = Amp**2
 
        EulerGamma=0.577

        x = 2*np.pi*fL*tm * s2yr

        norm = (fL**(2*alpha - 2)) * 2**(alpha - 3) / \
            (3 * np.pi**1.5 * ss.gamma(1.5 - alpha))

        # introduce a high-frequency cutoff
        xi = (fH/s2yr)/fL
        
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

    :param err: Error bars on toas in seconds
    :param efac: Multiplier on error bar component of covariance matrix
    :param equad: Extra toa independent white noise in seconds
    :param tau: Extra time scale of correlation if appropriate. If this
                parameter is specified must also read in matrix of time lags
    :param tm: Matrix of time lags.

    :return: White noise covariance matrix in seconds^2

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

    :param N: number of pulsars in the search
    :param fp0: The measured value of the Fp-statistic

    :returns: False alarm probability ad defined in Eq (64)
              of Ellis, Seiemens, Creighton (2012)

    """

    n = np.arange(0,N)

    return np.sum(np.exp(n*np.log(fp0)-fp0-np.log(ss.gamma(n+1))))

def dailyAverage(pulsar):
    """

    Function to compute daily averaged residuals such that we
    have one residual per day per frequency band.

    :param pulsar: pulsar class from Michele Vallisneri's 
                     libstempo library.

    :return: mtoas: Average TOA of a single epoch
    :return: qmatrix: Linear operator that transforms residuals to
                      daily averaged residuals
    :return: merr: Daily averaged error bar
    :return: mfreqs: Daily averaged frequency value
    :return: mbands: Frequency band for daily averaged residual

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

    :param psr: List of pulsar object instances

    :return: Numpy array of pairwise ORF values for every pulsar
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

    :param psr: List of pulsar object instances

    :return: Matrix that has the ORF values for every pulsar
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

    :param Amp: trial amplitude in GW units
    :param D: Vector of eigenvalues from diagonalized red noise
              covariance matrix
    :param c: Residuals written new diagonalized basis
    :param b: constant factor multiplying B matrix in total covariance C = aA + bB

    :return: loglike: The log-likelihood for this pulsar

    """

    loglike = -0.5 * np.sum(np.log(2*np.pi*(Amp**2*D + 1)) + c**2/(Amp**2*D + 1)) 

    return loglike

def angularSeparation(theta1, phi1, theta2, phi2):
    """
    Calculate the angular separation of two points on the sky.

    :param theta1: Polar angle of point 1 [radian]
    :param phi1: Azimuthal angle of point 1 [radian]
    :param theta2: Polar angle of point 2 [radian]
    :param phi2: Azimuthal angle of point 2 [radian]

    :return: Angular separation in radians

    """
    
    # unit vectors
    rhat1 = np.array([np.sin(theta1)*np.cos(phi1),
                    np.sin(theta1)*np.sin(phi1), 
                    np.cos(theta1)]).flatten()
    rhat2 = np.array([np.sin(theta2)*np.cos(phi2), 
                    np.sin(theta2)*np.sin(phi2), 
                    np.cos(theta2)]).flatten()

    cosMu = np.dot(rhat1, rhat2)

    return np.arccos(cosMu)

def weighted_values(values, probabilities, size):
    """
    Draw a weighted value based on its probability

    :param values: The values from which to choose
    :param probabilities: The probability of choosing each value
    :param size: The number of values to return

    :return: size values based on their probabilities

    """

    bins = np.add.accumulate(probabilities)
    return values[np.digitize(np.random.random_sample(size), bins)]

def computeNormalizedCovarianceMatrix(cov):
    """
    Compute the normalized covariance matrix from the true covariance matrix

    :param cov: covariance matrix

    :return: cnorm: normalized covaraince matrix

    """

    # get size of array
    ndim = cov.shape[0]
    cnorm = np.zeros((ndim, ndim))

    # compute normalized covariance matrix
    for ii in range(ndim):
        for jj in range(ndim):
            cnorm[ii,jj] = cov[ii,jj]/np.sqrt(cov[ii,ii]*cov[jj,jj])

    return cnorm 


def createfourierdesignmatrix(t, nmodes, freq=False, Tspan=None,
                              logf=False, fmin=None, fmax=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    :param t: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing

    :return: F: fourier design matrix
    :return: f: Sampling frequencies (if freq=True)

    """

    N = len(t)
    F = np.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    if fmin is not None and fmax is not None:
        f = np.linspace(fmin, fmax, nmodes)
    else:
        f = np.linspace(1/T, nmodes/T, nmodes)
    if logf:
        f = np.logspace(np.log10(1/T), np.log10(nmodes/T), nmodes)
        #f = np.logspace(np.log10(1/2/T), np.log10(nmodes/T), nmodes)
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

def singlefourierdesignmatrix(t, freqs):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013
    for a given set of frequencies

    :param t: vector of time series in seconds
    :param freqs: Frequencies at which to evaluate fourier components

    :return: F: fourier design matrix

    """

    N = len(t)
    F = np.zeros((N, 2*len(freqs)))


    for ii in range(len(freqs)):
        F[:,2*ii] = np.cos(2*np.pi*freqs[ii]*t)
        F[:,2*ii+1] = np.sin(2*np.pi*freqs[ii]*t)

    return F

def createGWB(psr, Amp, gam, DM=False, noCorr=False, seed=None, turnover=False, f0=1e-9, \
              beta=1, power=1, interpolate=True):
    """
    Function to create GW incuced residuals from a stochastic GWB as defined
    in Chamberlin, Creighton, Demorest et al. (2013)
    
    :param psr: pulsar object for single pulsar
    :param Amp: Amplitude of red noise in GW units
    :param gam: Red noise power law spectral index
    :param DM: Add time varying DM as a power law (only valid for single pulsars)
    :param noCorr: Add red noise with no spatial correlations
    
    :return: Vector of induced residuals
    
    """

    if seed is not None:
        np.random.seed(seed)

    # get maximum number of points
    #npts = np.max([len(p.toas) for p in psr])
    
    # get maximum number of epochs
    npts = 300
    #npts = np.max([p.avetoas for p in psr])

    Npulsars = len(psr)

    # current Hubble scale in units of 1/day
    H0=(2.27e-18)*(60.*60.*24.)

    # create simulated GW time span (start and end times). 
    # Will be slightly larger than real data span

    #gw start and end times for entire data set
    start = np.min([p.toas.min() for p in psr]) - 86400
    stop = np.max([p.toas.max() for p in psr]) + 86400
        
    # define "how much longer" or howml variable, needed because IFFT 
    # cannot quite match the value of the integral of < |r(t)|^2 > 
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

    # strain amplitude
    f1yr = 1/3.16e7
    alpha = -0.5 * (gam-3)
    hcf = Amp * (f/f1yr)**(alpha)
    if turnover:
        si = alpha - beta
        hcf /= (1+(f/f0)**(power*si))**(1/power)

    C = 1 / 96 / np.pi**2 * hcf**2 / f**3 * dur * howml

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

        if interpolate:
            f = interp.interp1d(ut, Res[ll,:], kind='linear')

            if DM and len(psr) == 1:
                print 'adding DM to toas'
                res_gw.append(f(psr[ll].toas)/((2.3687e-16)*psr[ll].freqs**2))
            else:
                res_gw.append(f(psr[ll].toas))
        else:
            ntoa = len(psr[ll].toas)
            res = Res_t[ll,10:(ntoa+10)]
            if DM and len(psr) == 1:
                print 'adding DM to toas'
                res_gw.append(res/((2.3687e-16)*psr[ll].freqs**2))
            else:
                res_gw.append(res)

    return res_gw

def createGWB_clean(psr, Amp, gam, noCorr=False, seed=None, turnover=False, \
                    f0=1e-9, beta=1, power=1, npts=600, howml=10):
    """
    Function to create GW incuced residuals from a stochastic GWB as defined
    in Chamberlin, Creighton, Demorest et al. (2013)
    
    :param psr: pulsar object for single pulsar
    :param Amp: Amplitude of red noise in GW units
    :param gam: Red noise power law spectral index
    :param noCorr: Add red noise with no spatial correlations
    :param seed: Random number seed
    :param turnover: Produce spectrum with turnover at frequency f0
    :param f0: Frequency of spectrum turnover
    :param beta: Spectral index of power spectram for f << f0
    :param power: Fudge factor for flatness of spectrum turnover
    :param npts: Number of points used in interpolation
    :param howml: Lowest frequency is 1/(howml * T) 

    
    :return: list of residuals for each pulsar
    
    """

    if seed is not None:
        np.random.seed(seed)

    # number of pulsars
    Npulsars = len(psr)

    # current Hubble scale in units of 1/day
    H0=(2.27e-18)*(60.*60.*24.)

    #gw start and end times for entire data set
    start = np.min([p.toas.min() for p in psr]) - 86400
    stop = np.max([p.toas.max() for p in psr]) + 86400
        
    # duration of the signal
    dur = stop - start
    
    # get maximum number of points
    if npts is None:
        # default to cadence of 2 weeks
        npts = dur/(86400*14)

    # make a vector of evenly sampled data points
    ut = np.linspace(start, stop, npts)

    # time resolution in days
    dt = dur/npts

    # compute the overlap reduction function
    if noCorr:
        ORF = np.diag(np.ones(Npulsars)*2)
    else:
        ORF = computeORFMatrix(psr)

    # Define frequencies spanning from DC to Nyquist. 
    # This is a vector spanning these frequencies in increments of 1/(dur*howml).
    f=np.arange(0, 1/(2*dt), 1/(dur*howml))
    f[0] = f[1] # avoid divide by 0 warning
    Nf = len(f)

    # Use Cholesky transform to take 'square root' of ORF
    M = np.linalg.cholesky(ORF)

    # Create random frequency series from zero mean, unit variance, Gaussian distributions
    w = np.zeros((Npulsars, Nf), complex)
    for ll in range(Npulsars):
        w[ll,:] = np.random.randn(Nf) + 1j*np.random.randn(Nf)

    # strain amplitude
    f1yr = 1/3.16e7
    alpha = -0.5 * (gam-3)
    hcf = Amp * (f/f1yr)**(alpha)
    if turnover:
        si = alpha - beta
        hcf /= (1+(f/f0)**(power*si))**(1/power)

    C = 1 / 96 / np.pi**2 * hcf**2 / f**3 * dur * howml

    ### injection residuals in the frequency domain
    Res_f = np.dot(M, w)
    for ll in range(Npulsars):
        Res_f[ll] = Res_f[ll] * C**(0.5)    # rescale by frequency dependent factor
        Res_f[ll,0] = 0			    # set DC bin to zero to avoid infinities
        Res_f[ll,-1] = 0		    # set Nyquist bin to zero also

    # Now fill in bins after Nyquist (for fft data packing) and take inverse FT
    Res_f2 = np.zeros((Npulsars, 2*Nf-2), complex)    
    Res_t = np.zeros((Npulsars, 2*Nf-2))
    Res_f2[:,0:Nf] = Res_f[:,0:Nf]
    Res_f2[:, Nf:(2*Nf-2)] = np.conj(Res_f[:,(Nf-2):0:-1])
    Res_t = np.real(np.fft.ifft(Res_f2)/dt)

    # shorten data and interpolate onto TOAs
    Res = np.zeros((Npulsars, npts))
    res_gw = []
    for ll in range(Npulsars):
        
        Res[ll,:] = Res_t[ll, 10:(npts+10)]
        f = interp.interp1d(ut, Res[ll,:], kind='linear')
        res_gw.append(f(psr[ll].toas))

    return res_gw

def real_sph_harm(ll, mm, phi, theta):
    """
    The real-valued spherical harmonics
    ADAPTED FROM vH piccard CODE
    """
    if mm>0:
        ans = (1./np.sqrt(2)) * \
                (ss.sph_harm(mm, ll, phi, theta) + \
                ((-1)**mm) * ss.sph_harm(-mm, ll, phi, theta))
    elif mm==0:
        ans = ss.sph_harm(0, ll, phi, theta)
    elif mm<0:
        ans = (1./(np.sqrt(2)*complex(0.,1))) * \
                (ss.sph_harm(-mm, ll, phi, theta) - \
                ((-1)**mm) * ss.sph_harm(mm, ll, phi, theta))

    return ans.real

def SetupSkymapPlottingGrid(lmax, skypos):
    """
    Compute the real spherical harmonics
    on a sky-grid defined by healpy for
    plotting purposes.
    """
    
    harmvals = [[0.0]*(2*ll+1) for ll in range(lmax+1)]
    for ll in range(len(harmvals)):
        for mm in range(len(harmvals[ll])):
            harmvals[ll][mm] = real_sph_harm(ll,mm-ll,skypos[:,1],skypos[:,0])

    return harmvals

def GWpower(clm, harmvals):
    """
    Construct the GW power flowing into each pixel
    """

    Pdist=0.
    for ll in range(len(harmvals)):
        for mm in range(len(harmvals[ll])):
            Pdist += clm[ ll**2 + mm ] * harmvals[ll][mm]
    
    return Pdist


def SetupPriorSkyGrid(lmax):
    """
    Check whether these anisotropy coefficients correspond to a physical
    angular-distribution of the metric-perturbation quadratic
    expectation-value.
    """
    ngrid_phi = 40
    ngrid_costheta = 40
    
    phi = np.arange(0.0,2.0*np.pi,2.0*np.pi/ngrid_phi)
    theta = np.arccos(np.arange(-1.0,1.0,2.0/ngrid_costheta))

    xx, yy = np.meshgrid(phi,theta)

    harm_sky_vals = [[0.0]*(2*ll+1) for ll in range(lmax+1)]
    for ll in range(len(harm_sky_vals)):
        for mm in range(len(harm_sky_vals[ll])):
            #print ll, mm-ll, real_sph_harm(ll,mm-ll,xx,yy)
            harm_sky_vals[ll][mm] = real_sph_harm(ll,mm-ll,xx,yy)

    return harm_sky_vals

def PhysPrior(clm,harm_sky_vals):
    """
    Check whether these anisotropy coefficients correspond to a physical
    angular-distribution of the metric-perturbation quadratic
    expectation-value.
    """
    """ngrid_phi = 20
    ngrid_costheta = 20
    
    phi = np.arange(0.0,2.0*np.pi,2.0*np.pi/ngrid_phi)
    theta = np.arccos(np.arange(-1.0,1.0,2.0/ngrid_costheta))
    xx, yy = np.meshgrid(phi,theta)
    harm_sky_vals = [[0.0]*(2*ll+1) for ll in range(lmax+1)]
    for ll in range(len(harm_sky_vals)):
        for mm in range(len(harm_sky_vals[ll])):
            harm_sky_vals[ll][mm] = real_sph_harm(ll,mm-ll,xx,yy)
    """

    Pdist=0.
    for ll in range(len(harm_sky_vals)):
        for mm in range(len(harm_sky_vals[ll])):
            Pdist += clm[ ll**2 + mm ] * harm_sky_vals[ll][mm]

    if np.any(Pdist<0.)==True:
        return -np.inf
    else:
        return 0




def fixNoiseValues(ptasignals, vals, pars, bvary=False, verbose=True):
    """
    Use fixed noise values to read into 
    ptasignal dictionary. This will be much
    easier if everything is switched to using
    parids.

    """

    for ct, p in enumerate(pars):
        for sig in ptasignals:

            # efac
            if sig['stype'] == 'efac':
                sig['bvary'][0] = bvary
                flag = p.split('efac-')[-1]
                if flag in sig['flagvalue']:
                    if verbose:
                        print 'Setting efac {0} value to {1}'.format(sig['flagvalue'], \
                                                                     vals[ct])
                    sig['pstart'][0] = vals[ct]
                    sig['bvary'][0] = bvary
                elif flag == 'efac':
                    if verbose:
                        print 'Setting efac {0} value to {1}'.format(sig['flagvalue'], \
                                                                     vals[ct])
                    sig['pstart'][0] = vals[ct]
                    sig['bvary'][0] = bvary
                    
            # equad
            if sig['stype'] == 'equad':
                sig['bvary'][0] = bvary
                flag = p.split('equad-')[-1]
                if flag in sig['flagvalue']:
                    if vals[ct] > 0:
                        vals[ct] = np.log10(vals[ct])
                    if verbose:
                        print 'Setting equad {0} value to {1}'.format(sig['flagvalue'], \
                                                                      vals[ct])
                    sig['pstart'][0] = vals[ct]
                    sig['bvary'][0] = bvary
                elif flag == 'equad':
                    if verbose:
                        print 'Setting equad {0} value to {1}'.format(sig['flagvalue'], \
                                                                     vals[ct])
                    sig['pstart'][0] = vals[ct]
                    sig['bvary'][0] = bvary

            # jitter equad
            if sig['stype'] == 'jitter_equad':
                sig['bvary'][0] = bvary
                flag = p.split('jitter_q-')[-1]
                if flag in sig['flagvalue']:
                    if vals[ct] > 0:
                        vals[ct] = np.log10(vals[ct])
                    if verbose:
                        print 'Setting ecorr {0} value to {1}'.format(sig['flagvalue'], \
                                                                      vals[ct])
                    sig['pstart'][0] = vals[ct]
                    sig['bvary'][0] = bvary
                elif flag == 'jitter_q':
                    if verbose:
                        print 'Setting ecorr {0} value to {1}'.format(sig['flagvalue'], \
                                                                     vals[ct])
                    sig['pstart'][0] = vals[ct]
                    sig['bvary'][0] = bvary

            if sig['stype'] == 'powerlaw':
                sig['bvary'][0] = bvary
                sig['bvary'][1] = bvary
                if p == 'RN-Amplitude':
                    if verbose:
                        print 'Setting RN Amp value to {0}'.format(vals[ct])
                    sig['pstart'][0] = vals[ct]
                    sig['bvary'][0] = bvary
                if p == 'RN-spectral-index':
                    if verbose:
                        print 'Setting RN spectral index value to {0}'.format(vals[ct])
                    sig['pstart'][1] = vals[ct]
                    sig['bvary'][1] = bvary

    return ptasignals

def python_block_shermor_2D(Z, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiZ

    :param Z:       The design matrix, array (n x m)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: Z.T * N^-1 * Z
    """
    ni = 1.0 / Nvec
    zNz = np.dot(Z.T*ni, Z)

    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
                niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                zn = np.dot(niblock, Zblock)
                zNz -= beta * np.outer(zn.T, zn)

    return zNz

def python_block_shermor_2D2(Z, X, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiX

    :param Z:       The design matrix, array (n x m)
    :param X:       The second design matrix, array (n x l)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: Z.T * N^-1 * X
    """
    ni = 1.0 / Nvec
    zNx = np.dot(Z.T*ni, X)

    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
                Xblock = X[Uinds[cc,0]:Uinds[cc,1], :]
                niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                zn = np.dot(niblock, Zblock)
                xn = np.dot(niblock, Xblock)
                zNx -= beta * np.outer(zn.T, xn)

    return zNx

def python_block_shermor_0D(r, Nvec, Jvec, Uinds): 
    """
    Sherman-Morrison block-inversion for Jitter 
    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    
    ni = 1/Nvec
    Nx = r/Nvec
    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                rblock = r[Uinds[cc,0]:Uinds[cc,1]]
                niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                Nx[Uinds[cc,0]:Uinds[cc,1]] -= beta * np.dot(niblock, rblock) * niblock

    return Nx


def python_block_shermor_1D(r, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: r.T * N^-1 * r, log(det(N))
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    xNx = np.dot(r, r * ni)
    
    if len(np.atleast_1d(Jvec)) > 1:
        for cc, jv in enumerate(Jvec):
            if jv > 0.0:
                rblock = r[Uinds[cc,0]:Uinds[cc,1]]
                niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

                beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
                xNx -= beta * np.dot(rblock, niblock)**2
                Jldet += np.log(jv) - np.log(beta)

    return Jldet, xNx


def quantize_fast(times, dt=1.0, calci=False):
    """ Adapted from libstempo: produce the quantisation matrix fast """
    isort = np.argsort(times)
    
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    
    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    
    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    rv = (t, U)

    if calci:
        Ui = ((1.0/np.sum(U, axis=0)) * U).T
        rv = (t, U, Ui)

    return rv


def quantize_split(times, flags, dt=1.0, calci=False):
    """
    As quantize_fast, but now split the blocks per backend. Note: for
    efficiency, this function assumes that the TOAs have been sorted by
    argsortTOAs. This is _NOT_ checked.
    """
    isort = np.arange(len(times))
    
    bucket_ref = [times[isort[0]]]
    bucket_flag = [flags[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt and flags[i] == bucket_flag[-1]:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_flag.append(flags[i])
            bucket_ind.append([i])
    
    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    
    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    rv = (t, U)

    if calci:
        Ui = ((1.0/np.sum(U, axis=0)) * U).T
        rv = (t, U, Ui)

    return rv



def argsortTOAs(toas, flags, which=None, dt=1.0):
    """
    Return the sort, and the inverse sort permutations of the TOAs, for the
    requested type of sorting

    NOTE: This one is _not_ optimized for efficiency yet (but is done only once)

    :param toas:    The toas that are to be sorted
    :param flags:   The flags that belong to each TOA (indicates sys/backend)
    :param which:   Which type of sorting we will use (None, 'jitterext', 'time')
    :param dt:      Timescale for which to limit jitter blocks, default [10 secs]

    :return:    perm, perminv       (sorting permutation, and inverse)
    """
    if which is None:
        isort = slice(None, None, None)
        iisort = slice(None, None, None)
    elif which == 'time':
        isort = np.argsort(toas, kind='mergesort')
        iisort = np.zeros(len(isort), dtype=np.int)
        for ii, p in enumerate(isort):
            iisort[p] = ii
    elif which == 'jitterext':
        tave, Umat = quantize_fast(toas, dt)

        isort = np.argsort(toas, kind='mergesort')
        uflagvals = list(set(flags))

        for cc, col in enumerate(Umat.T):
            for flagval in uflagvals:
                flagmask = (flags[isort] == flagval)
                if np.sum(col[isort][flagmask]) > 1:
                    # This observing epoch has several TOAs
                    colmask = col[isort].astype(np.bool)
                    epmsk = flagmask[colmask]
                    epinds = np.flatnonzero(epmsk)
                    
                    if len(epinds) == epinds[-1] - epinds[0] + 1:
                        # Keys are exclusively in succession
                        pass
                    else:
                        # Sort the indices of this epoch and backend
                        # We need mergesort here, because it is stable
                        # (A stable sort keeps items with the same key in the
                        # same relative order. )
                        episort = np.argsort(flagmask[colmask], kind='mergesort')
                        isort[colmask] = isort[colmask][episort]
                else:
                    # Only one element, always ok
                    pass

        # Now that we have a correct permutation, also construct the inverse
        iisort = np.zeros(len(isort), dtype=np.int)
        for ii, p in enumerate(isort):
            iisort[p] = ii
    else:
        isort, iisort = np.arange(len(toas)), np.arange(len(toas))

    return isort, iisort

def checkTOAsort(toas, flags, which=None, dt=1.0):
    """
    Check whether the TOAs are indeed sorted as they should be according to the
    definition in argsortTOAs

    :param toas:    The toas that are supposed to be already sorted
    :param flags:   The flags that belong to each TOA (indicates sys/backend)
    :param which:   Which type of sorting we will check (None, 'jitterext', 'time')
    :param dt:      Timescale for which to limit jitter blocks, default [10 secs]

    :return:    True/False
    """
    rv = True
    if which is None:
        isort = slice(None, None, None)
        iisort = slice(None, None, None)
    elif which == 'time':
        isort = np.argsort(toas, kind='mergesort')
        if not np.all(isort == np.arange(len(isort))):
            rv = False
    elif which == 'jitterext':
        tave, Umat = quantize_fast(toas, dt)

        #isort = np.argsort(toas, kind='mergesort')
        isort = np.arange(len(toas))
        uflagvals = list(set(flags))

        for cc, col in enumerate(Umat.T):
            for flagval in uflagvals:
                flagmask = (flags[isort] == flagval)
                if np.sum(col[isort][flagmask]) > 1:
                    # This observing epoch has several TOAs
                    colmask = col[isort].astype(np.bool)
                    epmsk = flagmask[colmask]
                    epinds = np.flatnonzero(epmsk)
                    
                    if len(epinds) == epinds[-1] - epinds[0] + 1:
                        # Keys are exclusively in succession
                        pass
                    else:
                        # Keys are not sorted for this epoch/flag
                        rv = False
                else:
                    # Only one element, always ok
                    pass
    else:
        pass

    return rv


def checkquant(U, flags, uflagvals=None):
    """
    Check the quantization matrix for consistency with the flags

    :param U:           quantization matrix
    :param flags:       the flags of the TOAs
    :param uflagvals:   subset of flags that are not ignored

    :return:            True/False, whether or not consistent

    The quantization matrix is checked for three kinds of consistency:
    - Every quantization epoch has more than one observation
    - No quantization epoch has no observations
    - Only one flag is allowed per epoch
    """
    if uflagvals is None:
        uflagvals = list(set(flags))

    rv = True
    collisioncheck = np.zeros((U.shape[1], len(uflagvals)), dtype=np.int)
    for ii, flagval in enumerate(uflagvals):
        flagmask = (flags == flagval)

        Umat = U[flagmask, :]

        simepoch = np.sum(Umat, axis=0)
        if np.all(simepoch <= 1) and not np.all(simepoch == 0):
            rv = False
            #raise ValueError("quantization matrix contains non-jitter-style data")

        collisioncheck[:, ii] = simepoch

        # Check continuity of the columns
        for cc, col in enumerate(Umat.T):
            if np.sum(col > 2):
                # More than one TOA for this flag/epoch
                epinds = np.flatnonzero(col)
                if len(epinds) != epinds[-1] - epinds[0] + 1:
                    rv = False
                    print("WARNING: checkquant found non-continuous blocks")
                    #raise ValueError("quantization matrix epochs not continuous")
        

    epochflags = np.sum(collisioncheck > 0, axis=1)

    if np.any(epochflags > 1):
        rv = False
        print("WARNING: checkquant found multiple backends for an epoch")
        #raise ValueError("Some observing epochs include multiple backends")

    if np.any(epochflags < 1):
        rv = False
        print("WARNING: checkquant found epochs without observations (eflags)")
        #raise ValueError("Some observing epochs include no observations... ???")

    obsum = np.sum(U, axis=0)
    if np.any(obsum < 1):
        rv = False
        print("WARNING: checkquant found epochs without observations (all)")
        #raise ValueError("Some observing epochs include no observations... ???")

    return rv


def quant2ind(U):
    """
    Convert the quantization matrix to an indices matrix for fast use in the
    jitter likelihoods

    :param U:       quantization matrix
    
    :return:        Index (basic slicing) version of the quantization matrix

    This function assumes that the TOAs have been properly sorted according to
    the proper function argsortTOAs above. Checks on the continuity of U are not
    performed
    """
    inds = np.zeros((U.shape[1], 2), dtype=np.int)
    for cc, col in enumerate(U.T):
        epinds = np.flatnonzero(col)
        inds[cc, 0] = epinds[0]
        inds[cc, 1] = epinds[-1]+1

    return inds

def quantreduce(U, eat, flags, calci=False):
    """
    Reduce the quantization matrix by removing the observing epochs that do not
    require any jitter parameters.

    :param U:       quantization matrix
    :param eat:     Epoch-averaged toas
    :param flags:   the flags of the TOAs
    :param calci:   Calculate pseudo-inverse yes/no

    :return     newU, jflags (flags that need jitter)
    """
    uflagvals = list(set(flags))
    incepoch = np.zeros(U.shape[1], dtype=np.bool)
    jflags = []
    for ii, flagval in enumerate(uflagvals):
        flagmask = (flags == flagval)
        
        Umat = U[flagmask, :]
        ecnt = np.sum(Umat, axis=0)
        incepoch = np.logical_or(incepoch, ecnt>1)

        if np.any(ecnt > 1):
            jflags.append(flagval)

    Un = U[:, incepoch]
    eatn = eat[incepoch]

    if calci:
        Ui = ((1.0/np.sum(Un, axis=0)) * Un).T
        rv = (Un, Ui, eatn, jflags)
    else:
        rv = (Un, eatn, jflags)

    return rv



def signalResponse_fast(ptheta_a, pphi_a, gwtheta_a, gwphi_a):
    """
    Create the signal response matrix FAST
    """
    npsrs = len(ptheta_a)

    # Create a meshgrid for both phi and theta directions
    gwphi, pphi = np.meshgrid(gwphi_a, pphi_a)
    gwtheta, ptheta = np.meshgrid(gwtheta_a, ptheta_a)

    return createSignalResponse(pphi, ptheta, gwphi, gwtheta)


def createSignalResponse(pphi, ptheta, gwphi, gwtheta):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    :param pphi:    Phi of the pulsars
    :param ptheta:  Theta of the pulsars
    :param gwphi:   Phi of GW propagation direction
    :param gwtheta: Theta of GW propagation direction

    :return:    Signal response matrix of Earth-term
    """
    Fp = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True)
    Fc = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=False)

    # Pixel maps are lumped together, polarization pixels are neighbours
    F = np.zeros((Fp.shape[0], 2*Fp.shape[1]))
    F[:, 0::2] = Fp
    F[:, 1::2] = Fc

    return F

def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True, norm=False):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    :param pphi:    Phi of the pulsars
    :param ptheta:  Theta of the pulsars
    :param gwphi:   Phi of GW propagation direction
    :param gwtheta: Theta of GW propagation direction
    :param plus:    Whether or not this is the plus-polarization
    :param norm:    Normalise the correlations to equal Jenet et. al (2005)

    :return:    Signal response matrix of Earth-term
    """
    # Create the unit-direction vectors. First dimension will be collapsed later
    # Sign convention of Gair et al. (2014)
    Omega = np.array([-np.sin(gwtheta)*np.cos(gwphi), \
                      -np.sin(gwtheta)*np.sin(gwphi), \
                      -np.cos(gwtheta)])
    
    mhat = np.array([-np.sin(gwphi), np.cos(gwphi), np.zeros(gwphi.shape)])
    nhat = np.array([-np.cos(gwphi)*np.cos(gwtheta), \
                     -np.cos(gwtheta)*np.sin(gwphi), \
                     np.sin(gwtheta)])

    p = np.array([np.cos(pphi)*np.sin(ptheta), \
                  np.sin(pphi)*np.sin(ptheta), \
                  np.cos(ptheta)])
    
    # There is a factor of 3/2 difference between the Hellings & Downs
    # integral, and the one presented in Jenet et al. (2005; also used by Gair
    # et al. 2014). This factor 'normalises' the correlation matrix, but I don't
    # see why I have to pull this out of my ass here. My antennae patterns are
    # correct, so does this mean our strain amplitude is re-scaled. Check this.
    npixels = Omega.shape[2]
    if norm:
        # Add extra factor of 3/2
        c = np.sqrt(1.5) / np.sqrt(npixels)
    else:
        c = 1.0 / np.sqrt(npixels)

    # Calculate the Fplus or Fcross antenna pattern. Definitions as in Gair et
    # al. (2014), with right-handed coordinate system
    if plus:
        # The sum over axis=0 represents an inner-product
        Fsig = 0.5 * c * (np.sum(nhat * p, axis=0)**2 - np.sum(mhat * p, axis=0)**2) / \
                (1 - np.sum(Omega * p, axis=0))
    else:
        # The sum over axis=0 represents an inner-product
        Fsig = c * np.sum(mhat * p, axis=0) * np.sum(nhat * p, axis=0) / \
                (1 - np.sum(Omega * p, axis=0))

    return Fsig



def almFromClm(clm):
    """
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function just
    takes the imaginary part of the abs(m) alm index.
    """
    maxl = int(np.sqrt(len(clm)))-1
    nclm = len(clm)

    nalm = hp.Alm.getsize(maxl)
    alm = np.zeros((nalm), dtype=np.complex128)

    clmindex = 0
    for ll in range(0, maxl+1):
        for mm in range(-ll, ll+1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))
            
            if mm == 0:
                alm[almindex] += clm[clmindex]
            elif mm < 0:
                alm[almindex] -= 1j * clm[clmindex] / np.sqrt(2)
            elif mm > 0:
                alm[almindex] += clm[clmindex] / np.sqrt(2)
            
            clmindex += 1
    
    return alm


def clmFromAlm(alm):
    """
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function just
    takes the imaginary part of the abs(m) alm index.
    """
    nalm = len(alm)
    maxl = int(np.sqrt(9.0 - 4.0 * (2.0-2.0*nalm))*0.5 - 1.5)   # Really?
    nclm = (maxl+1)**2

    # Check the solution. Went wrong one time..
    if nalm != int(0.5 * (maxl+1) * (maxl+2)):
        raise ValueError("Check numerical precision. This should not happen")

    clm = np.zeros(nclm)

    clmindex = 0
    for ll in range(0, maxl+1):
        for mm in range(-ll, ll+1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))
            
            if mm == 0:
                clm[clmindex] = alm[almindex].real
            elif mm < 0:
                clm[clmindex] = - alm[almindex].imag * np.sqrt(2)
            elif mm > 0:
                clm[clmindex] = alm[almindex].real * np.sqrt(2)
            
            clmindex += 1
    
    return clm



def mapFromClm_fast(clm, nside):
    """
    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    :param clm:     Array of C_{lm} values (inc. 0,0 element)
    :param nside:   Nside of the healpix pixelation

    return:     Healpix pixels

    Use Healpix spherical harmonics for computational efficiency
    """
    maxl = int(np.sqrt(len(clm)))-1
    alm = almFromClm(clm)

    h = hp.alm2map(alm, nside, maxl, verbose=False)

    return h

def mapFromClm(clm, nside):
    """
    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    :param clm:     Array of C_{lm} values (inc. 0,0 element)
    :param nside:   Nside of the healpix pixelation

    return:     Healpix pixels

    Use real_sph_harm for the map
    """
    npixels = hp.nside2npix(nside)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    
    h = np.zeros(npixels)

    ind = 0
    maxl = int(np.sqrt(len(clm)))-1
    for ll in range(maxl+1):
        for mm in range(-ll, ll+1):
            h += clm[ind] * real_sph_harm(mm, ll, pixels[1], pixels[0])
            ind += 1

    return h


def clmFromMap_fast(h, lmax):
    """
    Given a pixel map, and a maximum l-value, return the corresponding C_{lm}
    values.

    :param h:       Sky power map
    :param lmax:    Up to which order we'll be expanding

    return: clm values

    Use Healpix spherical harmonics for computational efficiency
    """
    alm = hp.sphtfunc.map2alm(h, lmax=lmax, regression=False)
    alm[0] = np.sum(h) * np.sqrt(4*np.pi) / len(h)  # Why doesn't healpy do this?

    return clmFromAlm(alm)


def clmFromMap(h, lmax):
    """
    Given a pixel map, and a maximum l-value, return the corresponding C_{lm}
    values.

    :param h:       Sky power map
    :param lmax:    Up to which order we'll be expanding

    return: clm values

    Use real_sph_harm for the map
    """
    npixels = len(h)
    nside = hp.npix2nside(npixels)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    
    clm = np.zeros( (lmax+1)**2 )
    
    ind = 0
    for ll in range(lmax+1):
        for mm in range(-ll, ll+1):
            clm[ind] += np.sum(h * real_sph_harm(mm, ll, pixels[1], pixels[0]))
            ind += 1
            
    return clm * 4 * np.pi / npixels



def getCov(sh00, nside, F_e):
    """
    Given a vector of clm values, construct the covariance matrix

    :param sh00:    Healpix map
    :param nside:   Healpix nside resolution
    :param F_e:     Signal response matrix

    :return:    Cross-pulsar correlation for this array of clm values
    """
    # Create a sky-map (power)
    # Use mapFromClm to compare to real_sph_harm. Fast uses Healpix

    # Double the power (one for each polarization)
    sh = np.array([sh00, sh00]).T.flatten()

    # Create the cross-pulsar covariance
    hdcov_F = np.dot(F_e * sh, F_e.T)

    # The pulsar term is added (only diagonals: uncorrelated)
    return hdcov_F + np.diag(np.diag(hdcov_F))

def CorrBasis(psr_locs, nside=32, direction='origin'):
    """
    Calculate the correlation basis matrices using the pixel-space
    transormations

    :param psr_locs:    Location of the pulsars [phi, theta]
    :param ntiles:      Number of tiles to grid the sky with
                        (should be a square number)
    :param nside:       What nside to use in the pixelation [32]

    Note: GW directions are in direction of GW propagation
    """
    npsrs = len(psr_locs)
    pphi = psr_locs[:,0]
    ptheta = psr_locs[:,1]

    print pphi, ptheta

    # Create the pixels
    npixels = hp.nside2npix(nside)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    if direction=='propagation':
        gwtheta = pixels[0]
        gwphi = pixels[1]
    else:
        gwtheta = np.pi-pixels[0]
        gwphi = np.pi+pixels[1]

    # Create the signal response matrix
    F_e = signalResponse_fast(ptheta, pphi, gwtheta, gwphi)
    print F_e.shape
    sh00 = mapFromClm_fast(np.array([1.0]), nside)
    print sh00.shape

    basis = []
    for ii in range(npixels):
        basis.append(getCov(sh00[ii], nside, F_e[:,2*ii:2*ii+2]))

    return basis





def constructPulsarMassFromFile(chain, pars, retSamps=True):
    """
    Construct puslar mass form chain file that uses DD/T2 model
    
    """

    # get values
    m2 = chain[:,list(pars).index('M2')] 
    try:
        cosi = np.cos(np.arcsin(chain[:,list(pars).index('SINI')]))
        sini = chain[:,list(pars).index('SINI')]
    except ValueError:
        cosi = np.cos(chain[:,list(pars).index('KIN')]*np.pi/180)
        sini = np.sin(chain[:,list(pars).index('KIN')]*np.pi/180)

    Pb = chain[:,list(pars).index('PB')]*86400
    X = chain[:,list(pars).index('A1')]*299.79e6/3e8
    M2 = m2*SOLAR2S
    mp = ((sini*(Pb/2/np.pi)**(2./3)*M2/X)**(3./2) - M2)/SOLAR2S

    return mp

def get_edot(F, mc, e):
    """
    Compute eccentricity derivative from Taylor et al. (2015)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param e: Eccentricity of binary

    :returns: de/dt

    """

    # chirp mass
    mc *= SOLAR2S

    dedt = -304/(15*mc) * (2*np.pi*mc*F)**(8/3) * e * \
        (1 + 121/304*e**2) / ((1-e**2)**(5/2))

    return dedt

def get_Fdot(F, mc, e):
    """
    Compute frequency derivative from Taylor et al. (2015)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param e: Eccentricity of binary

    :returns: dF/dt

    """

    # chirp mass
    mc *= SOLAR2S

    dFdt = 48 / (5*np.pi*mc**2) * (2*np.pi*mc*F)**(11/3) * \
        (1 + 73/24*e**2 + 37/96*e**4) / ((1-e**2)**(7/2))

    return dFdt

def get_gammadot(F, mc, q, e):
    """
    Compute gamma dot from Barack and Cutler (2004)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    :param e: Eccentricity of binary

    :returns: dgamma/dt

    """

    # chirp mass
    mc *= SOLAR2S

    #total mass
    m = (((1+q)**2)/q)**(3/5) * mc

    dgdt = 6*np.pi*F * (2*np.pi*F*m)**(2/3) / (1-e**2) * \
        (1 + 0.25*(2*np.pi*F*m)**(2/3)/(1-e**2)*(26-15*e**2))

    return dgdt

def get_coupled_ecc_eqns(y, t, mc, q):
    """
    Computes the coupled system of differential
    equations from Peters (1964) and Barack &
    Cutler (2004). This is a system of three variables:
    
    F: Orbital frequency [Hz]
    e: Orbital eccentricity
    gamma: Angle of precession of periastron [rad]
    phase0: Orbital phase [rad]
    
    :param y: Vector of input parameters [F, e, gamma]
    :param t: Time [s]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    
    :returns: array of derivatives [dF/dt, de/dt, dgamma/dt, dphase/dt]
    """
    
    F = y[0]
    e = y[1]
    gamma = y[2]
    phase = y[3]
    
    #total mass
    m = (((1+q)**2)/q)**(3/5) * mc    
    
    dFdt = get_Fdot(F, mc, e)
    dedt = get_edot(F, mc, e)
    dgdt = get_gammadot(F, mc, q, e)
    dphasedt = 2*np.pi*F
     
    return np.array([dFdt, dedt, dgdt, dphasedt])

def solve_coupled_ecc_solution(F0, e0, gamma0, phase0, mc, q, t):
    """
    Compute the solution to the coupled system of equations
    from from Peters (1964) and Barack & Cutler (2004) at 
    a given time.
    
    :param F0: Initial orbital frequency [Hz]
    :param e0: Initial orbital eccentricity
    :param gamma0: Initial angle of precession of periastron [rad]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    :param t: Time at which to evaluate solution [s]
    
    :returns: (F(t), e(t), gamma(t), phase(t))
    
    """
    
    y0 = np.array([F0, e0, gamma0, phase0])

    y, infodict = odeint(get_coupled_ecc_eqns, y0, t, args=(mc,q), full_output=True)
    
    if infodict['message'] == 'Integration successful.':
        ret = y
    else:
        ret = 0
    
    return ret

def get_an(n, mc, dl, F, e):
    """
    Compute a_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    
    :returns: a_n
    
    """
    
    # convert to seconds
    mc *= SOLAR2S
    dl *= MPC2S
    
    omega = 2 * np.pi * F
    
    amp = n * mc**(5/3) * omega**(2/3) / dl
    
    ret = -amp * (ss.jn(n-2,n*e) - 2*e*ss.jn(n-1,n*e) +
                  (2/n)*ss.jn(n,n*e) + 2*e*ss.jn(n+1,n*e) -
                  ss.jn(n+2,n*e))

    return ret

def get_bn(n, mc, dl, F, e):
    """
    Compute b_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    
    :returns: b_n
    
    """
    
    # convert to seconds
    mc *= SOLAR2S
    dl *= MPC2S 
    
    omega = 2 * np.pi * F
    
    amp = n * mc**(5/3) * omega**(2/3) / dl
        
    ret = -amp * np.sqrt(1-e**2) *(ss.jn(n-2,n*e) - 2*ss.jn(n,n*e) +
                  ss.jn(n+2,n*e)) 

    return ret

def get_cn(n, mc, dl, F, e):
    """
    Compute c_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    
    :returns: c_n
    
    """
    
    # convert to seconds
    mc *= SOLAR2S
    dl *= MPC2S
    
    omega = 2 * np.pi * F
    
    amp = 2 * mc**(5/3) * omega**(2/3) / dl
     
    ret = amp * ss.jn(n,n*e) 

    return ret

def calculate_splus_scross(nmax, mc, dl, F, e, t, l0, gamma, gammadot, inc):
    """
    Calculate splus and scross summed over all harmonics. 
    This waveform differs slightly from that in Taylor et al (2015) 
    in that it includes the time dependence of the advance of periastron.
    
    :param nmax: Total number of harmonics to use
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    :param t: TOAs [s]
    :param l0: Initial eccentric anomoly [rad]
    :param gamma: Angle of periastron advance [rad]
    :param gammadot: Time derivative of angle of periastron advance [rad/s]
    :param inc: Inclination angle [rad]

    """ 
    
    n = np.arange(1, nmax)

    # time dependent amplitudes
    an = get_an(n, mc, dl, F, e)
    bn = get_bn(n, mc, dl, F, e)
    cn = get_cn(n, mc, dl, F, e)

    # time dependent terms
    omega = 2*np.pi*F
    gt = gamma + gammadot * t
    lt = l0 + omega * t

    # tiled phase
    phase1 = n * np.tile(lt, (nmax-1,1)).T
    phase2 = np.tile(gt, (nmax-1,1)).T
    phasep = phase1 + 2*phase2
    phasem = phase1 - 2*phase2

    # intermediate terms
    sp = np.sin(phasem)/(n*omega-2*gammadot) + \
            np.sin(phasep)/(n*omega+2*gammadot)
    sm = np.sin(phasem)/(n*omega-2*gammadot) - \
            np.sin(phasep)/(n*omega+2*gammadot)
    cp = np.cos(phasem)/(n*omega-2*gammadot) + \
            np.cos(phasep)/(n*omega+2*gammadot)
    cm = np.cos(phasem)/(n*omega-2*gammadot) - \
            np.cos(phasep)/(n*omega+2*gammadot)
    

    splus_n = -0.5 * (1+np.cos(inc)**2) * (an*sp - bn*sm) + \
            (1-np.cos(inc)**2)*cn * np.cos(phase1)
    scross_n = np.cos(inc) * (an*cm - bn*cp)
        

    return np.sum(splus_n, axis=1), np.sum(scross_n, axis=1)

def compute_eccentric_residuals(psr, gwtheta, gwphi, mc,
                                dist, F, inc, psi,
                                gamma0, e0, l0, q, nmax=400,
                                pdist=None, pphase=None,
                                pgam=None, psrTerm=True,
                                tref=0, check=False):

    """
    Simulate GW from eccentric SMBHB. Waveform models from
    Taylor et al. (2015) and Barack and Cutler (2004).

    WARNING: This residual waveform is only accurate if the
    GW frequency is not significantly evolving over the 
    observation time of the pulsar.

    :param psr: pulsar object
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param mc: Chirp mass of SMBMB [solar masses]
    :param dist: Luminosity distance to SMBMB [Mpc]
    :param F: Orbital frequency of SMBHB [Hz]
    :param inc: Inclination of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param gamma0: Initial angle of periastron [radians]
    :param e0: Initial eccentricity of SMBHB
    :param l0: Initial mean anomoly [radians]
    :param q: Mass ratio of SMBHB
    :param nmax: Number of harmonics to use in waveform decomposition
    :param pdist: Pulsar distance [kpc]
    :param pphase: Pulsar phase [rad]
    :param pgam: Pulsar angle of periastron [rad]
    :param psrTerm: Option to include pulsar term [boolean] 
    :param tref: Fidicuial time at which initial parameters are referenced [s]
    :param check: Check if frequency evolves significantly over obs. time

    :returns: Vector of induced residuals
    """
    
    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])
    
    res = []
    for ct, p in enumerate(psr):
        
        # use definition from Sesana et al 2010 and Ellis et al 2012
        phat = np.array([np.sin(p.theta)*np.cos(p.phi), np.sin(p.theta)*np.sin(p.phi),\
                np.cos(p.theta)])

        fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
        fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
        cosMu = -np.dot(omhat, phat)
        
        # get values from pulsar object
        toas = p.toas - tref
        
        if pdist is None:
            pd = p.pdist
        else:
            pd = pdist[ct]   

        # convert units
        pd *= KPC2S   
        
        # get pulsar time
        tp = toas - pd * (1-cosMu)

        if check:
            # check that frequency is not evolving significantly over obs. time
            y = solve_coupled_ecc_solution(F, e0, gamma0, l0, mc, q,
                                              np.array([0.0,toas.max()]))
            
            # initial and final values over observation time
            Fc0, ec0, gc0, phic0 = y[0,:]
            Fc1, ec1, gc1, phic1 = y[-1,:]

            # observation time
            Tobs = 1/(toas.max()-toas.min())

            if np.abs(Fc0-Fc1) > 1/Tobs:
                print('WARNING: Frequency is evolving over more than one frequency bin.')
                print('F0 = {0}, F1 = {1}, delta f = {2}'.format(Fc0, Fc1, 1/Tobs))
                return np.ones(len(p.toas)) * np.nan


        
        # get gammadot for earth term
        gammadot = get_gammadot(F, mc, q, e0)

        # get number of harmonics to use
        if not isinstance(nmax, int):
            if e0 < 0.999 and e0 > 0.001:
                nharm = int(nmax(e0))
            elif e0 < 0.001:
                nharm = 2
            else:
                nharm = int(nmax(0.999))
        else:
            nharm = nmax
        
        # no more than 100 harmonics
        nharm = min(nharm, 100)
        
        ##### earth term #####
        splus, scross = calculate_splus_scross(nharm, mc, dist, F, e0, toas,
                                               l0, gamma0, gammadot, inc)
        
        ##### pulsar term #####
        if psrTerm:
            # solve coupled system of equations to get pulsar term values
            y = solve_coupled_ecc_solution(F, e0, gamma0, l0, mc,
                                           q, np.array([0.0, tp.min()]))
            
            # get pulsar term values
            if np.any(y):
                Fp, ep, gp, phip = y[-1,:]
                
                # get gammadot at pulsar term
                gammadotp = get_gammadot(Fp, mc, q, ep)

                # get phase at pulsar
                if pphase is None:
                    lp = phip 
                else:
                    lp = pphase[ct] 
                
                # get angle of periastron at pulsar
                if pgam is None:
                    gp = gp
                else:
                    gp = pgam[ct] 

                # get number of harmonics to use
                if not isinstance(nmax, int):
                    if e0 < 0.999 and e0 > 0.001:
                        nharm = int(nmax(e0))
                    elif e0 < 0.001:
                        nharm = 2
                    else:
                        nharm = int(nmax(0.999))
                else:
                    nharm = nmax
        
                # no more than 1000 harmonics
                nharm = min(nharm, 100)
                splusp, scrossp = calculate_splus_scross(nharm, mc, dist, Fp,
                                                         ep, toas, lp, gp, 
                                                         gammadotp, inc)

                rr = (fplus*cos2psi - fcross*sin2psi) * (splusp - splus) + \
                    (fplus*sin2psi + fcross*cos2psi) * (scrossp - scross)

            else:
                rr = np.ones(len(p.toas)) * np.nan
                
        else:
            rr = - (fplus*cos2psi - fcross*sin2psi) * splus - \
                (fplus*sin2psi + fcross*cos2psi) * scross
                
        res.append(rr)

    return res

def binresults(x, y, yerr, nbins=20):

    xedges = np.linspace(x.min(), x.max(), nbins+1)
    
    xx = 0.5*(xedges[1:] + xedges[:-1])
    newx = []
    newy = []
    newyerr = []

    for ll, ledge in enumerate(xedges[:-1]):
        ind = np.logical_and(x >= ledge, x < xedges[ll+1])
        if np.sum(ind) > 0:
            newy.append(np.average(y[ind], weights=1.0/yerr[ind]**2, ))
            newyerr.append(1.0 / np.sqrt(np.sum(1.0/yerr[ind]**2)))
            newx.append(xx[ll])
    
    return np.array(newx), np.array(newy), np.array(newyerr)


def make_linear_interp_basis(t, npts=80):

    x = np.linspace(t.min(), t.max(), npts)

    M = np.zeros((len(t), len(x)))

    for ii in range(len(x)-1):
        idx = np.logical_and(t>=x[ii], t<=x[ii+1])
        M[idx, ii] = (t[idx] - x[ii+1]) / (x[ii] - x[ii+1])
        M[idx, ii+1] = (t[idx] - x[ii]) / (x[ii+1] - x[ii])

    return M
