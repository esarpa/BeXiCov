'''
Module to compute the expected clustering signal of observed data.

Functions are divided into 2 categories:
    - Powerspectrum templates
    - 2PCF templates
'''

# Import needed libraries
from scipy.interpolate import interp1d
from scipy.integrate import quad
import hankl
import camb
from camb import model, initialpower
import numpy as np

def DefaultCosmology():
    '''
    Function returning a list of fiducial Euclidean cosmological parameters.

    Returns:
        Cosmo (dict): Dictionary of cosmological parameters
    '''
    Cosmo = {
        'Omega_m': 0.319,
        'Omega_b': 0.049,
        'Omega_r': 0.,
        'Omega_L': 1.,
        'Omega_nu': 0.,
        'h': 0.67,
        'As': 2.11065e-9,
        'scalar_pivot': 0.05,
        'ns': 0.97,
        'wa': 0.,
        'w0': -1.,
        'z': 0,
        'bias': np.array([1])
    }
    Cosmo['Omega_L'] = 1 - Cosmo['Omega_m']
    return Cosmo


# Powerspectrum templates

def Pk_linear(cosmo):
    '''
    Compute the isotropic linear powerspectrum.

    Args:
        cosmo (dict): List of cosmological parameters, example in DefaultCosmology()

    Returns:
        kh (array): K modes
        pk_linear (array): Linear powerspectrum evaluated in kh
        f (float): Growth rate of structures
        sigma8 (float): Amplitude of perturbations
        T_cmb (float): Temperature of CMB
    '''
    # ... (code implementation here)
    return kh, pk_linear, f, sigma8, T_cmb


def Pk_nowiggle(k, Pk_l, cosmo, sigma8, T_cmb):
    '''
    Routine to compute Pk_nowiggle as in Vlah et al 2016.

    Args:
        k (array): Separation vector in units of (h/Mpc)
        Pk_l (array): Linear Pk computed at k by Pk_linear(cosmo)
        cosmo (dict): List of cosmological parameters as produced by Recon_challenge.GetData import EuclidCosmology
        sigma8 (float): Sigma8 of the linear Pk
        T_cmb (float): Temperature of CMB

    Returns:
        pk_nw (array): Pk_nowiggle evaluated at k
    '''
    # ... (code implementation here)
    return pk_nw


def P_k_mu_models(cosmo, space, rectype=''):
    '''
    Compute the anisotropic power spectrum for the desired clustering model.

    Args:
        cosmo (dict): List of cosmological parameters, example in DefaultCosmology()
        space (str): 'RealSpace' if the template has to be computed in real-space, 'RedshiftSpace' if in redshift-space
        rectype (str): String of reconstruction type. Leave empty for no reconstruction, 'rec-sym', \
                       'rec-iso' for the Zel'dovich reconstruction without and with RSD removal

    Returns:
        kh (array): Array at which to evaluate Pk_linear
        P_k_mu (function): Function describing the anisotropic Powerspectrum
    '''
    # ... (code implementation here)
    return kh, P_k_mu
    
    
#-------- P_mu_k_parametric ------
def P_mu_k_parametric(par, kh, Pk_l, Pk_now):
    '''
    Construct a continuous function describing the anisotropic P(mu,k) given fitting parameters.

    Args:
        par (dict): List of fitting parameters
        kh (array): Separation vector in units of (h/Mpc)
        Pk_l (array): Linear Pk computed at k
        Pk_now (array): No-wiggle power spectrum

    Returns:
        P_mu_k (function): Function describing the anisotropic power spectrum given the fiducial parameters
    '''
    
    interp_Pk_l = interp1d(kh, Pk_l)
    interp_Pk_now = interp1d(kh, Pk_now)
    
    beta = par['f'] / par['bias']
    
    # Setting up reconstruction template
    if par['Sigma_rec'] != 0:
        Sk_array = np.exp(-kh**2 * par['Sigma_rec']**2 / 2.)  # Smoothing filter
    else:
        Sk_array = np.zeros(len(kh))
    Sk = interp1d(kh, Sk_array)
    
    sigma_v2 = lambda mu: (1. - mu**2) * par['Sigma_perp']**2 / 2. + mu**2 * par['Sigma_par']**2 / 2.  # Anisotropic dumping
    Kaiser = lambda mu, k: (1. + mu**2 * beta * (1. - Sk(k)))**2  # Kaiser distortion
    FoG = lambda mu, k: 1. / (1. + k**2 * mu**2 * par['Sigma_s']**2 / 2.)**2  # Fingers of God
    
    P_mu_k = lambda mu, k: par['bias']**2 * Kaiser(mu, k) * FoG(mu, k) * (
            (interp_Pk_l(k) - interp_Pk_now(k)) * np.exp(-k**2 * sigma_v2(mu)) + interp_Pk_now(k))
    
    return P_mu_k

#------ ZAPkDamping -----
def ZAPkDamping(kh,Pk_l,Rs):
    '''
        damping factor to model post-ZAreconstruction powerspectrum, Padmanabhan White choen 2009 eq 32
        kh: separation vectorn in k-space
        Pk_l: linear Pk comlutded at k
        Rs: size of gaussian smoothing kernel
        returns: post reconstruction damping array
    '''
    
    interp_pk=interp1d(kh,Pk_l)
    Sk_array=np.exp(-kh**2*Rs**2/2.) #smoothing filter eq.33
    Sk=interp1d(kh,Sk_array)
    #computing sigmas
    sigma_ss_2 = quad( lambda x : interp_pk(np.exp(x))*Sk(np.exp(x))**2 * np.exp(x), np.log(1.e-4), np.log(100))[0] / (3. * np.pi**2)
    sigma_dd_2 = quad( lambda x : interp_pk(np.exp(x))*(1.-Sk(np.exp(x)))**2 * np.exp(x), np.log(1.e-4), np.log(100))[0] / (3. * np.pi**2)
    sigma_sd_2=0.5*(sigma_ss_2+sigma_dd_2)
    
    Dk= Sk(kh)**2*np.exp(-kh**2*sigma_ss_2/2.)+ (1.-Sk(kh))**2*np.exp(-kh**2*sigma_dd_2/2.)+\
        2.*Sk(kh)*(1.-Sk(kh))*np.exp(-kh**2*sigma_sd_2/2.)
        
    return Dk
 
#------ NLPkDamping -----
def NLPkDamping(kh,Pk_l):
    '''
        damping factor to model observed powerspectrum, Padmanabhan White choen 2009
        kh: separation vectorn in k-space
        Pk_l: linear Pk comlutded at k
        returns: non-linear damping array
    '''
    
    interp_pk=interp1d(kh,Pk_l)
    #computing sigmas
    sigma_nl_2 = quad( lambda x : interp_pk(np.exp(x))* np.exp(x), np.log(1.e-4), np.log(100))[0] / (3. * np.pi**2)
   
    ExpSigma= lambda sigma2: np.exp(-kh**2*sigma2/2.)
    
    Dk=ExpSigma(sigma_nl_2)

    
    return Dk

#======== 2PCF =======

#-------- ximur ------
def ximur(kh, Pmuk_dewiggle):
    '''
    Construct the damped anisotropic 2PCF xi(mu, r).

    Args:
        kh (array): Separation array on which Pkmu_dewiggle was interpolated
        Pmuk_dewiggle (function): Damped anisotropic Powerspectrum: P(mu, k)

    Returns:
        rtemp (array): Base separation vector corresponding to kh, a 2D function xi(mu, r)
        xi_mu_r (function): Analytical function describing the damped and redshifted correlation function xi(mu, r)
    '''
    elles = np.array([0, 2, 4])  # Multiples

    # Sample P_mu_k
    mus = np.linspace(-1, 1, 2000)
    P_mu_k = Pmuk_dewiggle(mus[:, None], kh[None, :])  # [mu][k]

    # Get Pk multiples
    Pk_elle = np.zeros((len(elles), len(kh)))
    for i in range(0, len(elles)):
        Ll = legendre(elles[i])(mus)
        dx = mus[1] - mus[0]
        Pk_elle[i] = (2. * elles[i] + 1.) / 2 * np.trapz(P_mu_k * Ll[:, None], x=mus, dx=dx, axis=0)

    # Get xi multiples
    rtemp, xi_0 = hankl.P2xi(kh, Pk_elle[0], l=0, lowring=True)
    xi_0 = interp1d(rtemp, xi_0)
    rtemp, xi_2 = hankl.P2xi(kh, Pk_elle[1], l=2, lowring=True)
    xi_2 = interp1d(rtemp, xi_2)
    rtemp, xi_4 = hankl.P2xi(kh, Pk_elle[2], l=4, lowring=True)
    xi_4 = interp1d(rtemp, xi_4)

    # Resum the multipoles
    xi_mu_r = lambda mu, r: xi_0(r) * legendre(0)(mu) + xi_2(r) * legendre(2)(mu) + xi_4(r) * legendre(4)(mu)

    return rtemp, xi_mu_r
    
#-------- xiell ------
def xiell(r, par, xi_mu_r):
    '''
    Generate the 2PCF multiples given the fitting parameters.

    Args:
        r (array): Separation vector in Mpc/h at which to compute xi(r, mu)
        par (dict): List of fitting parameters
        xi_mu_r (function): Analytical function describing the damped and redshifted correlation function

    Returns:
        xi_elle (array): 2PCF multiples evaluated in r. Shape: (3, len(r))
    '''
    # Adding AP shift
    muprime = lambda mu: mu * par['alpha_par'] * (mu**2 * par['alpha_par']**2 + (1. - mu**2) * par['alpha_perp']**2)**(-0.5)
    rprime = lambda mu, r: r * (mu**2 * par['alpha_par']**2 + (1. - mu**2) * par['alpha_perp']**2)**(0.5)
    
    xi_muprime_rprime = lambda mu, r: xi_mu_r(muprime(mu), rprime(mu, r))
    
    # Computing multipoles
    elles = np.array([0, 2, 4])  # Multiples
    xi_elle = np.zeros((len(elles), len(r)))
    
    # Sample xi_muprime_rprime
    mus = np.linspace(-1, 1, 2000)
    xi_mup_rp = xi_muprime_rprime(mus[:, None], r[None, :])  # [mu][r]
    
    # Get xi multiples
    for i in range(0, len(elles)):
        Ll = legendre(elles[i])(mus)
        xi_elle[i] = (2. * elles[i] + 1.) / 2. * np.trapz(xi_mup_rp * Ll[:, None], x=mus, axis=0, dx=mus[1] - mus[0])
    
    return xi_elle


#-------- model_xi_ell ------
def model_xi_ell(r, par, kh, Pk_l, Pk_now):
    '''
    Model the multiples of the 2PCF.

    Args:
        r (array): Separation vector in Mpc/h at which to compute xi(r, mu)
        par (dict): List of fitting parameters
        kh (array): Separation vector in units of (h/Mpc)
        Pk_l (array): Linear Pk computed at k
        Pk_now (array): No-wiggle power spectrum

    Returns:
        xi_l_ref (array): xi multiples evaluated in r. Shape: (3, len(r))
    '''
    # Construct anisotropic power spectrum
    Pmu_k_dew = P_mu_k_parametric(par, kh, Pk_l, Pk_now)(par, kh, Pk_l, Pk_now)
    # Construct anisotropic 2PCF
    rtemp, xi_mu_r = ximur(kh, Pmu_k_dew)
    # Get multipoles
    xi_l_ref = xiell(r, par, xi_mu_r)
    
    return xi_l_ref


#-------- BroadBand ------
def BroadBand(r, bbpar):
    '''
    Generate the polynomial broad band term given the fiducial parameters bbpar.

    Args:
        r (array): Separation vector in Mpc/h at which to compute xi(r, mu)
        bbpar (dict): List of broad band parameters

    Returns:
        BB (array): Broad band multiples evaluated in r. Shape: (3, len(r))
    '''
    elles = np.array([0, 2, 4])
    BB = np.zeros((len(elles), len(r)))
    norm = 0.0015  # Norm representing the value of xi at r=rref
    rref = 80.
    for l in range(0, len(elles)):
        for i in range(-2, 3):
            bli = bbpar['b' + str(elles[l]) + str(i)]
            BB[l] += bli * r**(-i) * norm * rref**(i)
    return BB
  

#-------- legendre ------
#return legendre polinomial of order elle
#elle: order of polynomial in (2,4,6)
#x position
#returns continuous function
def legendre(elle):
    
    L={
        '0': lambda x: x-x+1.,
        '2': lambda x: 1./2.*(3*x**2-1),
        '4': lambda x: 1./8.*(35*x**4-30*x**2+3)
    }
    return L[str(elle)]
