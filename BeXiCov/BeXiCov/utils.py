''' module to compute no-wiggle and de-wiggle Powserspectrum
    author: A.Veropalumbo
    '''
    
import numpy as np
from scipy.special import gamma
from scipy.fft import rfft


class PkFFTlog:
    def __init__(self, bias=-1.6, n_max=256, k_min_0=1.e-5, k_max_0=100):
        self.delta = None
        self.k_n = None
        self.k_n_bias = None
        self.eta_m = None
        self.nu_m = None

        self.n_max = None
        self.bias = None
        self.k_min_0 = None
        self.k_max_0 = None
        self.k_nu_matrix = None

        self.set_parameters(bias=bias, n_max=n_max, k_min_0=k_min_0, k_max_0=k_max_0)

    def set_parameters(self, bias=-1.6, n_max=256, k_min_0=1.e-5, k_max_0=100):
        self.n_max = n_max
        self.bias = bias
        self.k_min_0 = k_min_0
        self.k_max_0 = k_max_0

        self.delta = 1 / (self.n_max - 1) * np.log(self.k_max_0 / self.k_min_0)
        self.k_n = np.array(
            [self.k_min_0 * np.exp(i * self.delta) for i in range(self.n_max)])
        self.eta_m = np.array([
            2 * np.pi / (self.n_max * self.delta) *
            (j - self.n_max / 2) for j in range(0, self.n_max + 1)
        ])
        self.nu_m = -0.5 * (self.bias + 1j * self.eta_m)
        self.k_n_bias = np.array(
            [np.exp(-self.bias * i * self.delta) for i in range(self.n_max)])
        X, Y = np.meshgrid(self.k_n, -2 * self.nu_m)
        self.k_nu_matrix = X ** Y

    def get_c_m(self, p_k):
        pk_bias = p_k * self.k_n_bias

        cm = rfft(pk_bias) / self.n_max
        c_m = 1j * np.zeros(self.n_max + 1)
        c_m[int(self.n_max / 2):] = cm
        c_m[0:int(self.n_max / 2)] = cm[-1:0:-1].conj()

        return c_m * self.k_min_0 ** (2 * self.nu_m)

def get_II(nu1, nu2):
    return 1. / (8 * np.pi ** 1.5) * gamma(1.5 - nu1) * gamma(
           1.5 - nu2) * gamma(nu1 + nu2 - 1.5) / (gamma(nu1) * gamma(nu2) *
           gamma(3 - nu1 - nu2))

class PkSPT(PkFFTlog):

    def __init__(self, bias=-1.6, n_max=256, k_min_0=1.e-5, k_max_0=100):
        super().__init__(bias=bias, n_max=n_max, k_min_0=k_min_0, k_max_0=k_max_0)

        self.M22 = None
        self.M13 = None

    def _get_M13(self, nu1):
        return (1 + 9 * nu1) / 4 * np.tan(nu1 * np.pi) / (28 * np.pi *
                                                          (nu1 + 1) * nu1 *
                                                          (nu1 - 1) *
                                                          (nu1 - 2) *
                                                          (nu1 - 3))

    def _get_M22(self, nu1, nu2):
        nu12 = nu1 + nu2
        den = 196 * nu1 * (1 + nu1) * (0.5 - nu1) * nu2 * (1 + nu2) * (0.5 -
                                                                       nu2)
        num = nu1 * nu2 * (98 * nu12 ** 2 - 14 * nu12 +
                           36) - 91 * nu12 ** 2 + 3 * nu12 + 58
        num = (1.5 - nu12) * (0.5 - nu12) * num * self.II
        return num / den

    def set_tables(self):
        X, Y = np.meshgrid(self.nu_m, self.nu_m)
        self.II = get_II(X, Y)
        self.M22 = self._get_M22(X, Y)
        self.M13 = self._get_M13(X)

    def __call__(self, interp_pk):
        p_k = interp_pk(self.k_n)
        c_m = self.get_c_m(p_k)

        # p_k_13 = self.k_n**3 * p_k * np.dot(c_m * self.M13, self.k_nu_matrix).real
        p_k_13 = self.k_n ** 3 * np.array([
            np.dot(c_m * line, np.dot(self.M13, c_m * line)).real
            for line in self.k_nu_matrix.T
        ])

        p_k_22 = self.k_n ** 3 * np.array([
            np.dot(c_m * line, np.dot(self.M22, c_m * line)).real
            for line in self.k_nu_matrix.T
        ])

        return self.k_n, p_k, p_k_22, p_k_13

class PkBias(PkSPT):
    def __init__(self, bias=-1.6, n_max=256, k_min_0=1.e-5, k_max_0=100):
        super().__init__(bias=bias, n_max=n_max, k_min_0=k_min_0, k_max_0=k_max_0)

        self.Md2 = None
        self.Mg2 = None
        self.MFg2 = None
        self.Md2d2 = None
        self.Mg2g2 = None
        self.Md2g2 = None
        self.kernels = ["d2", "g2", "d2d2", "g2g2", "d2g2"]

    def _get_Md2(self, nu1, nu2):
        nu12 = nu1 + nu2

        return (3 - 2 * nu12) * (4 - 7 * nu12) / (14 * nu1 * nu2) * self.II

    def _get_Mg2(self, nu1, nu2):
        nu12 = nu1 + nu2

        return -(3 - 2 * nu12) * (1 - 2 * nu12) * (6 + 7 * nu12) / (28 * nu1 * (1 + nu1) * nu2 * (1 + nu2)) * self.II

    def _get_MFg2(self, nu1):
        return -15 * np.tan(nu1 * np.pi) / (28 * np.pi * (nu1 + 1) * nu1 *
                                            (nu1 - 1) * (nu1 - 2) * (nu1 - 3))

    def _get_Md2d2(self, nu1, nu2):
        return 2 * self.II

    def _get_Mg2g2(self, nu1, nu2):
        nu12 = nu1 + nu2

        return (3 - 2 * nu12) * (1 - 2 * nu12) / ((nu1) * (1 + nu1) * nu2 *
                                                  (1 + nu2)) * self.II

    def _get_Md2g2(self, nu1, nu2):
        nu12 = nu1 + nu2
        return (3 - 2 * nu12) / (nu1 * nu2) * self.II

    def set_tables(self):
        X, Y = np.meshgrid(self.nu_m, self.nu_m)
        self.II = get_II(X, Y)

        self.MFg2 = self._get_MFg2(X)  # -0.5 * self.eta_m)

        for kn in self.kernels:
            setattr(self, "M%s" % kn, getattr(self, "_get_M%s" % kn)(X, Y))

    def __call__(self, interp_pk):
        p_k = interp_pk(self.k_n)
        c_m = self.get_c_m(p_k)

        pk_d2 = self.k_n ** 3 * np.array([np.dot(c_m * line,
                                                 np.dot(self.Md2, c_m * line)).real \
                                          for line in self.k_nu_matrix.T])

        pk_g2 = self.k_n ** 3 * np.array([np.dot(c_m * line,
                                                 np.dot(self.Mg2, c_m * line)).real \
                                          for line in self.k_nu_matrix.T])

        pk_22 = self.k_n ** 3 * np.array([np.dot(c_m * line,
                                                 np.dot(self.Md2d2, c_m * line)).real \
                                          for line in self.k_nu_matrix.T])

        pk_d2g2 = self.k_n ** 3 * np.array([np.dot(c_m * line,
                                                   np.dot(self.Md2g2, c_m * line)).real \
                                            for line in self.k_nu_matrix.T])

        pk_g2g2 = self.k_n ** 3 * np.array([np.dot(c_m * line,
                                                   np.dot(self.Mg2g2, c_m * line)).real \
                                            for line in self.k_nu_matrix.T])

        pk_Fg2 = self.k_n ** 3 * np.array([np.dot(c_m * line, np.dot(self.MFg2, c_m * line)).real \
                                           for line in self.k_nu_matrix.T])

        # p_k_Fg2 = self.k_n**3 * p_k * np.dot(c_m * self.MFg2, self.k_nu_matrix).real

        return pk_d2, pk_g2, pk_Fg2, pk_22 - pk_22[0], pk_g2g2, pk_d2g2


class PkToXi(PkSPT):
    def __init__(self, bias=-1.6, n_max=256, k_min_0=1.e-5, k_max_0=100):
        super().__init__(bias=bias, n_max=n_max, k_min_0=k_min_0, k_max_0=k_max_0)

        self.omega_m = None
        self.Mtilde_11 = None

    def set_tables(self):
        super().set_tables()
        self.omega_m = self.nu_m - 1.5

        X, Y = np.meshgrid(self.nu_m, self.nu_m)
        XY = X + Y

        self.Mtilde_11 = 1. / (2 * np.pi ** 2) * gamma(2 - 2 * self.nu_m) * \
                         np.sin(np.pi * self.nu_m)

    def __call__(self, rad, interp_pk):
        p_k = interp_pk(self.k_n)
        c_m = self.get_c_m(p_k)

        X, Y = np.meshgrid(rad, 2 * self.omega_m)
        romega = X ** Y
        xi = np.dot(c_m * self.Mtilde_11, romega).real

        return xi

import camb
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn as jn


def compute_sigmaR(pk_func, kmin=1.e-4, kmax=1.e1, r=8):
    def integrand(log_k):
        k = np.exp(log_k)
        kr = k * r
        wf = 3 * (np.sin(kr) - kr * np.cos(kr)) / kr ** 3
        return k ** 3 * wf ** 2 * pk_func(k)

    return np.sqrt(1. / (2 * np.pi ** 2) * quad(integrand, np.log(kmin), np.log(kmax), epsrel=1.e-4, epsabs=0)[0])


def compute_sigma_v_squared(pk_func, kmin=1.e-4, kmax=0.2, l_bao=110):
    def integrand(log_q):
        q = np.exp(log_q)
        return q * pk_func(q) * (1 - jn(0, q * l_bao) + 2 * jn(2, q * l_bao))

    return 1. / (6 * np.pi ** 2) * quad(integrand, np.log(kmin), np.log(kmax), epsrel=1.e-4, epsabs=0)[0]


def get_R(k, pk_lin, pk_approx):
    return pk_lin(k) / pk_approx(k)


def get_filter(k, pk_lin, pk_approx, lambda_scale=0.25):
    log_k = np.log10(k)
    norm = 2 * lambda_scale ** 2

    def integrand(log_q):
        return get_R(10 ** log_q, pk_lin, pk_approx) * np.exp(-(log_k - log_q) ** 2 / norm)

    return 1. / (np.sqrt(2 * np.pi) * lambda_scale) * \
           quad(integrand, log_k - 4 * lambda_scale, log_k + 4 * lambda_scale, epsrel=1.e-4, epsabs=0)[0]


class EisensteinHu:

    def __init__(self):
        self.omega_m = None
        self.omega_b = None
        self.T_CMB = None
        self.h = None
        self.H0 = None
        self.ns = None

        self.theta = None
        self.k_equality = None
        self.alpha_gamma = None
        self.s = None
        self.sigma8 = None
        self.normalization = 1

    def _compute_internal(self, sigma8, kmin=1.e-4, kmax=10):
        self.theta = (self.T_CMB / 2.7)

        self.k_equality = 0.0746 * self.omega_m * self.h ** 2 / self.theta ** 2 / self.h

        self.alpha_gamma = 1 - 0.328 * np.log(431 * self.omega_m * self.h ** 2) * (self.omega_b / self.omega_m) + \
                           0.38 * np.log(22.3 * self.omega_m * self.h ** 2) * (self.omega_b / self.omega_m) ** 2
        self.gamma = self.omega_m * self.h
        self.s = 44.5 * np.log(9.83 / (self.omega_m * self.h ** 2)) / \
                 np.sqrt(1 + 10 * (self.omega_b * self.h ** 2) ** (3. / 4)) * self.h

        self._compute_normalisation(sigma8)

    def _compute_normalisation(self, sigma8, kmin=1.e-4, kmax=10.):
        self.sigma8 = sigma8
        self.normalization = self.sigma8 ** 2 / compute_sigmaR(self.power_spectrum, kmin, kmax, r=8) ** 2

    def set_params(self, omega_m, omega_b, T_CMB, h, ns, sigma8, kmin, kmax):
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.T_CMB = T_CMB
        self.h = h
        self.ns = ns

        self._compute_internal(sigma8, kmin, kmax)

    def set_params_from_camb_ini(self, param_file, sigma8, kmin=1.e-4, kmax=10):
        params = camb.read_ini(param_file)

        self.set_params(params.omegam,
                        params.omegab,
                        params.TCMB,
                        params.h,
                        params.InitPower.ns,
                        sigma8,
                        kmin,
                        kmax)

    def transfer_function(self, kh):
        gamma_eff = self.omega_m * self.h ** 2 * (
                    self.alpha_gamma + (1 - self.alpha_gamma) / (1 + (0.43 * kh * self.s) ** 4))
        q = kh / 13.41 / self.k_equality * self.omega_m * self.h ** 2 / gamma_eff

        L0 = np.log(2 * np.e + 1.8 * q)
        C0 = 14.2 + 731. / (1 + 62.5 * q)
        return L0 / (L0 + C0 * q ** 2)

    def power_spectrum(self, kh):
        return self.normalization * self.transfer_function(kh) ** 2 * kh ** self.ns

