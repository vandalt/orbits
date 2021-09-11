from functools import partial
from typing import Union

import aesara_theano_fallback.tensor as tt
import pymc3 as pm
from celerite2.theano import terms
from pymc3.gp.cov import Covariance


def get_qp_kernel(
    input_dim: int,
    period: Union[tt.TensorVariable, float],
    ls_per: Union[tt.TensorVariable, float],
    ls_exp: Union[tt.TensorVariable, float],
) -> Covariance:
    """
    Helper function to create a quasi-periodic covariance function in PyMC3.

    :param period: Period of the periodic component
    :type period: Union[tt.TensorVariable, float]
    :param ls_per: Length scale of the periodic component
    :type ls_per: Union[tt.TensorVariable, float]
    :param ls_exp: Length scale of the squared exponential component
    :type ls_exp: Union[tt.TensorVariable, float]
    :return: Quasi-periodic convariance function
    :rtype: Covariance
    """

    sqexp_kernel = pm.gp.cov.ExpQuad(input_dim, ls=ls_exp)
    per_kernel = pm.gp.cov.Periodic(input_dim, period, ls=ls_per)

    return sqexp_kernel * per_kernel


def get_sum_matern(sigma1, sigma2, rho1, rho2):

    ker1 = terms.Matern32Term(sigma=sigma1, rho=rho1)
    ker2 = terms.Matern32Term(sigma=sigma2, rho=rho2)

    return ker1 + ker2


def get_pm3_kernel(
    kernel_name: str, sigma: Union[tt.TensorVariable, float], **kwargs
) -> Covariance:
    """
    Function that creates a PyMC3 kernel by adding an amplitude to the built-in
    covraiance functions.

    All kwargs are passed to the PyMC3 covariance function directy.

    NOTE: This does not handle celerite2 terms/kernels. As they usually
    include sigma and are built for 1d timeseries.

    :param kernel_name: Kernel name (name of the class, but as a string)
    :type kernel_name: str
    :param sigma: Amplitude of the GP (pymc3 cov is multiplied by sigma^2).
                  NOTE: Not 100% sure about the type below.
    :type sigma: Union[tt.TensorVariable, float]
    :return: PyMC3 covariance function that can be used in regular pymc3 gp.
    :rtype: Covariance
    """

    # Always use 1 as input dimention because we deal with timeseries
    return sigma ** 2 * PYMC3_COVS[kernel_name](1, **kwargs)


CELERITE_KERNELS = {
    "SHOTerm": terms.SHOTerm,
    "Matern32Term": terms.Matern32Term,
    "SumMatern32Term": get_sum_matern,
    "RotationTerm": terms.RotationTerm,
}
PYMC3_COVS = {
    "ExpQuad": pm.gp.cov.ExpQuad,
    "Matern32": pm.gp.cov.Matern32,
    "Matern52": pm.gp.cov.Matern52,
    "Periodic": pm.gp.cov.Periodic,
    "QuasiPeriodic": get_qp_kernel,
}
PYMC3_KERNELS = {kname: partial(get_pm3_kernel, kname) for kname in PYMC3_COVS}
KERNELS = {**CELERITE_KERNELS, **PYMC3_KERNELS}
KERNEL_LIST = list(KERNELS.keys())
KERNEL_PARAMS = {
    "SHOTerm": ["sigma", "rho", "Q"],
    "Matern32Term": ["sigma", "rho"],
    "SumMatern32Term": ["sigma1", "sigma2", "rho1", "rho2"],
    "RotationTerm": ["sigma", "period", "Q0", "dQ", "f"],
    "ExpQuad": ["sigma", "ls"],
    "Matern32": ["sigma", "ls"],
    "Matern52": ["sigma", "ls"],
    "Periodic": ["sigma", "period", "ls"],
    "QuasiPeriodic": ["sigma", "period", "ls_per", "ls_exp"],
}
