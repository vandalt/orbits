from typing import Union

import pymc3 as pm
from celerite2.theano import terms
from pymc3.gp.cov import Covariance


def get_qp_kernel(period, ls_exp, ls_per) -> Covariance:

    sqexp_kernel = pm.gp.cov.ExpQuad(1, ls=ls_exp)
    per_kernel = pm.gp.cov.Periodic(1, period, ls=ls_per)

    return sqexp_kernel * per_kernel


CELERITE_KERNELS = {"SHOTerm": terms.SHOTerm}
PYMC3_KERNELS = {
    "ExpQuad": pm.gp.cov.ExpQuad,
    "Matern32": pm.gp.cov.Matern32,
    "Matern52": pm.gp.cov.Matern52,
    "Periodic": pm.gp.cov.Periodic,
    "QuasiPeriodic": get_qp_kernel,
}
KERNELS = {**CELERITE_KERNELS, **PYMC3_KERNELS}
KERNEL_PARAMS = {
    "SHOTerm": ["sigma", "rho", "Q"],
    "ExpQuad": ["sigma", "ls"],
    "Matern32": ["sigma", "ls"],
    "Matern52": ["sigma", "ls"],
    "Periodic": ["sigma", "period", "ls"],
    "QuasiPeriodic": ["sigma", "period", "ls_per", "ls_exp"],
}


def get_pm3_kernel(
    kernel_name: str, sigma: Union[pm.Distribution, float], **kwargs
) -> Covariance:
    """
    Function that creates a PyMC3 kernel by adding an amplitude to the built-in
    covraiance functions.

    NOTE: This does not handle celerite terms/kernels.

    :param kernel_name: Kernel name (name of the class, but as a string)
    :type kernel_name: str
    :param sigma: Amplitude of the GP (pymc3 cov is multiplied by sigma^2).
                  NOTE: Not 100% sure about the type below.
    :type sigma: Union[pm.Distribution, float]
    :return: PyMC3 covariance function that can be used in regular pymc3 gp.
    :rtype: Covariance
    """

    # Always use 1 as input dimention because we deal with timeseries
    return sigma ** 2 * PYMC3_KERNELS[kernel_name](1, **kwargs)
