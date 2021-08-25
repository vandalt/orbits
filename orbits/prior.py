from typing import Optional

import aesara_theano_fallback.tensor as tt
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from pymc3.distributions.distribution import Distribution


def fixed_pymc3_param(name: str, value: float) -> Distribution:
    """
    Helper function to create a fixed PyMC3 parameter.

    :param name: Parameter name (in prior)
    :type name: str
    :param value: Fixed value of the parameter
    :type value: float
    :return: PyMC3 Determinic distribution.
    :rtype: Distribution
    """
    return pm.Deterministic(name, tt.as_tensor_variable(value))


def data_normal_prior(
    name: str,
    data_used=None,
    sd: Optional[float] = None,
    nsigma: Optional[float] = None,
    central_measure: str = "mean",
) -> Distribution:
    """
    Helper function to create a "DataNormal" distribution. This distribution
    is a normal distribution around the center of a given dataset.

    One of sd and nsigma must be used.

    :param name: Parameter name
    :type name: str
    :param data_used: Dataset used to define the distribution, defaults to None
    :type data_used: np.ndarray, optional
    :param sd: Standard deviation of the distribution, defaults to None
    :type sd: Optional[float], optional
    :param nsigma: Constant to scale the std. dev. from the dataset's
                   deviation, defaults to None
    :type nsigma: Optional[float], optional
    :param central_measure: Central measured use from the dataset, defaults to
                            "mean"
    :type central_measure: str, optional
    :return: PyMC3 normal distribution
    :rtype: Distribution
    """

    # values = pm.Model.get_context()[data_used].get_value()
    if data_used is None:
        raise TypeError("data_used must be provided.")
    values = data_used

    if sd is not None and nsigma is not None:
        raise TypeError("Only one of sd and nsigma must be provided")
    elif sd is not None:
        std_dev = sd
    elif nsigma is not None:
        std_dev = nsigma * np.std(values)
    else:
        raise TypeError("sd or nsigma must be provided")

    if central_measure == "mean":
        center = np.mean(values)
    elif central_measure == "median":
        center = np.median(values)
    else:
        raise ValueError("central_measure must be mean or median")

    return pm.Normal(name, center, std_dev)


PYMC3_PRIORS = {
    "Uniform": pm.Uniform,
    "Normal": pm.Normal,
    "TruncatedNormal": pm.TruncatedNormal,
    "Fixed": fixed_pymc3_param,
    "DataNormal": data_normal_prior,
    "UnitDisk": pmx.UnitDisk,
}


def load_params(params: dict[str, dict]) -> dict[str, Distribution]:
    """
    Read a dictionary of paramter definitions information and create a dict
    of PyMC3 distributions.

    Must be used in a PyMC3 model context.

    :param params: Dictionary of parameter informations
    :type params: dict[str, dict]
    :return: Dictioanry of PyMC3 distributions
    :rtype: dict[str, Distribution]
    """

    out_dict = dict()

    for pname, pdict in params.items():

        out_dict[pname] = read_prior(pname, pdict)

    return out_dict


def read_prior(pname: str, pdict: dict[str, dict]) -> Distribution:
    """
    Read information for a single parameter and return a PyMC3.

    :param pname: Parameter name
    :type pname: str
    :param pdict: Dictionary of parameter info
    :type pdict: dict[str, dict]
    :return: PyMC3 distribution corresponding to input info
    :rtype: Distribution
    """
    return PYMC3_PRIORS[pdict["dist"]](pname, **pdict["kwargs"])
