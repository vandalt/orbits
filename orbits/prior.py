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


DATA_NORMAL_FUNCTIONS = {
    "log": np.log,
    "exp": np.exp,
}


def data_normal_prior(
    name: str,
    data_used: str,
    sd: Optional[float] = None,
    nsigma: Optional[float] = None,
    central_measure: str = "mean",
    apply: Optional[str] = None,
) -> Distribution:
    """
    Helper function to create a "DataNormal" distribution. This distribution
    is a normal distribution around the center of a given dataset.

    One of sd and nsigma must be used.

    :param name: Parameter name
    :type name: str
    :param data_used: Name of the data used to define the prior
                      (pymc3 data object)
    :type data_used: np.ndarray, optional
    :param sd: Standard deviation of the distribution, defaults to None
    :type sd: Optional[float], optional
    :param nsigma: Constant to scale the std. dev. from the dataset's
                   deviation, defaults to None
    :type nsigma: Optional[float], optional
    :param central_measure: Central measured use from the dataset, defaults to
                            "mean"
    :type central_measure: str, optional
    :param apply: Name of the function applied to values before using them,
                  defaults to None
    :type apply: Optional[str], optional
    :return: PyMC3 normal distribution
    :rtype: Distribution
    """

    try:
        # Get values from PyMC3 Data
        values = pm.Model.get_context()[data_used].get_value()
    except KeyError:
        # If not pymc3 data, try attribute array
        values = getattr(pm.Model.get_context(), data_used)
    except AttributeError:
        raise ValueError(
            "data_used must point to a PyMC3 dataset or array in the model."
        )

    if apply is not None:
        if apply in DATA_NORMAL_FUNCTIONS:
            values = DATA_NORMAL_FUNCTIONS[apply](values)
        else:
            raise ValueError(
                f"apply must be None or one of {list(DATA_NORMAL_FUNCTIONS)}"
            )

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
