from typing import Optional, Union

import aesara_theano_fallback.tensor as tt
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from pymc3.distributions.distribution import Distribution
from pymc3.model import DeterministicWrapper
from theano.tensor.var import TensorConstant, TensorVariable
from exoplanet.interp import regular_grid_interp

def fixed_pymc3_param(name: str, value: float) -> DeterministicWrapper:
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


def wmed(
    values: Union[np.ndarray, TensorVariable, TensorConstant],
    weights: Union[np.ndarray, TensorVariable, TensorConstant],
    use_np: bool = True,
) -> Union[float, TensorVariable, TensorConstant]:
    """
    Calculate a weighted median with numpy or theano.

    :param values: Sample values to get the median.
    :type values: Union[np.ndarray, TensorVariable, TensorConstant]
    :param weights: Weights associated to each value.
    :type weights: Union[np.ndarray, TensorVariable, TensorConstant]
    :return: Weighted median of the sample values.
    :rtype: Union[float, TensorVariable, TensorConstant]
    """
    if use_np:
        values = np.array(values)
        weights = np.array(weights)
        if not np.all(np.diff(values) >= 0):
            inds = np.argsort(values)
            values = values[inds]
            weights = weights[inds]
        norm_weights = weights / weights.sum()
        wquants = np.cumsum(norm_weights) - 0.5 * norm_weights
        res = np.interp(0.5, wquants, values)
    else:
        values = tt.as_tensor_variable(values)
        weights = tt.as_tensor_variable(weights)
        if not tt.all(tt.ge(tt.extra_ops.diff(values), 0)):
            inds = tt.argsort(values)
            values = values[inds]
            weights = weights[inds]
        norm_weights = weights / weights.sum()
        wquants = tt.extra_ops.cumsum(norm_weights) - 0.5 * norm_weights
        # exoplanet interp function requires specific shapes
        # 0-d (scalar) does not matter much but let's be consistent
        res = regular_grid_interp([wquants], values, np.array([[0.5]]).T)[0]

    return res


def data_fixed_prior(
    name: str,
    data_used: str,
    central_measure: str = "mean",
    error_used: Optional[str] = None,
) -> DeterministicWrapper:
    # TODO: Merge this and data_normal_prior with class or helper functions

    try:
        # Get values from PyMC3 Data
        values = pm.Model.get_context()[data_used].get_value()
        if error_used is not None:
            err = pm.Model.get_context()[error_used].get_value()
    except KeyError:
        # If not pymc3 data, try attribute array
        values = getattr(pm.Model.get_context(), data_used)
        if error_used is not None:
            err = getattr(pm.Model.get_context(), error_used)
    except AttributeError:
        raise ValueError(
            "data_used must point to a PyMC3 dataset or array in the model."
        )

    if central_measure == "mean":
        if error_used is not None:
            center = np.average(values, weights=err ** -2)
        else:
            center = np.mean(values)
    elif central_measure == "median":
        if error_used is not None:
            center = wmed(values, weights=err ** -2)
        else:
            center = np.median(values)
    else:
        raise ValueError("central_measure must be mean or median")

    return fixed_pymc3_param(name, center)


PYMC3_PRIORS = {
    "Uniform": pm.Uniform,
    "Normal": pm.Normal,
    "TruncatedNormal": pm.TruncatedNormal,
    "Fixed": fixed_pymc3_param,
    "DataNormal": data_normal_prior,
    "DataFixed": data_fixed_prior,
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
