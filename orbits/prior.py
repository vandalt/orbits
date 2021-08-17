from typing import Optional

import aesara_theano_fallback.tensor as tt
import numpy as np
import pymc3 as pm
from pymc3.distributions.distribution import Distribution
import pymc3_ext as pmx


def fixed_pymc3_param(name: str, value: float) -> Distribution:
    return pm.Deterministic(name, tt.as_tensor_variable(value))


def data_normal_prior(
    name: str,
    data_used=None,
    sd: Optional[float] = None,
    nsigma: Optional[float] = None,
    central_measure: str = "mean",
) -> Distribution:

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
    "UnitDisk": pmx.UnitDisk
}


def load_params(params: dict[str, dict]) -> dict[str, Distribution]:

    out_dict = dict()

    for pname, pdict in params.items():

        out_dict[pname] = read_prior(pname, pdict)

    return pdict


def read_prior(pname: str, pdict: dict[str, dict]) -> Distribution:
    return PYMC3_PRIORS[pdict["dist"]](pname, **pdict["kwargs"])
