from typing import Union

import aesara_theano_fallback.tensor as tt
import numpy as np


def timetrans_to_timeperi(
    tc: Union[tt.TensorVariable, np.ndarray, float],
    per: Union[tt.TensorVariable, np.ndarray, float],
    ecc: Union[tt.TensorVariable, np.ndarray, float],
    omega: Union[tt.TensorVariable, np.ndarray, float],
    use_np: bool = False,
) -> Union[tt.TensorVariable, np.ndarray, float]:
    """
     Convert transit time to time of periastron.

    This is a modified version of the similar function in `radvel.orbit`, but
     with support for Theano/PyMC3 variables. The radvel source code can be
     found here: https://github.com/California-Planet-Search/radvel.

     :param tc: Time of transit (conjunction)
     :type tc: Union[tt.TensorVariable, np.ndarray, float]
     :param per: Orbit period
     :type per: Union[tt.TensorVariable, np.ndarray, float]
     :param ecc: Eccentricity
     :type ecc: Union[tt.TensorVariable, np.ndarray, float]
     :param omega: Argument of periastron in radians (of the star's orbit)
     :type omega: Union[tt.TensorVariable, np.ndarray, float]
     :param use_np: Use numpy if True (theano otherwise), defaults to False
     :type use_np: bool, optional
     :return: Time of periastron
     :rtype: Union[tt.TensorVariable, np.ndarray, float]
    """

    # TODO: add type checking
    if use_np:
        f = np.pi / 2 - omega
        ee = 2 * np.arctan(np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc)))
        tp = tc - per / (2 * np.pi) * (ee - ecc * np.sin(ee))
    else:
        f = np.pi / 2 - omega
        ee = 2 * tt.arctan(tt.tan(f / 2) * tt.sqrt((1 - ecc) / (1 + ecc)))
        tp = tc - per / (2 * np.pi) * (ee - ecc * tt.sin(ee))

    return tp


def timeperi_to_timetrans(
    tp: Union[tt.TensorVariable, np.ndarray, float],
    per: Union[tt.TensorVariable, np.ndarray, float],
    ecc: Union[tt.TensorVariable, np.ndarray, float],
    omega: Union[tt.TensorVariable, np.ndarray, float],
    use_np: bool = False,
) -> Union[tt.TensorVariable, np.ndarray, float]:
    """
    Convert time of periastron to transit time.

    This is a modified version of the similar function in `radvel.orbit`, but
    with support for Theano/PyMC3 variables. The radvel source code can be
    found here: https://github.com/California-Planet-Search/radvel.

    :param tp: Time of periastron
    :type tp: Union[tt.TensorVariable, np.ndarray, float]
    :param per: Orbit period
    :type per: Union[tt.TensorVariable, np.ndarray, float]
    :param ecc: Eccentricity
    :type ecc: Union[tt.TensorVariable, np.ndarray, float]
    :param omega: Argument of periastron (of the star's orbit)
    :type omega: Union[tt.TensorVariable, np.ndarray, float]
    :param use_np: Use numpy if True (theano otherwise), defaults to False
    :type use_np: bool, optional
    :return: Time of transit (conjunction)
    :rtype: Union[tt.TensorVariable, np.ndarray, float]
    """

    if use_np:
        f = np.pi / 2 - omega
        ee = 2 * np.arctan(np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc)))
        tc = tp + per / (2 * np.pi) * (ee - ecc * np.sin(ee))
    else:
        f = np.pi / 2 - omega
        ee = 2 * tt.arctan(tt.tan(f / 2) * tt.sqrt((1 - ecc) / (1 + ecc)))
        tc = tp + per / (2 * np.pi) * (ee - ecc * tt.sin(ee))

    return tc
