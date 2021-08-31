"""
Theano calculations to convert orbit parameters easily.

NOTE: Alternative OOP setup could be:
    OrbitParam base object with name, alt parmetrizations/converter list
    Implementations with dict of mapping from alt to methods to convert
    Then could have extra things per parameter
    But would still need to be in model context...
    Prefix stuff would be abstracted
"""
import aesara_theano_fallback.tensor as tt
import pymc3 as pm

import orbits.utils as ut

SYNTH_PARAMS = ["per", "tp", "e", "w", "k"]

ALT_PARAMS = {
    "per": ["per", "logper"],
    "ew": [("e", "w"), ("secosw", "sesinw"), ("ecosw", "esinw"), "secsw"],
    "tp": ["tp", "tc"],
    "k": ["k", "logk"],
}


def get_synth_params(
    params: dict[str, tt.TensorVariable], prefix: str = ""
) -> dict[str, tt.TensorVariable]:
    """
    Get parameters in "synth" basis from radvel (per, tp, e, w, k).
    These parameters will be used for orbit calculations.
    This function assumes a pymc3 model context, because it creates
    deterministic variables

    NOTE: This might move to a more flexible approach with many small
    functions or an OrbitParameter class that handles parameter transformation
    (similar to RadVel's Basis, but per parameter).

    :param params: Parameter dictionary
                   (usually `.named_vars` from pymc3 model)
    :type params: dict[str, tt.TensorVariable]
    :param prefix: Prefix before usual param name (usually pymc3 model name),
                   so `_` will be added after, defaults to ""
    :type prefix: str, optional
    :return: Dictionary with synth params
    :rtype: dict[str, tt.TensorVariable]
    :raises KeyError: If no parameter is found for a given synth param
    """

    prefix = f"{prefix}_" if prefix != "" else ""

    synth_params = dict()

    # Period-related parameters
    # exoplanet's KeplerianOrbit requires period or semi-major axis
    PER_PARAMS = [f"{prefix}per", f"{prefix}logper"]
    if f"{prefix}per" in params:
        synth_params["per"] = params[f"{prefix}per"]
    elif f"{prefix}logper" in params:
        # NOTE: No prefix when declaring bc model name takes care of it
        # but required to access with nvars
        synth_params["per"] = pm.Deterministic(
            "per", tt.exp(params[f"{prefix}logper"])
        )
    else:
        raise KeyError(f"Should have one of: {PER_PARAMS}")

    # Ecc/omega parameters come in pair
    EW_PARAMS = [
        (f"{prefix}e", f"{prefix}w"),
        (f"{prefix}secosw", f"{prefix}sesinw"),
        (f"{prefix}ecosw", f"{prefix}esinw"),
        f"{prefix}secsw",
    ]
    if f"{prefix}e" in params and f"{prefix}w" in params:
        synth_params["e"] = params[f"{prefix}e"]
        synth_params["w"] = params[f"{prefix}w"]
    elif "secosw" in params and "sesinw" in params:
        synth_params["e"] = pm.Deterministic(
            "e",
            params[f"{prefix}secosw"] ** 2 + params[f"{prefix}sesinw"] ** 2,
        )
        synth_params["w"] = pm.Deterministic(
            "w",
            tt.arctan2(params[f"{prefix}sesinw"], params[f"{prefix}secosw"]),
        )
    elif "ecosw" in params and "esinw" in params:
        synth_params["e"] = pm.Deterministic(
            "e",
            tt.sqrt(
                params[f"{prefix}ecosw"] ** 2 + params[f"{prefix}esinw"] ** 2
            ),
        )
        synth_params["w"] = pm.Deterministic(
            "w",
            tt.arctan2(params[f"{prefix}sesinw"], params[f"{prefix}secosw"]),
        )
    elif "secsw":
        secsw = params[f"{prefix}secsw"]
        synth_params["e"] = pm.Deterministic("e", tt.sum(secsw ** 2))
        synth_params["w"] = pm.Deterministic(
            "w", tt.arctan2(secsw[1], secsw[0])
        )
    else:
        raise KeyError(f"Should have one of: {EW_PARAMS}")

    TIME_PARAMS = [f"{prefix}tc", f"{prefix}tp"]
    if f"{prefix}tp" in params:
        synth_params["tp"] = params[f"{prefix}tp"]
    elif f"{prefix}tc" in params:
        per = synth_params["per"]
        ecc = synth_params["e"]
        omega = synth_params["w"]
        synth_params["tp"] = pm.Deterministic(
            "tp",
            ut.timetrans_to_timeperi(params[f"{prefix}tc"], per, ecc, omega),
        )
    else:
        raise KeyError(f"Should have one of: {TIME_PARAMS}")

    K_PARAMS = [f"{prefix}k", f"{prefix}logk"]
    if f"{prefix}k" in params:
        synth_params["k"] = params[f"{prefix}k"]
    elif f"{prefix}logk" in params:
        synth_params["k"] = pm.Deterministic(
            "k", tt.exp(params[f"{prefix}logk"])
        )
    else:
        raise KeyError(f"Should have one of: {K_PARAMS}")

    return synth_params
