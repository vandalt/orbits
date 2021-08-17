import numpy as np
import aesara_theano_fallback.tensor as tt


# TODO: Credit radvel in docstrings.

def timetrans_to_timeperi(tc, per, ecc, omega, use_np=False):

    if use_np:
        f = np.pi/2 - omega
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))
        tp = tc - per/(2*np.pi) * (ee - ecc*np.sin(ee))
    else:
        f = np.pi/2 - omega
        ee = 2 * tt.arctan(tt.tan(f/2) * tt.sqrt((1-ecc)/(1+ecc)))
        tp = tc - per/(2*np.pi) * (ee - ecc*tt.sin(ee))

    return tp


def timeperi_to_timetrans(tp, per, ecc, omega, use_np=False):

    if use_np:
        f = np.pi/2 - omega
        ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))
        tc = tp + per/(2*np.pi) * (ee - ecc*np.sin(ee))
    else:
        f = np.pi/2 - omega
        ee = 2 * tt.arctan(tt.tan(f/2) * tt.sqrt((1-ecc)/(1+ecc)))
        tc = tp + per/(2*np.pi) * (ee - ecc*tt.sin(ee))

    return tc
