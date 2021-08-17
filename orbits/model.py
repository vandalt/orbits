from typing import Optional

import aesara_theano_fallback.tensor as tt
import exoplanet as xo
import numpy as np
import pymc3 as pm
from pymc3 import Model

import orbits.utils as ut
from orbits.prior import load_params


class RVModel(Model):
    def __init__(
        self,
        t: np.ndarray,
        vrad: np.ndarray,
        svrad: np.ndarray,
        params: dict[str, dict],
        num_planets: int,
        name: str = "",
        model: Optional[Model] = None,
        t_ref: Optional[float] = None,
    ):
        super().__init__(name=name, model=model)

        # Set the data attributes
        self.t = t.copy()
        self.vrad = vrad.copy()
        self.svrad = svrad.copy()

        if t_ref is None:
            self.t_ref = 0.5 * (self.t.min() + self.t.max())
        else:
            self.t_ref = t_ref

        self.num_planets = num_planets

        # Load all parameters supplied
        load_params(params["system"])
        self.submodels = dict()
        for prefix in params:
            if prefix == "system":
                # We system is used in the parent model
                continue

            # For each submodel (usually planets) we load parameters
            with pm.Model(name=prefix, model=self) as submodel:
                load_params(params[prefix])

            self.submodels[prefix] = submodel

        self._get_orbit_dict()
        self._get_rv_dict()

        # NOTE: This will move to a more generic model when other tiemseries are included
        # NOTE: Using kwds directly. To use **, would need to "translate" radvel names
        # to exoplanet or just use exoplanet names by default.
        self.orbit = xo.orbits.KeplerianOrbit(
            period=self.per, t_periastron=self.tp, ecc=self.e, omega=self.w
        )

        # Once we have our parameters, we can define the RV model at data points
        # NOTE: Wihtout name, this is just self.rv_model (Determinstic)
        self.get_rv_model(self.t)

        # Then, with the rv_model values, we define a likelihood with the RV data
        try:
            self.err = tt.sqrt(self.svrad ** 2 + self.wn ** 2)
        except AttributeError:
            self.err = self.svrad

        pm.Normal("obs", mu=self.rv_model, sd=self.err, observed=self.vrad)

    def _get_orbit_dict(self):

        odict = dict(
            per=[],
            tp=[],
            e=[],
            w=[],
        )

        nvars = self.named_vars

        for prefix in self.submodels:

            # Period-related parameters
            # exoplanet's KeplerianOrbit requires either period or semi-major axis
            PER_PARAMS = [f"{prefix}_per", f"{prefix}_logper"]
            if f"{prefix}_per" in nvars:
                odict["per"].append(nvars[f"{prefix}_per"])
            elif f"{prefix}_logper" in nvars:
                odict["per"].append(tt.exp(nvars[f"{prefix}_logper"]))
            else:
                raise KeyError(f"Should have one of: {PER_PARAMS}")

            # Ecc/omega parameters come in pair
            EW_PARAMS = [
                (f"{prefix}_e", f"{prefix}_w"),
                (f"{prefix}_secosw", f"{prefix}_sesinw"),
                (f"{prefix}_ecosw", f"{prefix}_esinw"),
                f"{prefix}_secsw",
            ]
            if f"{prefix}_e" in nvars and f"{prefix}_w" in nvars:
                odict["e"].append(nvars[f"{prefix}_e"])
                odict["w"].append(nvars[f"{prefix}_w"])
            elif "secosw" in nvars and "sesinw" in nvars:
                odict["e"].append(
                    nvars[f"{prefix}_secosw"] ** 2
                    + nvars[f"{prefix}_sesinw"] ** 2
                )
                odict["w"].append(
                    tt.arctan2(
                        nvars[f"{prefix}_sesinw"], nvars[f"{prefix}_secosw"]
                    )
                )
            elif "ecosw" in nvars and "esinw" in nvars:
                odict["e"].append(
                    tt.sqrt(
                        nvars[f"{prefix}_ecosw"] ** 2
                        + nvars[f"{prefix}_esinw"] ** 2
                    )
                )
                odict["w"].append(
                    tt.arctan2(
                        nvars[f"{prefix}_sesinw"], nvars[f"{prefix}_secosw"]
                    )
                )
            elif "secsw":
                secsw = nvars[f"{prefix}_secsw"]
                odict["e"].append(tt.sum(secsw ** 2))
                odict["w"].append(tt.arctan2(secsw[1], secsw[0]))
            else:
                raise KeyError(f"Should have one of: {EW_PARAMS}")

            # TODO: Add support for tau (epoch periastron)
            TIME_PARAMS = [f"{prefix}_tc", f"{prefix}_tp"]
            if f"{prefix}_tp" in nvars:
                odict["tp"].append(nvars[f"{prefix}_tp"])
            elif f"{prefix}_tc" in nvars:
                # odict["tc"].append(nvars["tc"])
                per = odict["per"][-1]
                ecc = odict["e"][-1]
                omega = odict["w"][-1]
                odict["tp"].append(
                    ut.timetrans_to_timeperi(
                        nvars[f"{prefix}_tc"], per, ecc, omega
                    )
                )
            else:
                raise KeyError(f"Should have one of: {TIME_PARAMS}")

        for pname in odict:
            odict[pname] = pm.Deterministic(
                pname, tt.as_tensor_variable(odict[pname])
            )

        self.odict = odict

    def _get_rv_dict(self):
        rvdict = dict()
        nvars = self.named_vars

        # TODO: Support Msini here ?
        # NOTE: This could also be in "main" orbit dict because of **kwargs
        # Probably not ideal though
        rvdict["k"] = []
        for prefix in self.submodels:
            K_PARAMS = [f"{prefix}_k", f"{prefix}_logk"]
            if f"{prefix}_k" in nvars:
                rvdict["k"] = nvars[f"{prefix}_k"]
            elif f"{prefix}_logk" in nvars:
                rvdict["k"] = tt.exp(nvars[f"{prefix}_logk"])
            else:
                raise KeyError(f"Should have one of: {K_PARAMS}")
        rvdict["k"] = pm.Deterministic("k", tt.as_tensor_variable(rvdict["k"]))

        # No alternative parametrization for this one,  also not required
        if "trend" in nvars:
            rvdict["trend"] = nvars["trend"]

        if "wn" in nvars:
            rvdict["wn"] = nvars["wn"]
        elif "logwn" in nvars:
            rvdict["wn"] = pm.Deterministic("wn", tt.exp(nvars["logwn"]))

        self.rvdict = rvdict

    def get_rv_model(self, t, name=""):
        rvorb = self.orbit.get_radial_velocity(t, K=self.k)
        pm.Deterministic("rv_orbits" + name, rvorb)

        # Define the background RV model
        # NOTE: If trend is not defined we just don't add bkg term
        try:
            # HACK: Not sure this is the best way to get the "trend" shape
            A = np.vander(t - self.t_ref, self.trend.shape.get_test_value()[0])
            bkg = pm.Deterministic("bkg" + name, tt.dot(A, self.trend))
        except AttributeError:
            bkg = pm.Deterministic("bkg", tt.as_tensor_variable(0.0))

        # Sum over planets and add the background to get the full model
        return pm.Deterministic("rv_model" + name, tt.sum(rvorb, axis=-1) + bkg)
