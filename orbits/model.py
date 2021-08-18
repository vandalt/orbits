from typing import Optional

import aesara_theano_fallback.tensor as tt
import exoplanet as xo
import numpy as np
import pymc3 as pm
from celerite2.theano import GaussianProcess, terms
from pymc3 import Model

import orbits.utils as ut
from orbits.prior import load_params

KERNELS = {"SHOTerm": terms.SHOTerm}

KERNEL_PARAMS = {"SHOTerm": ["sigma", "rho", "Q"]}


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
        gp_kernel: Optional[str] = None,
        quiet_celerite: bool = False,
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
        # TODO: Maybe should specify extra level for ['planets'] in input to loop only this.
        load_params(params["system"])
        if gp_kernel is not None:
            with pm.Model(name="gp", model=self) as gpmodel:
                load_params(params["gp"])
            self.gpmodel = gpmodel
            self._get_gp_dict(gp_kernel)
        self.submodels = dict()
        for prefix in params:
            if prefix in ["system", "gp"]:
                # We system is used in the parent model
                continue

            # For each planet, we load parameters.
            # Using submodel makes "{letter}_" prefix auto
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
        self.err = tt.sqrt(self.svrad ** 2 + self.wn ** 2)

        if gp_kernel is None:
            pm.Normal("obs", mu=self.rv_model, sd=self.err, observed=self.vrad)
        else:
            self.kernel = KERNELS[gp_kernel](**self.gpdict)
            self.gp = GaussianProcess(self.kernel, t=self.t, yerr=self.err, quiet=quiet_celerite)
            self.resid = self.vrad - self.rv_model
            self.gp.marginal("obs", observed=self.resid)

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
                rvdict["k"].append(nvars[f"{prefix}_k"])
            elif f"{prefix}_logk" in nvars:
                rvdict["k"].append(tt.exp(nvars[f"{prefix}_logk"]))
            else:
                raise KeyError(f"Should have one of: {K_PARAMS}")
        rvdict["k"] = pm.Deterministic("k", tt.as_tensor_variable(rvdict["k"]))

        if "gamma" in nvars:
            rvdict["gamma"] = nvars["gamma"]
        else:
            rvdict["gamma"] = pm.Deterministic(
                "gamma", tt.as_tensor_variable(0.0)
            )

        if "dvdt" in nvars:
            rvdict["dvdt"] = nvars["dvdt"]
        else:
            rvdict["dvdt"] = pm.Deterministic(
                "dvdt", tt.as_tensor_variable(0.0)
            )

        if "curv" in nvars:
            rvdict["curv"] = nvars["curv"]
        else:
            rvdict["curv"] = pm.Deterministic(
                "curv", tt.as_tensor_variable(0.0)
            )

        if "wn" in nvars:
            rvdict["wn"] = nvars["wn"]
        elif "logwn" in nvars:
            rvdict["wn"] = pm.Deterministic("wn", tt.exp(nvars["logwn"]))
        else:
            rvdict["wn"] = pm.Deterministic("wn", tt.as_tensor_variable(0.0))

        self.rvdict = rvdict

    def _get_gp_dict(self, kernel_name: str):
        gpdict = dict()

        nvars = self.gpmodel.named_vars
        for pname in KERNEL_PARAMS[kernel_name]:
            if f"gp_{pname}" in nvars:
                gpdict[f"{pname}"] = nvars[f"gp_{pname}"]
            elif f"gp_log{pname}" in nvars:
                gpdict[f"{pname}"] = pm.Deterministic(
                    f"gp_{pname}", tt.exp(nvars[f"gp_log{pname}"])
                )
            else:
                raise ValueError(
                    f"Kernel {kernel_name} requires parameter {pname} or log{pname}"
                )
        self.gpdict = gpdict

    def get_rv_model(self, t, name=""):
        rvorb = self.orbit.get_radial_velocity(t, K=self.k)
        pm.Deterministic("rv_orbits" + name, rvorb)

        # Define the background RV model
        # NOTE: If trend is not defined we just don't add bkg term
        # HACK: Not sure this is the best way to get the "trend" shape
        # A = np.vander(t - self.t_ref, self.trend.shape.get_test_value()[0])
        t_shift = t - self.t_ref
        bkg = self.gamma
        bkg += self.dvdt * t_shift
        bkg += self.curv * t_shift ** 2
        bkg = pm.Deterministic("bkg" + name, bkg)

        # Sum over planets and add the background to get the full model
        return pm.Deterministic(
            "rv_model" + name, tt.sum(rvorb, axis=-1) + bkg
        )
