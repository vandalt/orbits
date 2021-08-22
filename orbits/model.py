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

SYNT_PARAMS = ["per", "tp", "e", "w", "k"]


class PlanetModel(Model):
    """Model that stores info about one planet"""

    # TODO: Add supported input parameters info
    def __init__(
        self,
        params: dict[str, dict],
        name: str = "",
        model: Optional[Model] = None,
    ):
        """
        PyMC3 model that represents a single planet. This model is mainly a
        container for planetary parameters of a single planet.

        :param params: Planet orbit parameters. These parameters are used to
                       define the orbit of the planet. Under the hood, the
                       model ensures that [per, tp, e, w, k] are available.
        :type params: dict[str, dict]
        :param name: PyMC3 model name that will prefix all variables,
                     defaults to ""
        :type name: str, optional
        :param model: Parent PyMC3 model, defaults to None
        :type model: Optional[Model], optional
        """
        super().__init__(name=name, model=model)

        # Load planet parameters and convert to synth params under the hood
        # (required for orbit modelling)
        load_params(params)
        self._get_synth_params()

    def _get_synth_params(self):
        """
        Get parameters in "synth" basis from radvel (per, tp, e, w, k).
        These parameters will be used for orbit calculations.
        """

        nvars = self.named_vars
        prefix = f"{self.name}_" if self.name != "" else ""

        # Period-related parameters
        # exoplanet's KeplerianOrbit requires period or semi-major axis
        PER_PARAMS = [f"{prefix}per", f"{prefix}logper"]
        if f"{prefix}per" in nvars:
            pass
        elif f"{prefix}logper" in nvars:
            # NOTE: No prefix when declaring bc model name takes care of it
            # but required to access with nvars
            pm.Deterministic("per", tt.exp(nvars[f"{prefix}logper"]))
        else:
            raise KeyError(f"Should have one of: {PER_PARAMS}")

        # Ecc/omega parameters come in pair
        EW_PARAMS = [
            (f"{prefix}e", f"{prefix}w"),
            (f"{prefix}secosw", f"{prefix}sesinw"),
            (f"{prefix}ecosw", f"{prefix}esinw"),
            f"{prefix}secsw",
        ]
        if f"{prefix}e" in nvars and f"{prefix}w" in nvars:
            pass
        elif "secosw" in nvars and "sesinw" in nvars:
            pm.Deterministic(
                "e",
                nvars[f"{prefix}secosw"] ** 2 + nvars[f"{prefix}sesinw"] ** 2,
            )
            pm.Deterministic(
                "w",
                tt.arctan2(nvars[f"{prefix}sesinw"], nvars[f"{prefix}secosw"]),
            )
        elif "ecosw" in nvars and "esinw" in nvars:
            pm.Deterministic(
                "e",
                tt.sqrt(
                    nvars[f"{prefix}ecosw"] ** 2 + nvars[f"{prefix}esinw"] ** 2
                ),
            )
            pm.Deterministic(
                "w",
                tt.arctan2(nvars[f"{prefix}sesinw"], nvars[f"{prefix}secosw"]),
            )
        elif "secsw":
            secsw = nvars[f"{prefix}secsw"]
            pm.Deterministic("e", tt.sum(secsw ** 2))
            pm.Deterministic("w", tt.arctan2(secsw[1], secsw[0]))
        else:
            raise KeyError(f"Should have one of: {EW_PARAMS}")

        # TODO: Add support for tau (epoch periastron)
        TIME_PARAMS = [f"{prefix}tc", f"{prefix}tp"]
        if f"{prefix}tp" in nvars:
            pass
        elif f"{prefix}tc" in nvars:
            per = self.per
            ecc = self.e
            omega = self.w
            pm.Deterministic(
                "tp",
                ut.timetrans_to_timeperi(
                    nvars[f"{prefix}tc"], per, ecc, omega
                ),
            )
        else:
            raise KeyError(f"Should have one of: {TIME_PARAMS}")

        # TODO: Support Msini here ?
        # NOTE: This could also be in "main" orbit dict because of **kwargs
        # Probably not ideal though
        K_PARAMS = [f"{prefix}k", f"{prefix}logk"]
        if f"{prefix}k" in nvars:
            pass
        elif f"{prefix}logk" in nvars:
            pm.Deterministic("k", tt.exp(nvars[f"{prefix}logk"]))
        else:
            raise KeyError(f"Should have one of: {K_PARAMS}")


class RVModel(Model):
    """Model for a given RV Dataset"""

    def __init__(
        self,
        t: np.ndarray,
        vrad: np.ndarray,
        svrad: np.ndarray,
        params: dict[str, dict],
        num_planets: int,
        t_ref: Optional[float] = None,
        gp_kernel: Optional[str] = None,
        quiet_celerite: bool = False,
        name: str = "",
        model: Optional[Model] = None,
    ):
        super().__init__(name=name, model=model)

        # Set the data attributes
        self.t = np.array(t)
        self.vrad = np.array(vrad)
        self.svrad = np.array(svrad)

        if t_ref is None:
            self.t_ref = 0.5 * (self.t.min() + self.t.max())
        else:
            self.t_ref = t_ref

        self.num_planets = num_planets

        # Load all parameters supplied
        load_params(params["system"])
        if gp_kernel is not None:
            with pm.Model(name="gp", model=self) as gpmodel:
                load_params(params["gp"])
            self.gpmodel = gpmodel
            self._get_gp_dict(gp_kernel)
        else:
            self.gpmodel = None

        # TODO: Probably safer to explicitely iterate planets somehow
        # (not just skip "non-planets")
        self.planets = dict()
        for prefix in params:

            if prefix in ["system", "gp"]:
                # We system is used in the parent model
                continue

            # For each planet, we load parameters.
            # Using submodel makes "{letter}_" prefix auto
            self.planets[prefix] = PlanetModel(
                params[prefix], name=prefix, model=self
            )

        self._get_synth_params()
        self._get_rv_params()

        # NOTE: This will move to a more generic model when other tiemseries
        # are included
        # NOTE: Using kwds directly. To use **, would need to "translate"
        # radvel names
        # to exoplanet or just use exoplanet names by default.
        self.orbit = xo.orbits.KeplerianOrbit(
            period=self.synth_dict["per"],
            t_periastron=self.synth_dict["tp"],
            ecc=self.synth_dict["e"],
            omega=self.synth_dict["w"],
        )

        # Once we have our parameters, we define the RV model at data points
        # NOTE: Wihtout name, this is just self.rv_model (Determinstic)
        self.get_rv_model(self.t)

        # Then, with the rv_model, we define a likelihood with the RV data
        self.err = tt.sqrt(self.svrad ** 2 + self.wn ** 2)

        if gp_kernel is None:
            pm.Normal("obs", mu=self.rv_model, sd=self.err, observed=self.vrad)
        else:
            self.kernel = KERNELS[gp_kernel](**self.gpdict)
            self.gp = GaussianProcess(
                self.kernel, t=self.t, yerr=self.err, quiet=quiet_celerite
            )
            self.resid = self.vrad - self.rv_model
            self.gp.marginal("obs", observed=self.resid)

    def _get_synth_params(self):
        synth_dict = dict()
        nvars = self.named_vars
        for pname in SYNT_PARAMS:
            synth_dict[pname] = pm.Deterministic(
                pname,
                tt.as_tensor_variable(
                    [nvars[f"{prefix}_{pname}"] for prefix in self.planets]
                ),
            )

        self.synth_dict = synth_dict

    def _get_rv_params(self):
        nvars = self.named_vars

        if "gamma" not in nvars:
            pm.Deterministic("gamma", tt.as_tensor_variable(0.0))

        if "dvdt" not in nvars:
            pm.Deterministic("dvdt", tt.as_tensor_variable(0.0))

        if "curv" not in nvars:
            pm.Deterministic("curv", tt.as_tensor_variable(0.0))

        if "wn" in nvars:
            pass
        elif "logwn" in nvars:
            pm.Deterministic("wn", tt.exp(nvars["logwn"]))
        else:
            pm.Deterministic("wn", tt.as_tensor_variable(0.0))

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
                    f"Kernel {kernel_name} requires parameter "
                    f"{pname} or log{pname}"
                )
        self.gpdict = gpdict

    def get_rv_model(self, t, name=""):
        rvorb = self.orbit.get_radial_velocity(t, K=self.synth_dict["k"])
        pm.Deterministic("rv_orbits" + name, rvorb)

        # Define the background RV model
        # NOTE: If trend is not defined we just don't add bkg term
        # HACK: Not sure this is the best way to get the "trend" shape
        t_shift = t - self.t_ref
        A = np.vander(t_shift, self.trend.shape.get_test_value()[0])
        bkg = pm.Deterministic("bkg" + name, tt.dot(A, self.trend))
        # bkg = self.gamma
        # bkg += self.dvdt * t_shift
        # bkg += self.curv * t_shift ** 2
        # bkg = pm.Deterministic("bkg" + name, bkg)

        # Sum over planets and add the background to get the full model
        return pm.Deterministic(
            "rv_model" + name, tt.sum(rvorb, axis=-1) + bkg
        )
