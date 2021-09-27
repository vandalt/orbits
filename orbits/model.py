import warnings
from typing import Optional, Union

import aesara_theano_fallback.tensor as tt
import exoplanet as xo
import numpy as np
import pymc3 as pm
from celerite2.theano import GaussianProcess
from pymc3 import Model

from orbits.kernel import (CELERITE_KERNELS, KERNEL_LIST, KERNEL_PARAMS,
                           KERNELS, PYMC3_KERNELS)
from orbits.params import SYNTH_PARAMS, get_synth_params
from orbits.prior import load_params


class PlanetModel(Model):
    def __init__(
        self,
        params: Optional[dict[str, dict]] = None,
        name: str = "",
        model: Optional[Model] = None,
    ):
        """
        PyMC3 model that represents a single planet. This model is mainly a
        container for planetary parameters of a single planet and it does not
        contain any orbit calculation. The nomenclature of planet parameters
        is mostly consistent with RadVel. This is generally used in an orbit
        modle (e.g. RVModel), but can be created directly by users.

        :param params: Planet orbit parameters. These parameters are used to
                       define the orbit of the planet. Under the hood, the
                       model ensures that [per, tp, e, w, k] are available.
                       Defaults to None
        :type params: Optional[dict[str, dict]], optional
        :param name: PyMC3 model name that will prefix all variables,
                     defaults to ""
        :type name: str, optional
        :param model: Parent PyMC3 model, defaults to None
        :type model: Optional[Model], optional
        """
        super().__init__(name=name, model=model)

        if params is not None:
            load_params(params)
            self._synth_params = self._get_synth_params()
        else:
            self._synth_params = None

    @property
    def synth_params(self):
        if self._synth_params is not None:
            return self._synth_params
        else:
            self._synth_params = self._get_synth_params()
            return self._synth_params

    def _get_synth_params(self):
        """
        Get parameters in "synth" basis from radvel (per, tp, e, w, k).
        These parameters will be used for orbit calculations.
        """

        # If this is called from a parent model, we need to change context
        with self:
            return get_synth_params(self.named_vars, prefix=self.name)


class GPModel(Model):
    def __init__(
        self,
        kernel_name: str,
        params: dict[str, dict] = None,
        name: str = "",
        model: Model = None,
    ):
        """
        PyMC3 model that stores parameters of a Gaussian Process.
        This can be used with the supported kernels from celerite2 or PyMC3.

        :param kernel_name: Name of the kernel used. Usually this is the name
                            of the class in celerite2 or PyMC3.
        :type kernel_name: str
        :param params: Dictionary with GP parameter info, defaults to None
        :type params: Optional[dict[str, dict]], optional
        :param name: PyMC3 model name that will prefix all variables,
                     defaults to ""
        :type name: str, optional
        :param model: Parent PyMC3 model, defaults to None
        :type model: Optional[Model], optional
        """
        super().__init__(name=name, model=model)

        self.kernel_name = kernel_name

        if params is not None:
            load_params(params)
            self._gp_dict = self._get_gp_dict()
            # KERNELS is just a dict mapping name to object constructors
            self._kernel = KERNELS[self.kernel_name](**self.gp_dict)
        else:
            # NOTE: Not useful for kernel (required), keeping this way in case
            self._kernel = None
            self._gp_dict = None

    @property
    def kernel(self):
        # If the kernel is not defined yet, we define it and return to the user
        if self._kernel is not None:
            return self._kernel
        else:
            with self:
                self._kernel = KERNELS[self.kernel_name](**self.gp_dict)
            return self._kernel

    @property
    def gp_dict(self):
        if self._gp_dict is not None:
            return self._gp_dict
        else:
            self._gp_dict = self._get_gp_dict()

    def _get_gp_dict(self):
        """
        Make sure that the model has all required GP parameters.
        This handles "log{param}" cases and also makes sure that all required
        parameters for a given kernel are there.
        """

        # If called from parent, need to switch context
        with self:
            gp_dict = dict()
            nvars = self.named_vars
            for pname in KERNEL_PARAMS[self.kernel_name]:
                if f"{self.prefix}{pname}" in nvars:
                    gp_dict[pname] = nvars[f"{self.prefix}{pname}"]
                elif f"{self.prefix}log{pname}" in nvars:
                    gp_dict[pname] = pm.Deterministic(
                        pname, tt.exp(nvars[f"{self.prefix}log{pname}"])
                    )
                else:
                    raise ValueError(
                        f"Kernel {self.kernel_name} requires parameter "
                        f"{pname} or log{pname}"
                    )

        return gp_dict


class RVModel(Model):
    def __init__(
        self,
        t: np.ndarray,
        vrad: np.ndarray,
        svrad: np.ndarray,
        num_planets: int,
        params: Optional[dict[str, dict]] = None,
        t_ref: Optional[float] = None,
        gp_kernel: Optional[str] = None,
        planet_names: Optional[Union[str, list[str]]] = None,
        quiet_celerite: bool = False,
        name: str = "",
        model: Optional[Model] = None,
    ):
        """
        PyMC3 model for an RV dataset using `exoplanet` to calculate orbits.

        :param t: Time values
        :type t: np.ndarray
        :param vrad: Radial velocity values
        :type vrad: np.ndarray
        :param svrad: Radial velocity uncertainties
        :type svrad: np.ndarray
        :param params: Dictionary with parameter information.
                       System-wide parameters should be in the "system" key.
                       GP parameters should be under the "gp" key.
                       All other keys are treated as planets (this may change
                       to avoid ambiguities in the future), defaults to None
        :type params: Optional[dict[str, dict]], optional
        :param num_planets: Number of planets to model in the dataset.
        :type num_planets: int
        :param t_ref: Reference time for RV trend parameters, defaults to None
        :type t_ref: Optional[float], optional
        :param gp_kernel: Name of the GP kernel, defaults to None
        :type gp_kernel: Optional[str], optional
        :param planet_names: Name of the planets, usually a single letter.
                             By default, the model uses lowercase letters
                             starting with b.
        :type planet_names: Optional[Union[str, list[str]]] , optional
        :param quiet_celerite: Silence celerite linear algebra errors,
                               defaults to False
        :type quiet_celerite: bool, optional
        :param name: PyMC3 model name that will prefix all variables,
                     defaults to ""
        :type name: str, optional
        :param model: Parent PyMC3 model, defaults to None
        :type model: Optional[Model], optional
        """
        super().__init__(name=name, model=model)

        # Set the data attributes
        self.t = np.array(t)
        self.vrad = np.array(vrad)
        self.svrad = np.array(svrad)

        pm.Data("vrad", self.vrad)
        pm.Data("svrad", self.svrad)

        # Set RV trend reference time
        if t_ref is None:
            self.t_ref = 0.5 * (self.t.min() + self.t.max())
        else:
            self.t_ref = t_ref

        try:
            self.num_planets = int(num_planets)
        except (ValueError, TypeError):
            raise TypeError("num_planets should be an integer.")

        if planet_names is not None:
            if isinstance(planet_names, str):
                planet_names = [planet_names]
            if len(planet_names) != num_planets:
                raise ValueError("Need one planet name per modelled planet")
            self.planet_names = planet_names
        else:
            self.planet_names = [chr(n + 98) for n in range(num_planets)]

        # If there are parameters, we load RV params first
        if params is not None:
            # Load parameters that affect the whole system.
            # NOTE: might not be hardcoded in future with multi-instrument
            if "rv" in params:
                load_params(params["rv"])
            self._rv_params = self._get_rv_params()
        # If no RV params, we set them to none
        else:
            self._rv_params = None

        # Always create pl models at init, but pass params only if available
        self.planets = dict()
        for prefix in self.planet_names:

            # NOTE: This means users can give params for one planet but skip
            # other and give its params later... Not sure we want this but
            # let's keep it for now
            if params is None:
                pl_params = None
            elif prefix in params:
                pl_params = params[prefix]
            else:
                pl_params = None

            self.planets[prefix] = PlanetModel(
                pl_params, name=prefix, model=self
            )

        # Always set to none, will be created on request
        self._synth_dict = None

        if len(self.planets) > 0:
            # Always set to none first, will be created on request
            # (planets might have their parameters defined later)
            self._orbit = None

        # Define data likelihood depending on GP type
        if gp_kernel is not None:
            if params is None:
                gp_params = None
            elif "gp" in params:
                gp_params = params["gp"]
            else:
                gp_params = None

            self.gpmodel = GPModel(gp_params, gp_kernel, name="gp", model=self)

            if gp_kernel in CELERITE_KERNELS:
                self.gp = GaussianProcess(
                    self.gpmodel.kernel,
                    t=self.t,
                    yerr=self.err,
                    quiet=quiet_celerite,
                )
            elif gp_kernel in PYMC3_KERNELS:
                self.gp = pm.gp.Marginal(cov_func=self.gpmodel.kernel)
            else:
                raise ValueError(
                    f"gp_kernel must be None, or one of {KERNEL_LIST}"
                )
        else:
            self.gpmodel = None
            self.gp = None

        if params is not None:
            # HACK: might not want this when use in global orbit model,
            # useful if wrapped by other model that handles likelihood
            if self.isroot:
                try:
                    self.add_likelihood()
                except (KeyError, AttributeError):
                    msg = (
                        "IMPORTANT: Tried to define 'obs' likelihood, "
                        "but some parameters are missing. Either add params "
                        "to the input dictionary or define them in the model "
                        "context. Then run model.add_likelihood().\n"
                    )
                warnings.warn(msg, RuntimeWarning)
        else:
            msg = (
                "No parameters. "
                " Don't forget to add them and call add_likelihood"
            )
            warnings.warn(
                msg,
                RuntimeWarning,
            )

    @property
    def rv_params(self):
        if self._rv_params is not None:
            return self._rv_params
        else:
            self._rv_params = self._get_rv_params()
            return self._rv_params

    @property
    def orbit(self):
        if self._orbit is not None:
            return self._orbit
        elif self._orbit is None and len(self.planets) > 0:
            self._orbit = self._get_orbit()
            return self._orbit
        else:
            return None

    @property
    def synth_dict(self):
        if self._synth_dict is not None:
            return self._synth_dict
        else:
            self._synth_dict = self._get_synth_dict()
            return self._synth_dict

    @property
    def rvmod(self):
        if "rv_model" not in self.named_vars:
            with self:
                self.calc_rv_model(self.t)
        return self.rv_model

    @property
    def resid(self):
        return self.vrad - self.rvmod

    @property
    def err(self):
        return tt.sqrt(self.svrad ** 2 + self.wn ** 2)

    def _get_synth_dict(self):

        # exoplanet orbit objects require a certain parmeterization.
        # We build vectors over planets in this "synth" parameterization
        # (used to synthesize RVs, following RadVel nomenclature).

        synth_dict = dict()
        for pname in SYNTH_PARAMS:
            synth_dict[pname] = pm.Deterministic(
                pname,
                tt.as_tensor_variable(
                    [pl.synth_params[pname] for pl in self.planets.values()]
                ),
            )

        return synth_dict

    def add_likelihood(self):
        if self.gpmodel is None:
            pm.Normal("obs", mu=self.rvmod, sd=self.err, observed=self.vrad)
        else:
            if self.gpmodel.kernel in CELERITE_KERNELS:
                self.gp.marginal("obs", observed=self.resid)
            elif self.gpmodel.kernel in PYMC3_KERNELS:
                self.gp.marginal_likelihood(
                    "obs", self.t[:, None], self.resid, noise=self.err
                )
            else:
                raise ValueError(
                    f"gp_kernel must be None, or one of {KERNEL_LIST}"
                )

    def _get_orbit(self):

        # NOTE: will move to more generic model to support all timeseries.
        # We access keys explicitely because the naming convention is
        # different between RadVel and exoplanet.
        # To use **, would need to "translate" radvel names to exoplanet or
        # just use exoplanet names by default.

        if len(self.planets) > 0:
            return xo.orbits.KeplerianOrbit(
                period=self.synth_dict["per"],
                t_periastron=self.synth_dict["tp"],
                ecc=self.synth_dict["e"],
                omega=self.synth_dict["w"],
            )
        else:
            return None

    def _get_rv_params(self):
        """
        Read RV-specific and create determinstic relations to include them in
        RV calculation
        """
        nvars = self.named_vars
        rv_params = dict()
        prefix = self.prefix

        if f"{prefix}gamma" not in nvars:
            rv_params["gamma"] = pm.Deterministic(
                "gamma", tt.as_tensor_variable(0.0)
            )
        else:
            rv_params["gamma"] = nvars[f"{prefix}gamma"]

        if f"{prefix}dvdt" not in nvars:
            rv_params["dvdt"] = pm.Deterministic(
                "dvdt", tt.as_tensor_variable(0.0)
            )
        else:
            rv_params["dvdt"] = nvars[f"{prefix}dvdt"]

        if f"{prefix}curv" not in nvars:
            rv_params["curv"] = pm.Deterministic(
                "curv", tt.as_tensor_variable(0.0)
            )
        else:
            rv_params["curv"] = nvars[f"{prefix}curv"]

        if f"{prefix}wn" in nvars:
            rv_params["wn"] = nvars[f"{prefix}wn"]
        elif f"{prefix}logwn" in nvars:
            rv_params["wn"] = pm.Deterministic(
                "wn", tt.exp(nvars[f"{prefix}logwn"])
            )
        else:
            rv_params["wn"] = pm.Deterministic(
                "wn", tt.as_tensor_variable(0.0)
            )

        return rv_params

    def calc_rv_model(self, t: np.ndarray, name: str = "") -> pm.Distribution:
        """
        Get the RV signal for the system.

        :param t: Time values where the signal is calculated
        :type t: np.ndarray
        :param name: Name of the RV prediction (name of the variable will be
                     rv_model{_name}
        :type name: str, optional
        :return: PyMC3 determinsitic variable rkepresenting the RV signal
        :rtype: pm.Distribution
        """

        suffix = "" if name == "" else f"_{name}"

        # Define the background RV model (constant at 0 if not provided)
        rvpars = self.rv_params
        t_shift = t - self.t_ref
        bkg = rvpars["gamma"]
        bkg += rvpars["dvdt"] * t_shift
        bkg += rvpars["curv"] * t_shift ** 2

        if len(self.planets) > 0:
            # Calculate RV signal for each planet's orbit (2D array)
            rvorb = self.orbit.get_radial_velocity(t, K=self.synth_dict["k"])
            pm.Deterministic("rv_orbits" + suffix, rvorb)

            # Sum over planets and add the background to get the full model
            rv_model = tt.sum(rvorb, axis=-1) + bkg
        else:
            rv_model = bkg

        # NOTE: This function may be called in unsuccseful add_likelihood
        # So we don't set bkg and rv_model before the end to avoid model errors
        pm.Deterministic("bkg" + suffix, bkg)
        return pm.Deterministic("rv_model" + suffix, rv_model)
