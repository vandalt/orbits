from typing import Optional

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
        params: dict[str, dict],
        num_planets: int,
        t_ref: Optional[float] = None,
        gp_kernel: Optional[str] = None,
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
                       to avoid ambiguities in the future)
        :type params: dict[str, dict]
        :param num_planets: Number of planets to model in the dataset.
        :type num_planets: int
        :param t_ref: Reference time for RV trend parameters, defaults to None
        :type t_ref: Optional[float], optional
        :param gp_kernel: Name of the GP kernel, defaults to None
        :type gp_kernel: Optional[str], optional
        :param quiet_celerite: [TODO:description], defaults to False
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

        # Load parameters that affect the whole system.
        load_params(params["system"])

        self.planets = dict()
        for prefix in params:

            # NOTE: In the future it might be better to replace this by
            # iterating over planets explicitely. Two special cases is
            # easy to manage for now.
            if prefix in ["system", "gp"]:
                # We skip non-planet entries in the parameter dictionary
                continue

            # For each planet, we load parameters
            # Using submodel makes "{letter}_" prefix auto
            self.planets[prefix] = PlanetModel(
                params[prefix], name=prefix, model=self
            )

        # We also make sure that parameters specific to the RV model are
        # available (e.g. reparamtrezie log{param})
        self._get_rv_params()

        if len(self.planets) > 0:
            # NOTE: will move to more generic model to support all timeseries.
            # We access keys explicitely because the naming convention is
            # different between RadVel and exoplanet.
            # To use **, would need to "translate" radvel names to exoplanet or
            # just use exoplanet names by default.

            # exoplanet orbit objects require a certain parmeterization.
            # We built vectors over planets in this "synth" parameterization
            # (used to synthesize RVs, following RadVel nomenclature).
            self._get_synth_params()

            self.orbit = xo.orbits.KeplerianOrbit(
                period=self.synth_dict["per"],
                t_periastron=self.synth_dict["tp"],
                ecc=self.synth_dict["e"],
                omega=self.synth_dict["w"],
            )

        # Once we have our parameters, we define the RV model at data points
        self.get_rv_model(self.t)
        self.resid = self.vrad - self.rv_model

        # Then, with the rv_model, we define a likelihood with the RV data
        self.err = tt.sqrt(self.svrad ** 2 + self.wn ** 2)

        # Define data likelihood depending on GP type
        if gp_kernel is None:
            pm.Normal("obs", mu=self.rv_model, sd=self.err, observed=self.vrad)
        else:
            self.gpmodel = GPModel(
                params["gp"], gp_kernel, name="gp", model=self
            )
            if gp_kernel in CELERITE_KERNELS:
                self.gp = GaussianProcess(
                    self.gpmodel.kernel,
                    t=self.t,
                    yerr=self.err,
                    quiet=quiet_celerite,
                )
                self.gp.marginal("obs", observed=self.resid)
            elif gp_kernel in PYMC3_KERNELS:
                self.gp = pm.gp.Marginal(cov_func=self.gpmodel.kernel)
                self.gp.marginal_likelihood(
                    "obs", self.t[:, None], self.resid, noise=self.err
                )
            else:
                raise ValueError(
                    f"gp_kernel must be None, or one of {KERNEL_LIST}"
                )

    def _get_synth_params(self):
        """
        Create vectors of parameters in "synth" basis to feed exoplanet orbit
        model.
        Stored as synth_dict attribute.
        """
        # NOTE: This will be good for all types of orbit models, not only RVs
        synth_dict = dict()
        nvars = self.named_vars
        for pname in SYNTH_PARAMS:
            synth_dict[pname] = pm.Deterministic(
                pname,
                tt.as_tensor_variable(
                    [nvars[f"{prefix}_{pname}"] for prefix in self.planets]
                ),
            )

        self.synth_dict = synth_dict

    def _get_rv_params(self):
        """
        Read RV-specific and create determinstic relations to include them in
        RV calculation
        """
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

    def get_rv_model(self, t: np.ndarray, name: str = "") -> pm.Distribution:
        """
        Get the RV signal for the system.

        :param t: Time values where the signal is calculated
        :type t: np.ndarray
        :param name: Name of the RV prediction (name of the variable will be
                     rv_model{_name}
        :type name: str, optional
        :return: PyMC3 determinsitic variable representing the RV signal
        :rtype: pm.Distribution
        """

        suffix = "" if name == "" else f"_{name}"

        # Define the background RV model (constant at 0 if not provided)
        t_shift = t - self.t_ref
        bkg = self.gamma
        bkg += self.dvdt * t_shift
        bkg += self.curv * t_shift ** 2
        bkg = pm.Deterministic("bkg" + suffix, bkg)

        if len(self.planets) > 0:
            # Calculate RV signal for each planet's orbit (2D array)
            rvorb = self.orbit.get_radial_velocity(t, K=self.synth_dict["k"])
            pm.Deterministic("rv_orbits" + suffix, rvorb)

            # Sum over planets and add the background to get the full model
            return pm.Deterministic(
                "rv_model" + suffix, tt.sum(rvorb, axis=-1) + bkg
            )
        else:
            return pm.Deterministic(
                "rv_mode" + suffix, bkg
            )
