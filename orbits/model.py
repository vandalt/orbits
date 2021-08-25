from typing import Optional

import aesara_theano_fallback.tensor as tt
import exoplanet as xo
import numpy as np
import pymc3 as pm
from celerite2.theano import GaussianProcess
from pymc3 import Model

import orbits.utils as ut
from orbits.kernel import (CELERITE_KERNELS, KERNEL_LIST, KERNEL_PARAMS,
                           KERNELS, PYMC3_KERNELS)
from orbits.prior import load_params

SYNTH_PARAMS = ["per", "tp", "e", "w", "k"]


class PlanetModel(Model):
    def __init__(
        self,
        params: dict[str, dict],
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

        K_PARAMS = [f"{prefix}k", f"{prefix}logk"]
        if f"{prefix}k" in nvars:
            pass
        elif f"{prefix}logk" in nvars:
            pm.Deterministic("k", tt.exp(nvars[f"{prefix}logk"]))
        else:
            raise KeyError(f"Should have one of: {K_PARAMS}")


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
        :type params: Optional[dict[str, dict]]
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
            self._kernel = KERNELS[self.kernel_name](**self.gpdict)

    @property
    def kernel(self):
        # If the kernel is not defined yet, we define it and return to the user
        if self._kernel is not None:
            return self._kernel
        else:
            self._kernel = KERNELS[self.kernel_name](**self.gpdict)
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

        gpdict = dict()
        nvars = self.gpmodel.named_vars
        prefix = f"{self.name}_" if self.name != "" else ""
        for pname in KERNEL_PARAMS[self.kernel_name]:
            if f"{prefix}{pname}" in nvars:
                gpdict[pname] = nvars[f"{prefix}{pname}"]
            elif f"{prefix}log{pname}" in nvars:
                gpdict[pname] = self.pm.Deterministic(
                    pname, tt.exp(nvars[f"{prefix}log{pname}"])
                )
            else:
                raise ValueError(
                    f"Kernel {self.kernel_name} requires parameter "
                    f"{pname} or log{pname}"
                )

        return gpdict


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
        :param gp_kernel: [TODO:description], defaults to None
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

        # exoplanet orbit objects require a certain parmeterization.
        # We built vectors over planets in this "synth" parameterization
        # (used to synthesize RVs, following RadVel nomenclature).
        self._get_synth_params()

        # We also make sure that parameters specific to the RV model are
        # available (e.g. reparamtrezie log{param})
        self._get_rv_params()

        # NOTE: will move to more generic model to support other timeseries.
        # We access keys explicitely because the naming convention is different
        # between RadVel and exoplanet
        # To use **, would need to "translate" radvel names to exoplanet or
        # just use exoplanet names by default.
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

        # Calculate RV signal for each planet's orbit (2D array)
        rvorb = self.orbit.get_radial_velocity(t, K=self.synth_dict["k"])
        suffix = "" if name == "" else f"_{name}"
        pm.Deterministic("rv_orbits" + suffix, rvorb)

        # Define the background RV model (constant at 0 if not provided)
        t_shift = t - self.t_ref
        bkg = self.gamma
        bkg += self.dvdt * t_shift
        bkg += self.curv * t_shift ** 2
        bkg = pm.Deterministic("bkg" + suffix, bkg)

        # Sum over planets and add the background to get the full model
        return pm.Deterministic(
            "rv_model" + suffix, tt.sum(rvorb, axis=-1) + bkg
        )
