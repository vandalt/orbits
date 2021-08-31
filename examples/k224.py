# %% [markdown]
# # Radial velocity fitting
# In this example, we will show how to fit radial velocity data using the _orbits_ Python API.
#
# This example replicates [the case study](https://gallery.exoplanet.codes/tutorials/rv/)
# from _[exoplanet](https://docs.exoplanet.codes/en/latest/)_ (which this package is built upon).
# The _exoplanet_ case study follows [a tutorial](https://radvel.readthedocs.io/en/latest/tutorials/K2-24_Fitting+MCMC.html)
# from [RadVel](https://radvel.readthedocs.io/en/latest/index.html).
# A lot of the nomenclature and some design ideas in _orbits_ are borrowed from RadVel.

# %%
import matplotlib.pyplot as plt
import numpy as np
import orbits.utils as ut
import pandas as pd
from orbits.model import RVModel

# %% [markdown]
# First, we can download the data from RadVel and plot it. We also make a finer
# grid that will be used to plot the model.

# %%
# Download and unpack the data
url = "https://raw.githubusercontent.com/California-Planet-Search/radvel/master/example_data/epic203771098.csv"
data = pd.read_csv(url, index_col=0)
t = np.array(data.t)
vrad = np.array(data.vel)
svrad = np.array(data.errvel)

# Reference time for RV trends later
x_ref = 0.5 * (t.min() + t.max())

# Fine grid for model plots
t_pred = np.linspace(t.min() - 5, t.max() + 5, 1000)

# Plot the data
plt.errorbar(data.t, data.vel, data.errvel, fmt="k.", capsize=2)
plt.xlabel("Time [days]")
plt.ylabel("RV [m/s]")
plt.show()

# %% [markdown]
# Next, we use literature values for the periods and transit times.
# Then we can use _exoplanet_ to estimate the RV semi-amplitude.

# %%
import exoplanet as xo

periods = [20.8851, 42.3633]
period_errs = [0.0003, 0.0006]
tc = [2072.7948, 2082.6251]
tc_errs = [0.0007, 0.0004]
k = xo.estimate_semi_amplitude(periods, t, vrad, svrad, t0s=tc)
print("Semi-amplitude estimate:", k, "m/s")


# %% [markdown]
# _exoplanet_ (and _orbits_) uses the PyMC3 modelling framework. PyMC3 models
# are a bit different than the Python models/functions we are used to. However,
# it has several useful features, including Hamiltonian-Monte Carlo as well as
# several pre-defined distributions that can be used as priors.
# The _exoplanet_ documentation has [a nice introduction to PyMC3](https://docs.exoplanet.codes/en/latest/tutorials/intro-to-pymc3/).
# The [PyMC3 documentation](https://docs.pymc.io/) is also a good resource to
# learn more about PyMC3 and probabilistic programming. The main thing to note
# is that parameters are defined directly as prior distributions.
#
# To define models in PyMC3, we need to be in a model context (using something
# like `with Model():`). However, if we do this, we need to setup relations
# between fitted parameters and parameters of interest manually. The same
# goes for a GP model, the RV signal, and everything that the model contains.
# This is where _orbits_ comes in. With pre-defined model such as `RVModel`,
# we can simply pass fitted parameters to the model and everything else
# (reparametrizations, GP kernels, orbit solver) is set up automatically.
#
# Because _orbits_ was first designed only to be a CLI tool that uses a YAML
# config file, parameters can currently only be passed at initialization
# inside a dictionary. An example dictionary is given below (a YAML version of
# this dictionary is available in `k224.yml`.

# %%
import pymc3_ext as pmx

params = {
    # This entry is the first planet with its orbital parameters.
    "b": {
        "logper": {
            "dist": "Normal",
            "kwargs": {"mu": np.log(20.8851), "sd": 0.0003 / 20.8851},
        },
        "tc": {
            "dist": "Normal",
            "kwargs": {"mu": 2072.7948, "sd": 0.0007},
        },
        "logk": {
            "dist": "Normal",
            "kwargs": {
                "mu": np.log(k[0]),
                "sd": 2.0,
                "testval": np.log(k[0]),
            },
        },
        # This parameter is a list with the sqrt(e)*cos(w) and sqrt(e)*sin(w)
        # exoplanet defines the UnitDisk prior which can bring performance
        # improvement and forces the eccentricity to be < 1.
        "secsw": {
            "dist": "UnitDisk",
            "kwargs": {"shape": 2, "testval": 0.01 * np.ones(2)},
        },
    },
    # This entry is the second planet with its orbital parameters.
    "c": {
        "logper": {
            "dist": "Normal",
            "kwargs": {"mu": np.log(42.3633), "sd": 0.0006 / 42.3633},
        },
        "tc": {
            "dist": "Normal",
            "kwargs": {"mu": 2082.6251, "sd": 0.0004},
        },
        "logk": {
            "dist": "Normal",
            "kwargs": {
                "mu": np.log(k[1]),
                "sd": 2.0,
                "testval": np.log(k[1]),
            },
        },
        "secsw": {
            "dist": "UnitDisk",
            "kwargs": {"shape": 2, "testval": 0.01 * np.ones(2)},
        },
    },
    # Some parameters affect the whole rv dataset, not just one planet.
    "rv": {
        "logwn": {
            "dist": "DataNormal",
            "kwargs": {"data_used": "svrad", "sd": 5.0, "apply": "log"},
        },
        "gamma": {
            "dist": "Normal",
            "kwargs": {"mu": 0.0, "sd": 1.0},
        },
        "dvdt": {
            "dist": "Normal",
            "kwargs": {"mu": 0.0, "sd": 0.1},
        },
        "curv": {
            "dist": "Normal",
            "kwargs": {"mu": 0.0, "sd": 0.01},
        },
    },
}

# %% [markdown]
# Now that our parameters are defined, we can create a PyMC3 model.
# Note that we can stil modify the model in the model context, as shown below
# with an additional eccentricity prior and an RV predicitive curve.

# %%
with RVModel(t, vrad, svrad, 2, params=params) as model:

    xo.eccentricity.vaneylen19(
        "ecc_prior",
        multi=True,
        shape=2,
        fixed=True,
        observed=model.synth_dict["e"],
    )

    rv_model_pred = model.calc_rv_model(t_pred, name="pred")

    print(model.named_vars)


# %% [markdown]
# However, forcing users to define a big nested dictionary in a notebook or
# script is not great for readability. It also breaks the original PyMC3
# workflow of defining parameters inside the model context. For this reason,
# `orbits` models also support defining parameters in the model conetxt, **with
# a few (IMPORTANT) limitation**:
# 1. Submodels (e.g. planets) are defined at initialization with no parameters.
#    To define their parameters, you need to use their context (see below).
# 2. The chi2 or GP likelihood for the data is never called explicitely,
#    it is only included in the total posterior if defined. I have not found a
#    good way to ensure it is created before sampling or optimization, so for
#    now, users must call `add_likelihood` after creating the model manually
#    (this is not required when passing a full parameter dictionary: the model
#    is then able to create it by itself at initialization).
#    There is a warning when this no likelihood is added

# %%
import pymc3 as pm
import pymc3_ext as pmx
from orbits.prior import data_normal_prior

with RVModel(t, vrad, svrad, 2) as model:

    print(model.named_vars)

    data_normal_prior("logwn", data_used="svrad", sd=5.0, apply="log")
    pm.Normal("gamma", mu=0.0, sd=1.0)
    pm.Normal("dvdt", mu=0.0, sd=0.1)
    pm.Normal("curv", mu=0.0, sd=0.01)

    # To add planet parameters, must be in their automatically created submodel
    with model.planets["b"]:
        pm.Normal("logper", mu=np.log(20.8851), sd=0.0003 / 20.8851)
        pm.Normal("tc", mu=2072.7948, sd=0.0007)
        pm.Normal("logk", mu=np.log(k[0]), sd=2.0, testval=np.log(k[0]))
        pmx.UnitDisk("secsw", shape=2, testval=0.01 * np.ones(2))
    with model.planets["c"]:
        pm.Normal("logper", mu=np.log(42.3633), sd=0.0006 / 42.3633)
        pm.Normal("tc", mu=2082.6251, sd=0.0004)
        pm.Normal("logk", mu=np.log(k[1]), sd=2.0, testval=np.log(k[1]))
        pmx.UnitDisk("secsw", shape=2, testval=0.01 * np.ones(2))

    xo.eccentricity.vaneylen19(
        "ecc_prior",
        multi=True,
        shape=2,
        fixed=True,
        observed=model.synth_dict["e"],
    )

    rv_model_pred = model.calc_rv_model(t_pred, name="pred")


    model.add_likelihood()

# %% [markdown]
# Now that we have a model, we can plot its prediction. But we only defined
# priors, so how do we evaluate the model ? PyMC3 priors have a `testval` that
# can be used to plot the model. The _pymc3-ext_ package, from the _exoplanet_
# developers, has a built-in `eval_in_model` function to do just this.

# %%
plt.errorbar(t, vrad, yerr=svrad, fmt=".k")

with model:
    plt.plot(t_pred, pmx.eval_in_model(model.rv_orbits_pred), "--k", alpha=0.5)
    plt.plot(t_pred, pmx.eval_in_model(model.bkg_pred), ":k", alpha=0.5)
    plt.plot(t_pred, pmx.eval_in_model(model.rv_model_pred), label="model")

plt.legend(fontsize=10)
plt.xlim(t_pred.min(), t_pred.max())
plt.xlabel("time [days]")
plt.ylabel("radial velocity [m/s]")
plt.title("initial model")
plt.show()

# %% [markdown]
# Our initial test values don't look too good, so we will find the
# maximum a posteriori (MAP) solution using `pymc3`/`pymc3-ext`.
# Sometimes it can help to optimize parameters sequentially, as we do below.

# %%
with model:
    # Optimize the offset and trend parameters only
    map_soln = pmx.optimize(
        start=model.test_point, vars=[model.gamma, model.dvdt, model.curv]
    )
    opt2_list = [model.gamma, model.dvdt, model.curv, model.logwn]

    # Now optimize some planet parameters as well, using previous solution as
    # starting point.
    for prefix in model.planets:
        opt2_list.extend(
            [
                model[f"{prefix}_logk"],
                model[f"{prefix}_tc"],
                model[f"{prefix}_logper"],
            ]
        )
    map_soln = pmx.optimize(start=map_soln, vars=opt2_list)

    # Optimize eccentricity parameters
    map_soln = pmx.optimize(
        start=map_soln,
        vars=[model[f"{prefix}_secsw"] for prefix in model.planets],
    )

    # Optimize everything
    map_soln = pmx.optimize(start=map_soln)


# %% [markdown]
# Let's now plot the MAP solution.

# %%
from orbits import plots


plots.rvplot(t, vrad, svrad, t_soln=t_pred, soln=map_soln, soln_name="pred")
plt.show()

# %% [markdown]
# We can now sample our model posterior to get a better estimate of our
# parameters and their uncertainty. We use _pymc3-ext_ as it wraps the PyMC3
# sampler with more appropriate defaults and tuning strategies (this is taken
# directly from the _exoplanet_ RV tutorial).

# %%
np.random.seed(42)
with model:
    trace = pmx.sample(
        tune=1000,
        draws=1000,
        cores=2,
        chains=2,
        target_accept=0.9,
        return_inferencedata=True,
    )


# %% [markdown]
# We now have the posterior distribution and the HMC chains stored in `trace`.
# We can use [ArviZ](https://arviz-devs.github.io/arviz/examples/index.html) to
# visualize the dataset.

# %%
import arviz as az

az.summary(
    trace,
    var_names=[
        "gamma",
        "dvdt",
        "curv",
        "logwn",
        "w",
        "e",
        "b_tc",
        "c_tc",
        "k",
        "per",
    ],
)

# %% [markdown]
# The [_corner_](https://corner.readthedocs.io/en/latest/) is now compatible
# with ArviZ objects, so we can use corner to view the corner plot of our
# posterior distribution.

# %%
import corner

# with model:
corner.corner(trace, var_names=["per", "k", "e", "w"])
plt.show()

# %% [markdown]
# Finally, we plot the full model and the phase-folded planet orbits

# %%
for n, letter in enumerate("bc"):
    plt.figure()

    # Get the posterior median orbital parameters
    p = np.median(trace.posterior["per"].values[:, :, n])
    tp = trace.posterior["tp"].values[:, :, n]
    per = trace.posterior["per"].values[:, :, n]
    ecc = trace.posterior["e"].values[:, :, n]
    omega = trace.posterior["w"].values[:, :, n]
    t0 = np.median(ut.timeperi_to_timetrans(tp, per, ecc, omega, use_np=True))
    # t0 = np.median(trace.posterior["t0"].values[:, :, n])

    # Compute the median of posterior estimate of the background RV
    # and the contribution from the other planet. Then we can remove
    # this from the data to plot just the planet we care about.
    other = np.median(
        trace.posterior["rv_orbits"].values[:, :, :, (n + 1) % 2], axis=(0, 1)
    )
    other += np.median(trace.posterior["bkg"].values, axis=(0, 1))

    # Plot the folded data
    x_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
    plt.errorbar(x_fold, vrad - other, yerr=svrad, fmt=".k")

    # Compute the posterior prediction for the folded RV model for this
    # planet
    t_fold = (t_pred - t0 + 0.5 * p) % p - 0.5 * p
    inds = np.argsort(t_fold)
    pred = np.percentile(
        trace.posterior["rv_orbits_pred"].values[:, :, inds, n],
        [16, 50, 84],
        axis=(0, 1),
    )
    plt.plot(t_fold[inds], pred[1], color="C0", label="model")
    art = plt.fill_between(
        t_fold[inds], pred[0], pred[2], color="C0", alpha=0.3
    )
    art.set_edgecolor("none")

    plt.legend(fontsize=10)
    plt.xlim(-0.5 * p, 0.5 * p)
    plt.xlabel("phase [days]")
    plt.ylabel("radial velocity [m/s]")
    plt.title("K2-24 {0}".format(letter))

    plt.show()
