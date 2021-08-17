# TODO: Add links to packages here

# %% [markdown]
# # Radial velocity fitting
# This example replicates the case study from exoplanet, which follows a tutorial from RadVel.

# %%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orbits.model import RVModel
import orbits.utils as ut

# %%
url = "https://raw.githubusercontent.com/California-Planet-Search/radvel/master/example_data/epic203771098.csv"
data = pd.read_csv(url, index_col=0)

x = np.array(data.t)
y = np.array(data.vel)
yerr = np.array(data.errvel)

x_ref = 0.5 * (x.min() + x.max())

# Also make a fine grid that spans the observation window for plotting purposes
t = np.linspace(x.min() - 5, x.max() + 5, 1000)
plt.errorbar(data.t, data.vel, data.errvel, fmt="k.", capsize=2)
plt.xlabel("Time [days]")
plt.ylabel("RV [m/s]")
plt.show()

# %%
import exoplanet as xo

periods = [20.8851, 42.3633]
period_errs = [0.0003, 0.0006]
t0s = [2072.7948, 2082.6251]
t0_errs = [0.0007, 0.0004]
Ks = xo.estimate_semi_amplitude(periods, x, y, yerr, t0s=t0s)
print(Ks, "m/s")


# %%
# Import pymc3 things
import pymc3_ext as pmx

# %% [markdown]
# Now, we define parameters as priors. The format is ugly here, but
# we need to be inside the pymc3 model to define actual priors. So in order
# to be compatible with config files, we get a dictionary like this.
#
# **NOTE: in the future, there might be a `Prior` class that is then able
# to "translate" priors to pymc3. This should alleviate the parameter definition.**

# %%
params = {
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
                "mu": np.log(Ks[0]),
                "sd": 2.0,
                "testval": np.log(Ks[0]),
            },
        },
        "secsw": {
            "dist": "UnitDisk",
            "kwargs": {"shape": 2, "testval": 0.01 * np.ones(2)},
        },
    },
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
                "mu": np.log(Ks[1]),
                "sd": 2.0,
                "testval": np.log(Ks[1]),
            },
        },
        "secsw": {
            "dist": "UnitDisk",
            "kwargs": {"shape": 2, "testval": 0.01 * np.ones(2)},
        },
    },
    "system": {
        "logwn": {
            "dist": "DataNormal",
            "kwargs": {"data_used": np.log(yerr), "sd": 5.0},
        },
        "trend": {
            "dist": "Normal",
            "kwargs": {
                "mu": 0.0,
                "sd": 10.0 ** -np.arange(3)[::-1],
                "shape": 3,
            },
        },
    },
}

# %%
with RVModel(x, y, yerr, params, 2) as model:

    # TODO: Not sure how will handle these yet
    # Eccentricity & argument of periasteron
    # ecs = pmx.UnitDisk("ecs", shape=(2, 2), testval=0.01 * np.ones((2, 2)))
    # ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
    # omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
    xo.eccentricity.vaneylen19(
        "ecc_prior", multi=True, shape=2, fixed=True, observed=model.e
    )

    rv_model_pred = model.get_rv_model(t, name="_pred")

    print(model.named_vars)

# %%
plt.errorbar(x, y, yerr=yerr, fmt=".k")

with model:
    plt.plot(t, pmx.eval_in_model(model.rv_orbits_pred), "--k", alpha=0.5)
    plt.plot(t, pmx.eval_in_model(model.bkg_pred), ":k", alpha=0.5)
    plt.plot(t, pmx.eval_in_model(model.rv_model_pred), label="model")

plt.legend(fontsize=10)
plt.xlim(t.min(), t.max())
plt.xlabel("time [days]")
plt.ylabel("radial velocity [m/s]")
plt.title("initial model")
plt.show()

# %%
with model:
    map_soln = pmx.optimize(start=model.test_point, vars=[model.trend])
    opt2_list = [model.trend, model.logwn]
    for prefix in model.submodels:
        opt2_list.extend(
            [
                model[f"{prefix}_logk"],
                model[f"{prefix}_tc"],
                model[f"{prefix}_logper"],
            ]
        )
    map_soln = pmx.optimize(start=map_soln, vars=opt2_list)
    map_soln = pmx.optimize(
        start=map_soln,
        vars=[model[f"{prefix}_secsw"] for prefix in model.submodels],
    )
    map_soln = pmx.optimize(start=map_soln)

# %%
plt.errorbar(x, y, yerr=yerr, fmt=".k")
plt.plot(t, map_soln["rv_orbits_pred"], "--k", alpha=0.5)
plt.plot(t, map_soln["bkg_pred"], ":k", alpha=0.5)
plt.plot(t, map_soln["rv_model_pred"], label="model")

plt.legend(fontsize=10)
plt.xlim(t.min(), t.max())
plt.xlabel("time [days]")
plt.ylabel("radial velocity [m/s]")
plt.title("MAP model")
plt.show()

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


# %%
# Import arviz for data visualization
import arviz as az

az.summary(
    trace, var_names=["trend", "logwn", "w", "e", "tp", "k", "per"]
)

# %%
# We still use corner for corner plots
import corner

with model:
    corner.corner(trace, var_names=["per", "k", "e", "w"])
    plt.show()

# %%
plt.errorbar(x, y, yerr=yerr, fmt=".k")

# Compute the posterior predictions for the RV model
rv_pred = trace.posterior["rv_model_pred"].values
pred = np.percentile(rv_pred, [16, 50, 84], axis=(0, 1))
plt.plot(t, pred[1], color="C0", label="model")
art = plt.fill_between(t, pred[0], pred[2], color="C0", alpha=0.3)
art.set_edgecolor("none")

plt.legend(fontsize=10)
plt.xlim(t.min(), t.max())
plt.xlabel("time [days]")
plt.ylabel("radial velocity [m/s]")
plt.title("posterior constraints")
plt.show()


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
    x_fold = (x - t0 + 0.5 * p) % p - 0.5 * p
    plt.errorbar(x_fold, y - other, yerr=yerr, fmt=".k")

    # Compute the posterior prediction for the folded RV model for this
    # planet
    t_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
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
