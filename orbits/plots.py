import numpy as np
import matplotlib.pyplot as plt
from xarray.core.dataset import Dataset


def rvplot(
    t,
    vrad,
    svrad,
    t_soln=None,
    soln=None,
    soln_name="",
    title=None,
    planet_names=None,
    path=None
):

    plt.errorbar(t, vrad, yerr=svrad, fmt=".k")

    if t_soln is not None and soln is not None:
        # MAP optimiation
        suffix = "" if soln_name == "" else f"_{soln_name}"
        if isinstance(soln, dict):
            if planet_names is None:
                planet_names = [
                    chr(n + 98) for n in range(soln[f"rv_orbits{suffix}"].shape[1])
                ]
            plt.plot(
                t_soln,
                soln[f"rv_orbits{suffix}"],
                "--",
                alpha=0.8,
                label=planet_names,
            )
            plt.plot(t_soln, soln[f"bkg{suffix}"], ":k", alpha=0.5, label="Trend")
            plt.plot(t_soln, soln[f"rv_model{suffix}"], "blue", label="Model")
        elif isinstance(soln, Dataset):

            if planet_names is None:
                planet_names = [
                    chr(n + 98) for n in range(soln[f"rv_orbits{suffix}"].shape[-1])
                ]
            keys = ["rv_model", "bkg", "rv_orbits"]
            markers = ["blue", ":k", "--"]
            colors = ["blue", "black", ["peachpuff", "green"]]
            labels = ["Model", "Trend", planet_names]
            alphas = [0.3, 0.1, 0.1]
            for i, key in enumerate(keys):
                samples = soln[f"{key}{suffix}"].values

                pred = np.percentile(samples, [16, 50, 84], axis=(0, 1))

                plt.plot(t_soln, pred[1], markers[i], label=labels[i])
                shape = samples.shape
                if len(shape) == 4:
                    for j in range(shape[-1]):
                        plt.fill_between(
                            t_soln,
                            pred[0, :, j],
                            pred[2, :, j],
                            color=colors[i][j],
                            alpha=alphas[i],
                        )
                else:
                    plt.fill_between(t_soln, pred[0], pred[2], color=colors[i], alpha=alphas[i])


        plt.xlim(t_soln.min(), t_soln.max())

    plt.legend(fontsize=10)
    plt.xlabel("Time [days]")
    plt.ylabel("RV [m/s]")
    if title is not None:
        plt.plot(title)
    if path is not None:
        plt.savefig(path)
