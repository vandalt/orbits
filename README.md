# orbits

**WARNING: This is still in early development. See the [Initial release milestone](https://github.com/vandalt/orbits/milestone/1) section for
details.**

Currently, only the RVModel is available. _orbits_ also provides functions to
load PyMC3 parameters and define GP kernel.

_orbits_ is a wrapper around
[exoplanet](https://github.com/exoplanet-dev/exoplanet) aimed at providing
pre-defined orbit models that can be quickly setup from the API or from a
configuration file. The goal is to have something similar to
[RadVel](https://github.com/California-Planet-Search/radvel), but using
`exoplanet` and PyMC3 as a backend.

The 0.1 release supports creating an RVModel with a parameter dictionary,
defining GP kernels, and using the RVModel object for inference.

## Installation

_orbits_ can be installed with pip: `python -m pip install orbits`.

To use the development version of _orbits_, clone the repository and install it:
```shell
git clone https://github.com/vandalt/orbits.git
python -m pip install -U -e ".[dev]"
```

## Using _orbits_
There is no detailed documentation yet, but examples will be added to the
`examples` directory. There is currently an example with the K2-24 dataset.
This replicates tutorials from _exoplanet_ and _RadVel_.
