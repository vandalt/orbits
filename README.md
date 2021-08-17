# orbits

**WARNING: This is still in early development. See the [Plans](##Plans) section for
details.**

_orbits_ is a wrapper around [exoplanet](https://github.com/exoplanet-dev/exoplanet) aimed at providing pre-defined
orbit models that can be quickly setup from the API or from a configuration
file. The goal is to have something similar to
[RadVel](https://github.com/California-Planet-Search/radvel), but using
`exoplanet` and PyMC3 as a backend.

## Using _orbits_
To use orbit, clone the repository and install it with `python -m pip install
.` (from the project's directory). To install development dependencies (required
to run the examples), use `".[dev]"` instead of `.`. The project will
(hopefully) be on PyPI soon, once the RV-only modelling is stable.

There is no detailed documentation yet, but examples will be added to the
`examples` directory.

## Plans
* [ ] Implement RVModel that supports mutliple planets in a single system from
  one instrument.
* [ ] Add GP activity modelling for RV
* [ ] Add RV plots
  * [ ] Full timeseries (with "decomposed" signals)
  * [ ] Phase-folded
* [ ] Add sampling plots
  * [ ] Trace plots
  * [ ] Corner plot
* [ ] Support multiple instruments.
* [ ] Implement an `OrbitModel` that wraps all the building blocks from
  `exoplanet` (all types of timeseries).
  * [ ] Transits (high priority)
  * [ ] Astrometry
  * [ ] TTVs
* [ ] Impelement a parameter interface similar to RadVel's `Parameter` class
