# numba reduced neuoronal models
Single neuronal models encoded in python language numba speed up.
Note model waveforms are returned in Neo Format, converting between array to neo might introduce a sometimes tolerable overhead.

Models:
* Adaptive Exponential.
* Izhikevich 2007 (not 2003).
* Multi Time Scale Adaptive Neuron

To install.
```
git clone -b dev https://github.com/russelljjarvis/jit_hub
sudo pip install -e .
```

[![jithub](https://circleci.com/gh/russelljjarvis/jit-hub.svg?style=svg)](https://app.circleci.com/pipelines/github/russelljjarvis/jit-hub)


[![DOI](https://zenodo.org/badge/304228004.svg)](https://zenodo.org/badge/latestdoi/304228004)
[![Build Status](https://circleci.com/gh/jit_hub/BluePyOpt.svg?style=svg)](https://app.circleci.com/pipelines/github/russelljjarvis/jit_hub/)
