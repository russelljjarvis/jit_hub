# numba reduced neuoronal models
Single neuronal models encoded in python language numba speed up.
Note model waveforms are returned in Neo Format and model currents and voltages are represented with quantities units.
Models:
* Adaptive Exponential.
* [Izhikevich 2007 (not 2003).](https://github.com/OpenSourceBrain/IzhikevichModel/blob/master/numba/faster_izhikevich_model.ipynb)
* Multi Time Scale Adaptive Neuron

To install.
```
git clone -b dev https://github.com/russelljjarvis/jit_hub
sudo pip install -e .
```


[![Build Status](https://circleci.com/gh/russelljjarvis/jit_hub/tree/neuronunit.svg?style=svg)](https://app.circleci.com/pipelines/github/russelljjarvis/jit_hub/)

[![DOI](https://zenodo.org/badge/304228004.svg)](https://zenodo.org/badge/latestdoi/304228004)

#[Examples](https://github.com/russelljjarvis/jit_hub/blob/neuronunit/jithub/examples/backend_test.ipynb)
