# Numba/JIT reduced neuoronal models
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
Single neuronal models encoded in python language numba speed up.
Note model waveforms are returned in Neo Format and model currents and voltages are represented with quantities units.
Models:
* Adaptive Exponential.
* [Izhikevich 2007 (not 2003).](https://github.com/OpenSourceBrain/IzhikevichModel/blob/master/numba/faster_izhikevich_model.ipynb)
* Multi Time Scale Adaptive Neuron

To install.
```
pip install jithub==0.1.0
```
or:
```
git clone -b dev https://github.com/russelljjarvis/jit_hub
sudo pip install -e .
```


[![Build Status](https://circleci.com/gh/russelljjarvis/jit_hub/tree/neuronunit.svg?style=svg)](https://app.circleci.com/pipelines/github/russelljjarvis/jit_hub/)

[![DOI](https://zenodo.org/badge/304228004.svg)](https://zenodo.org/badge/latestdoi/304228004)

[Examples](https://github.com/russelljjarvis/jit_hub/blob/neuronunit/jithub/examples/backend_test.ipynb)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://russelljjarvis.github.io/home/"><img src="https://avatars.githubusercontent.com/u/7786645?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Russell Jarvis</b></sub></a><br /><a href="https://github.com/russelljjarvis/jit_hub/commits?author=russelljjarvis" title="Code">💻</a> <a href="https://github.com/russelljjarvis/jit_hub/commits?author=russelljjarvis" title="Documentation">📖</a> <a href="#ideas-russelljjarvis" title="Ideas, Planning, & Feedback">🤔</a> <a href="#design-russelljjarvis" title="Design">🎨</a> <a href="#infra-russelljjarvis" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!