# Jet clustering
## References
- Fastjet python library: https://fastjet.readthedocs.io/en/latest/index.html
- Fastjet manual: https://fastjet.fr/repo/fastjet-doc-3.4.1.pdf
- Anti-kT algorithm paper: https://iopscience.iop.org/article/10.1088/1126-6708/2008/04/063

## Installation
To perform the jet clustering, the [Fastjet](https://fastjet.readthedocs.io/en/latest/index.html) Python library is used.
It is advised to work in a clean conda environment. It is recommended to use [**micromamba**](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) to setup the environment.
First of all, if `micromamba` is not installed in your system, you need to install it. Then you create a new Python environment:

```bash
# Install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba create -n fastjet python=3.11 -c conda-forge
micromamba activate fastjet
```

Install the `fastjet` python package:
```bash
pip install fastjet
```

Whenever you want to work in this environment, you can activate it with:
```bash
micromamba activate fastjet
```

## Example
A first basic example to perform the jet clustering is provided in `jet_clustering.py`.
