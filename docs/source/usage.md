(installation)=

# Installation

Clone the github repository

```console
$ git clone https://github.com/prajeeshag/mitgcm_utils.git
$ cd mitgcm_utils
$ micromamba env create -f environment.yml
$ micromamba activate mitgcm
$ pip install .
```

This will create a new conda environment named *mitgcm* and installs all neccesary dependencies and packages.

Activate this environment to use the cli utilities.
```console
$ micromamba activate mitgcm
```
:::{note}
You can use `conda` or `mamba` in place of `micromamba`. I prefer `micromamba`, as it is much faster.
- Refer [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install) for installing *micromamba*
- Refer [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for installing *conda*.
- Refer [Mamba documentation](https://mamba.readthedocs.io/en/latest/installation.html) for installing *mamba*
:::

# Usage

```{eval-rst}
.. click:: mitgcm_utils.mkMITgcmEXF:app_click
   :prog: mkMITgcmEXF
   :nested: full
```
