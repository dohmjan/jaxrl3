# jaxrl3

JAXRL3 is a fork of the amazing [JAXRL2](https://github.com/ikostrikov/jaxrl2), adding support for gymnasium and the latest version of jax. DMC is realized via [Shimmy](https://shimmy.farama.org). Focus is on online RL, hence DRQ and SAC is kept while the rest is cut for now.

## Installation

Run
```bash
pip install --upgrade pip

pip install -e .

# either CPU
pip install --upgrade "jax[cpu]"
# or GPU
# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

See instructions for other versions of CUDA [here](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier).

## Examples

[Here.](examples/)

# Acknowledgements 

Based on work by Ilya Kostrikov.
```
@misc{jaxrl,
  author = {Kostrikov, Ilya},
  doi = {10.5281/zenodo.5535154},
  month = {10},
  title = {{JAXRL: Implementations of Reinforcement Learning algorithms in JAX}},
  url = {https://github.com/ikostrikov/jaxrl2},
  year = {2022},
  note = {v2}
}
```
