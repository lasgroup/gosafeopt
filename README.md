# Tuning Legged Locomotion Controllers via Safe Bayesian Optimization

![Bayesian Optimization animation](/doc/animation.gif?raw=true "Bayesian optimization animation")

This repository contains the code for our paper [Tuning Legged Locomotion Controllers via Safe Bayesian Optimization](https://arxiv.org/abs/2306.07092). A demo video is available at [https://www.youtube.com/watch?v=pceHWpFr3ng](https://www.youtube.com/watch?v=zDBouUgegrU).

All relevant code for the paper is in `gosafeopt/aquisitions/*`, `gosafeopt/models/__init__.py`, `gosafeopt/optim/swarm_opt.py` and `gosafeopt/optim/base_optimizer.py`.

## Setup

```
#With poetry
poetry install

#With pip/venv
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

It might be necessary to create a wandb account at wandb.ai if not already existing.

## Examples

To train the pendulum toy problem model with a specific aquisition function and show the result plot run

```
poetry install --with examples
python examples/pendulum.py
```
