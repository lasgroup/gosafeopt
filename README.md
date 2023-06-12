# Safe Controller Tuning for Quadrupedal Locomotion
Actively learning the disturbance parameters of a dynamical systems.

Relevant code for the paper is in `bayopt/aquisitions/*`, `bayopt/models/__init__.py`, `bayopt/optim/swarm_opt.py` and `bayopt/optim/base_optimizer.py`.

## Setup
```
python -m venv .venv
. .venv/bin/activate
pip install -e . 
wandb login
```
 
It might be necessary to create a wandb account at wandb.ai if not already existing.

## Examples 

To train the pendulum toy problem model with a specific aquisition function and show the result plot run 

```
 python examples/pendulum.py train --aquisition GoSafeOpt --seed 42

 python examples/pendulum.py plot --data-path wandb/wandb/{RUN_NAME}/files/res               
```

The aquisition function can be one of `GoSafeOpt, SafeOptMultiStage, SafeUCB, SafeEI, UCB`.


## Settings
Most settings can be changed in `config.txt`

