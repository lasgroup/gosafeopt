from setuptools import setup

setup(
    name="gosafeopt",
    version="0.1.0",
    description="Safe Bayesian Optimisation",
    packages=["gosafeopt", "examples"],
    install_requires=[
        "torch==2.0.1",
        "typer==0.9.0",
        "numpy==1.25.2",
        "seaborn==0.12.2",
        "rich==13.5.2",
        "gpytorch==1.10",
        "botorch==v0.8.5",
        "gymnasium==0.29.0",
        "wandb==0.15.8",
        "tensorboard==2.13.0",
        "plotly==5.15.0",
        "moviepy==1.0.3",
        "gymnasium[classic-control]",
        "guppy3==3.1.3",
    ],
)
