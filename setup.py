from setuptools import setup

setup(
    name='gosafeopt',
    version='0.1.0',    
    description='Safe Bayesian Optimisation',
    packages=['gosafeopt', 'examples'],
    install_requires=[
        'torch',
        'typer',
        'numpy',
        'seaborn',
        'rich',
        'gpytorch',
        'botorch',
        'gymnasium',
        'wandb',
        'tensorboard',
        'plotly',
        'moviepy',
        'gymnasium[classic-control]',
        "guppy3"
        ]
)
