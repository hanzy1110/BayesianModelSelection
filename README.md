# BayesianModelSelection
Bayesian model selection for Oxidation Kinetics modelling \\
Install requirements in a python enviroment using pip install -r requirements.txt

Milestones completed:
6/2/2022
- Implemented Inference for diferent models
- Posterior Predictive plots for every model
- Results in the IPython Notebook!
- Compute BIC and logLikelihood for every model
- Estimate ELPD using WAIC and compare each model
- Extend the analysis to the whole dataset
19/2/2022
- Added Paralinear model to the model selection framework (taken from Paralinear Oxidation of Cr-Si-C-Coated C/SiC at 1300�C in Wet
and Dry Air Environments WU2019)
- Spline regression implemented (https://docs.pymc.io/projects/examples/en/latest/splines/spline.html)

TODO
- Perform data sanitation/normalizing
- Add spline regression to the model selection framework
