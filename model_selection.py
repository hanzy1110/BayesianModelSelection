#%%
from typing import Callable, Dict, Tuple, Any
import pandas as pd
import numpy as np
import arviz as az
import pymc3 as pm
from pymc3.backends.base import MultiTrace

import matplotlib.pyplot as plt

data:pd.DataFrame = pd.read_csv('data/testdata.csv')

plt.scatter(data['time'], data[' mass gain'], label = 'Experimental Data')
plt.xlabel('Time')
plt.ylabel('Mass Gain')

# plt.scatter(np.log(data['time']),data[' mass gain'])

def build_model(time_funtion: Callable, 
                observed_values_transformation:Callable, 
                data:pd.DataFrame = data) -> Tuple[pm.Model, np.ndarray, np.ndarray]:

    time = np.array(data['time'].values)
    time = time_funtion(time)

    mass_gain = np.array(data[' mass gain'].values)
    mass_gain = observed_values_transformation(mass_gain)
    
    with pm.Model() as model:

        nu = pm.HalfNormal('Nu', sigma=1)

        k = pm.Normal('k', mu=0, sigma = 10)
        C = pm.HalfCauchy('C', beta = 8)
        
        noise = pm.HalfCauchy('noise',beta=5)
        mean = k*time + C
        
        likelihood = pm.StudentT('likelihood', nu=nu, mu = mean, sigma = noise, observed = mass_gain)
        
    return model, time, mass_gain

def poly_model(time:np.ndarray, n:int=1) -> np.ndarray:
    return np.power(time, 1/n)

def log_model(time:np.ndarray) -> np.ndarray:
    return np.log(time)

def inv_log_model(time:np.ndarray) -> np.ndarray:
    return 1/time

# trace = pm.sample(draws=5000, chains = 3, tune=4000, target_accept=0.8)

model_mapping: Dict[str, Tuple[Callable, Callable]] = {
    'linear_model': (lambda x: poly_model(x, n=1), lambda x: poly_model(x, n=1)),
    'cuadratic_model': (lambda x: poly_model(x, n=2),lambda x: poly_model(x, n=1)),
    'cubic_model': (lambda x: poly_model(x, n=3),lambda x: poly_model(x, n=1)),
    'cuartic_model': (lambda x: poly_model(x, n=4),lambda x: poly_model(x, n=1)),
    'log_model': (log_model, poly_model),
    'inv_log_model': (log_model, inv_log_model)
}

#%%
# Initial_inference process->
trace_dict: Dict[str, az.InferenceData] = {}

for model, args in model_mapping.items():
    print('-'*30)
    print(f'Performing Inference for: {model}')
    
    Inference_model, time, mass_gain = build_model(*args)
    
    with Inference_model:

        trace = pm.sample(return_inferencedata=True)
        trace_dict[model] = trace
        sum = az.summary(trace)
        sum.to_csv(f'summaries/{model}.csv')
        
        # print(sum)
        az.plot_trace(trace)

#%%
# Plotting the posterior ->
for model, args in model_mapping.items():
    Inference_model, time, mass_gain = build_model(*args)
    
    old_time = np.array(data['time'].values)
    
    plt.figure(figsize=(7, 7))
    plt.xlabel(f'Time transformed according to: {model}')
    plt.ylabel(f'Mass Gain transformed according to: {model}')
        
    plt.scatter(time, mass_gain, label = 'Experimental Data')
    pm.plot_posterior_predictive_glm(trace_dict[model],
                                     eval = time,
                                     lm = lambda x,sample: sample['C'] + sample['k'] * x, 
                                     samples=100, 
                                     label="Posterior Predictive regression lines")
    plt.legend()

# %%
