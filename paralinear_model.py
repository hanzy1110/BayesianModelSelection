# %%
import numpy as np
import pymc3 as pm
import arviz as az
import pandas as pd

import theano.tensor as tt
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def helper(mass_gain, t, kp, kl):
    return mass_gain - kp/kl * (1-np.exp(mass_gain)**(kl/kp)) - t

def helper_2(mass_gain, t, kp, kl):
    return kp/(2*kl**2)*((-2*kl/kp)*mass_gain - np.log(1-(2*kl/kp) * mass_gain)) - t

def helper_3(mass_gain, t, kp, kl):
    return (1-(2*kl/kp)*mass_gain-np.exp(-(2*kl**2/kp) * t)*np.exp(-(2*kl/kp) * mass_gain))

def helper_3_theano(mass_gain, t, kp, kl):
    return (1-(2*kl/kp)*mass_gain-tt.exp(-(2*kl**2/kp) * t)*tt.exp(-(2*kl/kp) * mass_gain))


def helper_theano(mass_gain, t, kp, kl):
    return kp/(2*kl**2)*((-2*kl/kp)*mass_gain - tt.log(1-(2*kl/kp) * mass_gain)) - t

def helperFprime(mass_gain, kp,kl):
    return (mass_gain/(kp-2*kl*mass_gain))

def paralinear_model(time, kp, kl):
    # mass_gain = np.fromiter(map(lambda t: fsolve(lambda x: helper(x, t, kp, kl), 1), time), dtype=np.float64)
    _mass_gain = np.ndarray((len(time),))
    for i, t in enumerate(time):
        _mass_gain[i] = fsolve(lambda x: helper_3(x, t, kp, kl), x0=1)
    return _mass_gain

# dm = kp/kl * ln( kp/(kp-kl(dm-klt))) was morphed into: t = dm -(kp/kl)(1-exp(dm*(kp/kl)))
# to help with the regression as now we have a form t = x + y ** a + C which may be easier for regression


path = 'test_data/81Fig.4_Al11.5-Cr23.0-Mo8.0-Nb23.0-Ti23.0-Zr11.5_1000C_160mmHg.csv'
data = pd.read_csv(path)
# data.plot()
mass_gain = np.array(data['mass gain'])
mass_gain = mass_gain[~np.isnan(mass_gain)]
exp_mass_gain = np.exp(mass_gain)

time = np.array(data['time'])
time = time[~np.isnan(time)]

plt.scatter(data['time'], data['mass gain'])
mass = paralinear_model(time, 100, 1)
plt.plot(time, mass)
# %%

with pm.Model() as model:
    # kp = ks[0] kl = ks[1]
    kp = pm.HalfCauchy('kp', beta=10)
    kl = pm.HalfNormal('kl', sigma=1)

    # mu = mass_gain - kp/kl * (1-exp_mass_gain**(kl/kp))

    mu = helper_theano(mass_gain, t=0, kp=kp, kl=kl)

    # Use studentT since it gave better results!
    nu = pm.HalfNormal('Nu', sigma=1)
    noise = pm.HalfNormal('noise', sigma=1)

    likelihood = pm.StudentT('likelihood',
                             mu=mu,
                             nu=nu,
                             sigma=noise,
                             observed=time)
    # step = pm.Metropolis()
    trace = pm.sample(return_inferencedata=False, target_accept=0.95)
    
    # trace = pm.sample_smc(draws=4000, p_acc_rate=0.95)

    az.plot_trace(trace)
    print(az.summary(trace))

# %%
def paralinear_model_inference(x, ks):
    return x - ks[0]/ks[1] * (1-np.exp(x)**(ks[1]/ks[0]))

pm.plot_posterior_predictive_glm(trace,
                                 eval=time,
                                 lm=lambda x, sample: paralinear_model(
                                     x, sample['kp'], sample['kl']),
                                 samples=100,
                                 label="Posterior Predictive regression lines")

plt.scatter(time, mass_gain, label='Noisy Data')
plt.ylabel('mass gain')
plt.xlabel('time')
plt.legend()
# %%
