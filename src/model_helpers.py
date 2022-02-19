import numpy as np
import theano.tensor as tt
from scipy.optimize import fsolve

def poly_model(time:np.ndarray, n:int=1) -> np.ndarray:
    return np.power(time, 1/n)

def log_model(time:np.ndarray) -> np.ndarray:
    return np.log(time)

def inv_log_model(time:np.ndarray) -> np.ndarray:
    return 1/time


# Helper functions to run and use the paralinear model


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