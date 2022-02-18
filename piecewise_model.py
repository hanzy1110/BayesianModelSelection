#%%
import numpy as np
import pymc3 as pm
import arviz as az
import theano.tensor as tt
import matplotlib.pyplot as plt

def poly_model(time:np.ndarray, n:int=1) -> np.ndarray:
    return np.power(time, 1/n)

xChange = 0.2
k = 5 #Linear slope
beta = 1#poly param
m_1 = 1 #offset
m_2 = m_1 + (beta * xChange**2 - k * xChange)

print(m_1, m_2)

x = np.linspace(0,1, 100)
x_1 = poly_model(x, 1)
x_2 = poly_model(x, 1/2)

y = tt.switch(x<xChange, beta*x_2 + m_1, k * x_1 + m_2)

observed_data = y.eval() + 0.1 * np.random.randn(len(x))
plt.plot(x, y.eval(), label='True line')
plt.scatter(x, observed_data, label='Noisy Data')
plt.legend()
# %%

with pm.Model() as model:
    # k = pm.Normal('k', mu = 0, sigma=5)
    betas = pm.Normal('betas', mu = 0, sigma=5, shape = 2)
    ms = pm.Normal('ms', mu = 0, sigma=5, shape=2)

    nu = pm.HalfNormal('Nu', sigma=1)

    noise = pm.HalfNormal('noise', sigma = 1)
    xChange = pm.Beta('xChange', alpha=2, beta=2)
    # y = tt.switch(x<xChange, beta*x**2 + ms[0], k * x + ms[1])

    y = (betas[0]*x**2 + ms[0]) * tt.switch(x<xChange, 1, 0) + (betas[1] * x + ms[1]) * (1-tt.switch(x<xChange, 1, 0)) 
    
    likelihood = pm.StudentT('likelihood',nu=nu,
                                          mu=y, sigma=noise,
                                          observed = observed_data)
    
    # trace = pm.sample(draws = 3000, tune = 2000)
    trace = pm.sample()
    print(az.summary(trace))
    az.plot_trace(trace)
    
# %%
def piecewise_model(betas, xChange, ms, x):
    return ((betas[0]*x_2 + ms[0]) * tt.switch(x<xChange, 1, 0) + (betas[1]* x_1 + ms[1]) * (1-tt.switch(x<xChange, 1, 0))).eval() 

pm.plot_posterior_predictive_glm(trace,
                                eval = x,
                                lm = lambda x,sample: piecewise_model(sample['betas'], sample['xChange'], sample['ms'], x), 
                                samples=100, 
                                label="Posterior Predictive regression lines")
plt.scatter(x, observed_data, label='Noisy Data')
plt.legend()

# %%
