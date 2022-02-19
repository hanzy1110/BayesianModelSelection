#%%
import arviz as az
import numpy as np
import pandas as pd
import pymc3 as pm

# import statsmodels.api as sm

import matplotlib.pyplot as plt
from patsy import dmatrix
data = pd.read_csv('testing/81Fig.4_Al12.0-Cr24.0-Mo4.0-Nb24.0-Ti24.0-Zr12.0_1000C_160mmHg.csv')
data.plot()

mass_gain = np.array(data['mass gain'])
mass_gain = mass_gain[~np.isnan(mass_gain)]
exp_mass_gain = np.exp(mass_gain)

time = np.array(data['time'])
time = time[~np.isnan(time)]

fig, ax = plt.subplots(2,1, figsize=(12,5))
ax[0].scatter(data['time'], data['mass gain'])
ax[1].hist(mass_gain)

max_quantiles = 4
knot_list = np.quantile(time, q=np.linspace(0,1,max_quantiles))

#%%

B = dmatrix("bs(time, knots=knots, degree=3, include_intercept=True) - 1",
    {"time": time, "knots": knot_list[1:-1]},
)

spline_df = (
    pd.DataFrame(B)
    .assign(time = time)
    .melt("time", var_name="spline_i", value_name="value")
)

color = plt.cm.magma(np.linspace(0, 0.80, len(spline_df.spline_i.unique())))

fig = plt.figure()
for i, c in enumerate(color):
    subset = spline_df.query(f"spline_i == {i}")
    subset.plot("time", "value", c=c, ax=plt.gca(), label=i)
plt.legend(title="Spline Index", loc="upper center", fontsize=8, ncol=6)
#%%

COORDS = {"obs": np.arange(len(mass_gain)), "splines": np.arange(B.shape[1])}
with pm.Model(coords=COORDS) as spline_model:
    a = pm.Normal("a", 100, 5)
    
    alphas = pm.Exponential("alphas", lam = 1, dims = "splines") 
    betas = pm.Exponential("betas", lam = 1, dims = "splines") 
    w = pm.Gamma("w", alpha=alphas, beta = betas, dims="splines")
    
    mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
    sigma = pm.Exponential("sigma", 1)
    # nu = pm.Exponential("nu", 1)
    # D = pm.StudentT("D", nu = nu, mu=mu, sigma=sigma, observed=mass_gain, dims="obs")
    D = pm.Normal("D", mu=mu, sigma=sigma, observed=mass_gain, dims="obs")

# %%
with spline_model:
    prior_pred = pm.sample_prior_predictive()
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=3,
        return_inferencedata=True,
    )
    post_pred = pm.sample_posterior_predictive(trace)
    trace.extend(az.from_pymc3(prior=prior_pred, posterior_predictive=post_pred))
    az.plot_trace(trace, var_names=["a", "w", "sigma"])
    az.plot_forest(trace, var_names=["w"], combined=False)
# %%
post_pred = az.summary(trace, var_names=["mu"]).reset_index(drop=True)
mass_gain_data_post = data.copy().reset_index(drop=True)
mass_gain_data_post["pred_mean"] = post_pred["mean"]
mass_gain_data_post["pred_hdi_lower"] = post_pred["hdi_3%"]
mass_gain_data_post["pred_hdi_upper"] = post_pred["hdi_97%"]

mass_gain_data_post.plot.scatter(
    "time",
    "mass gain",
    color="cornflowerblue",
    s=10,
    title="Mass gain data with posterior predictions",
    ylabel="Mass gain",
)
for knot in knot_list:
    plt.gca().axvline(knot, color="grey", alpha=0.4)

mass_gain_data_post.plot("time", "pred_mean", ax=plt.gca(), lw=3, color="firebrick")
plt.fill_between(
    mass_gain_data_post.time,
    mass_gain_data_post.pred_hdi_lower,
    mass_gain_data_post.pred_hdi_upper,
    color="firebrick",
    alpha=0.4,
)

# %%
