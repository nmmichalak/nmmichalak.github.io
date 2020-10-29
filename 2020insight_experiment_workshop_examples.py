# import packages and/or modules
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linearmodels as lm

# set pyplot style
plt.style.use(["ggplot", "seaborn-colorblind"])

# example experiment data
experiment = pd.read_csv("ivexample.csv")

# recode
experiment["treated"] = experiment["assign"].replace({0: -0.5, 1: 0.5})
experiment["complied"] = experiment["takeup"].replace({0: -0.5, 1: 0.5})

# 2-stage least squares
iv2sls_fit1 = lm.IV2SLS.from_formula("y ~ 1 + [complied ~ treated]", data = experiment).fit(cov_type = "robust")

# results
iv2sls_fit1

# a path model
ols_a1 = smf.ols("complied ~ treated", data = experiment)

# b path model
ols_b1 = smf.ols("y ~ treated + complied", data = experiment)

# mediation
mediation_fit1 = sm.stats.Mediation(ols_b1, ols_a1, exposure = "treated", mediator = "complied").fit()

# results
mediation_fit1.summary()

# data
cigarettes_sw = sm.datasets.get_rdataset(dataname = "CigarettesSW", package = "AER").data

# data documentation
sm.datasets.get_rdataset(dataname = "CigarettesSW", package = "AER").__doc__

# pairs of scatter plots and histograms
pd.plotting.scatter_matrix(cigarettes_sw.loc[:, ["cpi", "packs", "tax", "price", "taxs"]], alpha = 0.25);

# compute sales tax on cigarettes
cigarettes_sw["cigarette_tax"] = (cigarettes_sw["taxs"] - cigarettes_sw["tax"]) / cigarettes_sw["cpi"]

# compute price per cpi
cigarettes_sw["price_cpi_adj"] = cigarettes_sw["price"] / cigarettes_sw["cpi"]

# subset 1995
cigarettes_sw1995 = cigarettes_sw.loc[cigarettes_sw["year"] == 1995, :].copy()

# mean-center predictors
cigarettes_sw1995["price_cpi_adj_c"] = cigarettes_sw1995["price_cpi_adj"] - cigarettes_sw1995["price_cpi_adj"].mean()
cigarettes_sw1995["cigarette_tax_c"] = cigarettes_sw1995["cigarette_tax"] - cigarettes_sw1995["cigarette_tax"].mean()

# descriptives
cigarettes_sw1995.loc[:, ["packs", "price_cpi_adj", "price_cpi_adj_c", "cigarette_tax", "cigarette_tax_c"]].describe()

# firt stage regression
ols_fit1 = smf.ols("np.log(price_cpi_adj) ~ 1 + cigarette_tax_c", data = cigarettes_sw1995).fit()

# results
ols_fit1.summary()

# add fitted values to data
cigarettes_sw1995["stage1_predicted"] = ols_fit1.predict()

# second stage regression
ols_fit2 = smf.ols("np.log(packs) ~ 1 + stage1_predicted", data = cigarettes_sw1995).fit()

# summary
ols_fit2.summary()

# 2-stage least squares
iv2sls_fit2 = lm.IV2SLS.from_formula("np.log(packs) ~ 1 + [np.log(price_cpi_adj) ~ cigarette_tax_c]", data = cigarettes_sw1995).fit(cov_type = "robust")

# results
iv2sls_fit2

# save list of sample sizes given range of effect sizes
d = np.arange(0.10, 1.0, 0.10)
n_80power = [sm.stats.tt_ind_solve_power(effect_size = d_i, alpha = 0.05, power = 0.80, ratio = 1, alternative = "two-sided") for d_i in d]

# figure and axes
fig, ax = plt.subplots(figsize = (12, 6))

# scatter plot
ax.scatter(x = d, y = n_80power)

# line plot
ax.plot(d, n_80power)

# titles
ax.set_xlabel("Effect size in stadard deviation units")
ax.set_title("Number of users needed to detect a given effect size\ntwo independent groups, 80% power, $\\alpha = 0.05$, two-sided alternative")
