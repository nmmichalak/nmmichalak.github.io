# import packages and/or modules
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import linearmodels as lm

# set pyplot style
plt.style.use(["ggplot", "seaborn-colorblind"])

# seaborn style
sns.set(palette = "colorblind", font_scale = 1.5)

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
ax.set_xlabel("Effect size in standard deviation units")
ax.set_title("Number of users needed to detect a given effect size\ntwo independent groups, 80% power, $\\alpha = 0.05$, two-sided alternative")

# packages
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# difference-in-difefrence example
## parameters
N = 100
b0 = 20
b1 = -1.5
b2 = -1
b3 = 3

## data
np.random.seed(357904)
wage_increase = np.repeat([-0.5, 0.5], repeats = N / 2)
period_change = np.repeat([-0.5, 0.5, -0.5, 0.5], repeats = N / 4)
full_time = b0 + b1 * (wage_increase) + b2 * (period_change) + b3 * (wage_increase * period_change) + np.random.normal(loc = 0, scale = 1, size = N)
did_data = pd.DataFrame({"full_time": full_time, "period_change": period_change, "wage_increase": wage_increase})

# labels
did_data["state"] = did_data["wage_increase"].replace({-0.5: "Pennsylvania", 0.5: "New Jersey"}).astype("category").cat.reorder_categories(["Pennsylvania", "New Jersey"])
did_data["period"] = did_data["period_change"].replace({-0.5: "Before", 0.5: "After"}).astype("category").cat.reorder_categories(["Before", "After"])

# plot
plt.figure(figsize = (12, 6))
sns.violinplot(x = "period", y = "full_time", hue = "state", data = did_data)
plt.legend(loc = "best")
plt.xlabel(None)
plt.ylabel(None)
plt.title("Full-Time Equivalent")
plt.tight_layout()

# regression
## create interaction term
did_data["did"] = np.where((period_change == -0.5) & (wage_increase == -0.5), 0.5, 
np.where((period_change == -0.5) & (wage_increase == 0.5), -0.5, 
np.where((period_change == 0.5) & (wage_increase == -0.5), -0.5, 
np.where((period_change == 0.5) & (wage_increase == 0.5), 0.5, np.nan))))

# fit regression
ols_fit3 = smf.ols("full_time ~ period_change + wage_increase + did", data = did_data).fit()

# results
ols_fit3.summary()
