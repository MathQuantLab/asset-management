# Authors : GHAITH Sarahnour, ROISEUX Thomas 
# Date: 2024/03/21

# Exercise 2: Equity portfolio optimization with net zero objectives

#--------------------------------------------
# Importing necessary libraries 
import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm

from typing import Dict, Callable

import warnings

warnings.filterwarnings("ignore")
#--------------------------------------------
#Question 1.a

beta = np.array(
    [0.95, 1.05, 0.45, 1.40, 1.15, 0.75, 1.00, 1.20, 1.10, 0.8, 0.7]
).reshape(-1, 1)
sigma_mat = (
    np.diag(
        [0.262, 0.329, 0.211, 0.338, 0.231, 0.259, 0.265, 0.271, 0.301, 0.274, 0.228]
    )
    ** 2
)

weights = np.array(
    [8.20, 12.30, 6.90, 3.10, 13.20, 12.60, 10.20, 23.00, 4.50, 2.80, 3.20]
) # vector of weights of the assets in the benchmark portfolio
weights = weights / 100 # Converting from percentage to decimal

# Computing covariance matrix
sigma_m = 0.2**2

cov_mat: np.ndarray = sigma_m * (beta.dot(beta.T)) + sigma_mat

# Computing correlation matrix
corr_mat = np.zeros_like(cov_mat)

for i in range(len(corr_mat)):
    for j in range(i, len(corr_mat)):
        corr_mat[i, j] = cov_mat[i, j] / (np.sqrt(sigma_mat[i, i] * sigma_mat[j, j]))
        corr_mat[j, i] = corr_mat[i, j]

# Computing sector volatility 
sector_volatility = np.sqrt(np.diag(cov_mat))

#--------------------------------------------
#Question 1.b

implied_risk_premia: np.ndarray = (
    0.25 * cov_mat.dot(weights) / np.sqrt(weights.T.dot(cov_mat).dot(weights))
)

r_m = 0.25 * 0.2 + 0.03

expected_returns = 0.03 + beta.T * (r_m - 0.03)

#--------------------------------------------
#Question 1.c

weighted_average = np.sum(weights * beta)

#--------------------------------------------
#Question 1.d

implied_risk_premia: np.ndarray = (
    0.25 * cov_mat.dot(weights) / np.sqrt(weights.T.dot(cov_mat).dot(weights))
)

r_m = 0.25 * 0.2 + 0.03

expected_returns = 0.03 + beta.T * (r_m - 0.03)

#--------------------------------------------
#Question 1.e

sci12 = np.array([24, 54, 47, 434, 19, 21, 105, 23, 559, 89, 1655]) # vector of carbon intensity of the assets

ci_b = np.sum(weights * sci12) # carbon intensity of the benchmark portfolio

cm12 = np.array([-2.8, -7.2, -1.8, -1.5, -8.3, -7.8, -8.5, -4.3, -7.1, -2.7, -9.9]) # vector of carbon momentum of the assets
cm12 = cm12 / 100 # Converting from percentage to decimal

cm_b = np.sum(weights * cm12) # carbon momentum of the benchmark portfolio

gii = np.array([0, 1.5, 0, 0.7, 0, 0, 2.4, 0.2, 0.8, 1.4, 8.4]) # vector of green intensity of the assets
gii = gii / 100 # Converting from percentage to decimal

gi_b = np.sum(weights * gii) # green intensity of the benchmark portfolio

#--------------------------------------------
