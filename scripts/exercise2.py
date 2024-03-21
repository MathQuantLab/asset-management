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
#Question 2.b

def solve_esg_qp_problem(t: float) -> np.ndarray:
    """Solver for the quadratic programming problem

    Args:
        t (float): t parameter

    Returns:
        np.ndarray: optimal portfolio weights
    """
    w = cp.Variable(11, "w")  # Portfolio weights

    objective = cp.Minimize(0.5 * cp.quad_form(w, cov_mat))
    constraints = [
        cp.sum(w) == 1,
        0 <= w,
        w <= 1,
        cp.sum(sci12 * w) <= 0.7 * (0.93**t) * ci_b,
    ]
    problem = cp.Problem(objective, constraints)

    problem.solve()

    return w.value

optimal_pfs12 = {}

for t in (0, 1, 2, 5, 10):

    sol = solve_esg_qp_problem(t)

    #print("Optimal portfolio for t =", t, ":", sol)

    tracking_eror_volatility = np.sqrt(
        (sol - weights).T.dot(cov_mat).dot(sol - weights)
    )

    #print("Tracking error volatility:", tracking_eror_volatility)

    carbon_intensity = sci12.dot(sol)

    #print("Carbon intensity:", carbon_intensity)

    carbon_momentum = cm12.dot(sol)

    #print("Carbon momentum:", carbon_momentum)
    green_intensity = gii.dot(sol)

    #print("Green intensity:", green_intensity)

    reduction_rate = 1 - (sci12.dot(sol)) / (ci_b)

    #print("Reduction rate:", reduction_rate)
    optimal_pfs12[t] = sol
    
#--------------------------------------------
#Question 2.c

scii13 = np.array([78, 203, 392, 803, 55, 124, 283, 123, 892, 135, 1867]) # vector of carbon intensity of the assets for scope 1, 2 and 3

ci13_b = np.sum(weights * scii13) # carbon intensity of the benchmark portfolio for scope 1, 2 and 3
ci13_b

cmi13 = np.array([-0.8, -1.6, -0.1, -0.2, -1.9, -2.0, -2.5, 2.1, -3.6, -0.8, -6.8]) # vector of carbon momentum of the assets for scope 1, 2 and 3
cmi13 = cmi13 / 100 # Converting from percentage to decimal

def solve_esg_qp_problem13(t: float) -> np.ndarray:
    """Solver for the quadratic programming problem

    Args:
        t (float): t parameter

    Returns:
        np.ndarray: optimal portfolio weights
    """
    w = cp.Variable(11, "w")  # Portfolio weights

    objective = cp.Minimize(0.5 * cp.quad_form(w, cov_mat))
    constraints = [
        cp.sum(w) == 1,
        0 <= w,
        w <= 1,
        cp.sum(scii13 * w) <= 0.7 * (0.93**t) * ci13_b,
    ]
    problem = cp.Problem(objective, constraints)

    problem.solve()

    return w.value

optimal_pfs13 = {}

for t in (0, 1, 2, 5, 10):

    sol = solve_esg_qp_problem13(t)

    #print("Optimal portfolio for t =", t, ":", sol)

    tracking_eror_volatility = np.sqrt(
        (sol - weights).T.dot(cov_mat).dot(sol - weights)
    )

    #print("Tracking error volatility:", tracking_eror_volatility)

    carbon_intensity = scii13.dot(sol)

    #print("Carbon intensity:", carbon_intensity)

    carbon_momentum = cmi13.dot(sol)

    #print("Carbon momentum:", carbon_momentum)
    green_intensity = gii.dot(sol)

    #print("Green intensity:", green_intensity)

    reduction_rate = 1 - (scii13.dot(sol)) / (ci13_b)

    #print("Reduction rate:", reduction_rate)
    optimal_pfs13[t] = sol
    
#--------------------------------------------
#Question 2.d
for key12, key13 in zip(optimal_pfs12, optimal_pfs13):
    implied_expected_return = expected_returns.dot(optimal_pfs12[key12])
    print(
        "Implied expected, taking into consideration 1 and 2, return for t =",
        key12,
        ":",
        np.sum(implied_expected_return - expected_returns),
    )
    implied_expected_return = expected_returns.dot(optimal_pfs13[key13])
    print(
        "Implied expected, taking into consideration 1, 2 and 3, return for t =",
        key13,
        ":",
        np.sum(implied_expected_return - expected_returns),
    )