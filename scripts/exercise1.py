import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm

from typing import Dict, Callable

import warnings

warnings.filterwarnings("ignore")

corr_mat = np.array(
    [
        [1, 0.5, 0.3, 0.6, 0.4],
        [0.5, 1, 0.3, 0.6, 0.3],
        [0.3, 0.3, 1, 0.6, 0.7],
        [0.6, 0.6, 0.6, 1, 0.3],
        [0.4, 0.3, 0.7, 0.3, 1],
    ],
    dtype=np.float64,
)

r = 0.02  # risk free rate

mu = np.array([0.05, 0.05, 0.06, 0.04, 0.07], dtype=np.float64).T
sigma = np.array([0.2, 0.22, 0.25, 0.18, 0.45], dtype=np.float64).T

# Question 1.a
cov_mat = np.zeros_like(corr_mat)  # Covariance matrix

for i in range(len(cov_mat)):
    for j in range(i, len(cov_mat)):
        cov_mat[i, j] = corr_mat[i, j] * sigma[i] * sigma[j]
        cov_mat[j, i] = cov_mat[i, j]


print(cov_mat)

# Question 1.b
sr = (mu - r) / sigma  # Sharpe ratio

for k in range(len(mu)):
    print("Sharpe ratio for asset", k + 1, ":", sr[k])


# Question 2.b
def solve_qp_problem(gamma: float) -> np.ndarray:
    """Solver for the quadratic programming problem

    Args:
        gamma (float): gamma parameter

    Returns:
        np.ndarray: optimal portfolio weights
    """
    x = cp.Variable(len(mu), "x")  # Portfolio weights

    objective = cp.Minimize(
        0.5 * cp.quad_form(x, cov_mat) - gamma * cp.matmul(mu - r, x)
    )
    constraints = [cp.sum(x) == 1, -10 <= x, x <= 10]
    problem = cp.Problem(objective, constraints)

    problem.solve()

    return x.value


optimal_x: Dict[float, np.ndarray] = {}
gammas = [0, 0.1, 0.2, 0.5, 1]

for gamma in gammas:
    sol = solve_qp_problem(gamma)
    print("Optimal portfolio for gamma =", gamma, ":", sol)
    optimal_x[gamma] = sol

for gamma, sol in optimal_x.items():
    print("Gamma:", gamma)
    expected_return = np.dot(sol.T, mu)
    volatility = sol.T.dot(cov_mat).dot(sol)
    sharpe_ratio = (expected_return - r) / volatility
    print("  Expected return:", expected_return)
    print("  Volatility:", volatility)
    print("  Sharpe ratio:", sharpe_ratio)

# Question 2.c

precise_gammas = np.linspace(-10, 10, 500)

solutions = [solve_qp_problem(gamma) for gamma in precise_gammas]
returns = np.array([np.dot(sol.T, mu) * 100 for sol in solutions])
volatilities = np.array([sol.T.dot(cov_mat).dot(sol) * 100 for sol in solutions])

plt.figure(figsize=(10, 6))
plt.plot(
    volatilities,
    returns,
    label="Efficient frontier",
)
plt.legend()
plt.grid(visible=True)
plt.xlabel("Volatility (in %)")
plt.ylabel("Expected return (in %)")
plt.title("Efficient frontier")
plt.show()

# Question 2.d
def bisection_algorithm(
    f: Callable[[float], float],
    a: float,
    b: float,
    target: float,
    /,
    tol: float = 10**-5,
    *,
    max_iter: int = 100,
) -> float:
    """Bisection algorithm

    Args:
        f (function): function to find the root of
        a (float): left bound
        b (float): right bound
        target (float): target value
        tol (float, optional): tolerance. Defaults to 10**-5.
        max_iter (int, optional): maximum number of iterations. Defaults to 100.

    Returns:
        float: root of the function
    """
    if b - a < 0:
        a, b = b, a

    gamma_bar = (a + b) / 2

    if b - a < tol or max_iter == 0:
        return gamma_bar

    f_bar = f(gamma_bar)

    if f_bar < target:
        a = gamma_bar
    else:
        b = gamma_bar

    return bisection_algorithm(f, a, b, target, tol=tol, max_iter=max_iter - 1)

targets = (16, 20)


def volatilities_function(gamma: float) -> float:
    sol = solve_qp_problem(gamma)
    return sol.T.dot(cov_mat).dot(sol) * 100


for target in targets:
    gamma = bisection_algorithm(
        volatilities_function,
        0,
        100,
        target,
    )
    print("Gamma for target volatility of", target, ":", gamma)
    print("  Expected return:", np.dot(solve_qp_problem(gamma).T, mu) * 100)
    print("  Volatility:", volatilities_function(gamma))
    print(
        "  Sharpe ratio:",
        (np.dot(solve_qp_problem(gamma).T, mu) - r) / volatilities_function(gamma),
    )

sharpe_ratios = (returns - r * 100) / volatilities  # Converting r into %

i = np.argmax(sharpe_ratios)
tp = solve_qp_problem(precise_gammas[i])
print("Optimal portfolio for maximum Sharpe ratio:", tp)
print("Maximum Sharpe ratio:", sharpe_ratios[i])
print("Expected return (in %):", returns[i])
print("Volatility (in %):", volatilities[i])

# Question 3.a

def solve_long_only_qp_problem(gamma: float) -> np.ndarray:
    """Solver for the quadratic programming problem

    Args:
        gamma (float): gamma parameter

    Returns:
        np.ndarray: optimal portfolio weights
    """
    x = cp.Variable(len(mu), "x")  # Portfolio weights

    objective = cp.Minimize(
        0.5 * cp.quad_form(x, cov_mat) - gamma * cp.matmul(mu - r, x)
    )
    constraints = [cp.sum(x) == 1, 0 <= x, x <= 1]
    problem = cp.Problem(objective, constraints)

    problem.solve()

    return x.value

for gamma in gammas:
    sol = solve_long_only_qp_problem(gamma)
    print("Optimal portfolio for gamma =", gamma, ":", sol)
    print("Gamma:", gamma)
    expected_return = np.dot(sol.T, mu)
    volatility = sol.T.dot(cov_mat).dot(sol)
    sharpe_ratio = (expected_return - r) / volatility
    print("  Expected return (in %):", expected_return * 100)
    print("  Volatility (in %):", volatility * 100)
    print("  Sharpe ratio:", sharpe_ratio)
    
# Question 3.b

long_solutions = [solve_long_only_qp_problem(gamma) for gamma in precise_gammas]
long_returns = np.array([np.dot(sol.T, mu) * 100 for sol in long_solutions])
long_volatilities = np.array(
    [sol.T.dot(cov_mat).dot(sol) * 100 for sol in long_solutions]
)

plt.figure(figsize=(10, 6))
plt.plot(
    volatilities,
    returns,
    label="Efficient frontier long long/short",
)
plt.plot(
    long_volatilities,
    long_returns,
    label="Efficient frontier long only",
)
plt.legend()
plt.grid(visible=True)
plt.xlabel("Volatility (in %)")
plt.ylabel("Expected return (in %)")
plt.title("Efficient frontier")
plt.show()

# Question 3.c

for target in targets:
    gamma = bisection_algorithm(
        volatilities_function,
        0,
        100,
        target,
    )
    print("Gamma for target volatility of", target, ":", gamma)
    print("  Expected return:", np.dot(solve_long_only_qp_problem(gamma).T, mu) * 100)
    print("  Volatility:", volatilities_function(gamma))
    print(
        "  Sharpe ratio:",
        (np.dot(solve_qp_problem(gamma).T, mu) - r) / volatilities_function(gamma),
    )

# Question 3.d
sharpe_ratios = (long_returns - r * 100) / long_volatilities  # Converting r into %

i = np.argmax(sharpe_ratios)
tp = solve_long_only_qp_problem(precise_gammas[i])
print("Optimal portfolio for maximum Sharpe ratio:", tp)
print("Maximum Sharpe ratio:", sharpe_ratios[i])
print("Expected return (in %):", returns[i])
print("Volatility (in %):", volatilities[i])

# Question 3.e

beta = cov_mat.dot(tp) / tp.T.dot(cov_mat).dot(tp)

for i, value in enumerate(beta):
    print("Beta for asset", i + 1, ":", value)