# gg02.py
# Copyright 2020 Alexandros Georgios Mountogiannakis

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import warnings
from scipy.special import erfinv
from numba import njit
import maths
import math


def optimal_modulation_variance(t, h, x, v, b):
    """
    Generates the optimal modulation variance for a certain group of noise parameters and a fixed reconciliation
    efficiency.
    :param t: The channel transmissivity.
    :param h: The setup efficiency.
    :param x: The excess noise.
    :param v: The electronic noise.
    :param b: The reconciliation efficiency.
    :return: The optimal modulation variance.
    """

    mu_range = np.arange(2, 500, 0.001)  # Identify the optimal mu
    R_opt = np.zeros_like(mu_range, dtype=np.float64)  # Optimized key rate for a fixed reconciliation efficiency β
    for i in range(len(mu_range)):
        mut, hol, R_opt[i] = key_rate_calculation(mu_range[i], t, h, x, v, b)
    mu = mu_range[np.argmax(R_opt)]  # Identify the index where the key rate is maximum and use it to get the optimal μ
    return mu


@njit(fastmath=True, parallel=True, cache=True)
def prepare_states(n, m):
    """
    In a coherent-state protocol, Alice prepares a bosonic mode A in a coherent state |α⟩ whose amplitude is
    Gaussian-modulated. In other words, we may write α = (q + ip) / 2, where x = q, p is the mean value of the generic
    quadrature xˆ =  qˆ, pˆ, which is randomly chosen according to a zero-mean Gaussian distribution with variance
    σ = μ - 1 >= 0.
    :param n: The number of coherent states prepared by Alice in a single block.
    :param m: The modulation variance.
    :return: The values of the q and p quadratures.
    """

    q = np.random.normal(0, np.sqrt(m - 1), n)  # Q-quadrature Gaussian distribution with μ = 0 and σ^2 = μ - 1
    p = np.random.normal(0, np.sqrt(m - 1), n)  # P-quadrature Gaussian distribution with μ = 0 and σ_^2 = μ - 1

    return q, p


@njit(fastmath=True, parallel=True, cache=True)
def transmit_states(n, q, p, t, h, s):
    """
    The coherent state is sent through an optical fiber, which can be modelled as a thermal-loss channel.
    :param n: The number of coherent states prepared by Alice in a single block.
    :param q: Alice's values of the q quadratures.
    :param p: Alice's values of the p quadratures.
    :param t: The channel transmittance.
    :param h: The setup efficiency.
    :param s: The variance of the Gaussian noise variable.
    :return: Bob's values of the q and p quadratures.
    """

    qB = np.empty(n, dtype=np.float64)
    pB = np.empty(n, dtype=np.float64)
    z = np.random.normal(0, np.sqrt(s), n)  # Total noise term following a centered normal distribution

    # The outcome of the homodyne detector satisfies the input-output formula
    for i in range(n):
        qB[i] = np.sqrt(t * h) * q[i] + z[i]
        pB[i] = np.sqrt(t * h) * p[i] + z[i]

    return qB, pB


@njit(fastmath=True, parallel=True, cache=True)
def measure_states(n, q, p):
    """
    Bob measures the incoming state using a homodyne detector, which is randomly switched between the two quadratures.
    :param n: The number of coherent states prepared by Alice in a single block.
    :param q: Bob's values of the q quadratures.
    :param p: Bob's values of the p quadratures.
    :return: Bob's measured quadratures and their corresponding values.
    """

    meas_q = np.empty(n, dtype=np.uint8)  # Measured quadratures
    vals_q = np.empty(n, dtype=np.float32)  # Saved values from the measured quadratures
    # A random-number generator (RNG) is used to select the phase of the local oscillator: 0 to measure q or pi/2 for p.
    # Independently generated random value that determines Bob's measured quadrature. 0: Q quadrature, 1: P quadrature.
    b = np.random.randint(low=0, high=2, size=n)
    for i in range(n):
        if b[i] == 0:
            meas_q[i] = 0
            vals_q[i] = q[i]
        else:
            meas_q[i] = 1
            vals_q[i] = p[i]
    return meas_q, vals_q


@njit(fastmath=True, cache=True)
def key_sifting(n, q, p, y):
    """
    Alice and Bob perform a sifting stage where Bob classically communicates to Alice which quadrature he has measured,
    so that the other quadrature is discarded.
    :param n: The number of instances of a single block.
    :param q: Alice's values for the q quadrature.
    :param p: Alice's values for the p quadrature.
    :param y: Bob's randomly chosen quadratures (q or p).
    :return: Alice's variable with a single-chosen quadrature.
    """

    x = np.empty(n, dtype=np.float32)  # Alice's measured quadratures
    # Alice keeps only the value of the quadrature that Bob measured
    for i in range(n):
        if y[i] == 0:
            x[i] = q[i]
        else:
            x[i] = p[i]
    return x


def key_rate_calculation(m, t, h, e, v, bt):
    """
    Calculates the secret key rate, after computing the mutual information and Holevo bound. The analysis follows the
    entanglement-based representation of the protocol.
    :param m: Alice's modulation variance.
    :param t: The channel transmissivity.
    :param h: The setup efficiency.
    :param e: The excess noise.
    :param v: The electronic noise.
    :param bt: The reconciliation efficiency.
    :return i: The mutual information.
    :return x: The Holevo bound, which is the maximum amount of information that Eve can steal in a collective attack.
    :return r: The secret key rate.
    """

    # Calculate the thermal noise
    omega = (t * e - t + 1) / (1 - t)

    # Calculate the mutual information
    s_y = t * h * (m - 1 + e) + v + 1  # Variance of Gaussian variable y
    s_xy = h * t * e + v + 1  # Conditional variance between x and y
    i = 0.5 * np.log2(s_y / s_xy)

    # Define the global output state parameters
    # c = np.sqrt(t * hd * (vA ** 2 - 1))
    b = t * h * (m + e) + 1 - (t * h) + v
    gamma = np.sqrt(h * (1 - t) * (omega ** 2 - 1))
    # delta = -np.sqrt((1 - t) * (vA ** 2 - 1))
    theta = np.sqrt(h * t * (1 - t)) * (omega - m)
    psi = np.sqrt(t * (omega ** 2 - 1))
    phi = t * omega + (1 - t) * m

    # The global output state ρA′BeE′ of Alice, Bob and Eve is zero-mean Gaussian with CM V_A'BeE'
    # To compute the Holevo bound, we need to derive the von Neumann entropies S(ρeE′) and S(ρeE′|y) which can be
    # computed from the symplectic spectra of the reduced CM VeE′ and the conditional CM VeE′|y
    v_eE = np.array([[omega, 0,     psi,    0],
                     [0,     omega, 0,      -psi],
                     [psi,   0,     phi,    0],
                     [0,    -psi,   0,      phi]])
    v_eEy = v_eE - (b ** -1) * np.array([[gamma ** 2,    0,     gamma * theta, 0],
                                             [0,             0,     0,             0],
                                             [gamma * theta, 0,     theta ** 2,    0],
                                             [0,             0,     0,             0]])

    v_1, v_2 = maths.symplectic_eigenvalue_calculation(v_eE)
    v_3, v_4 = maths.symplectic_eigenvalue_calculation(v_eEy)

    x = maths.von_neumann_entropy(v_1) + maths.von_neumann_entropy(v_2) - maths.von_neumann_entropy(v_3) - maths.von_neumann_entropy(v_4)  # Holevo bound
    r = bt * i - x  # Secret key rate

    return i, x, r


def sacrificed_states_selection(n_bks, n, m, M, x, y):
    """

    :param n_bks: The number of blocks.
    :param n: The length of each block.
    :param m: The parameter estimation instances per block.
    :param M: The total number of parameter estimation instances.
    :param x: Alice's local variable.
    :param y: Bob's local variable.
    :return: Alice's and Bob's key generation points and parameter estimation points.
    """

    # Alice and Bob declare m random instances {xi} and {yi} of their local variables x and y.
    x_PE = np.empty(shape=(n_bks, m), dtype=np.float64)
    y_PE = np.empty(shape=(n_bks, m), dtype=np.float64)
    x_key = np.empty(shape=(n_bks, n), dtype=np.float64)
    y_key = np.empty(shape=(n_bks, n), dtype=np.float64)
    tempX = np.copy(x)
    tempY = np.copy(y)

    # The chosen instances for parameter estimation are sacrificed and will not be used for key generation
    # On average m = M / n_bks positions are picked in each block, where n = N − m points remain for key
    blk = 0
    while blk < n_bks:
        # Alice randomly picks m positions from each block
        # Using a classical channel, Alice communicates the M pairs to Bob
        seed = np.random.randint(1000)  # Shuffle both sequences in the same way
        np.random.seed(seed)
        np.random.shuffle(tempX[blk])
        np.random.seed(seed)
        np.random.shuffle(tempY[blk])
        x_PE[blk] = tempX[blk][:m]
        y_PE[blk] = tempY[blk][:m]
        x_key[blk] = tempX[blk][m:]
        y_key[blk] = tempY[blk][m:]
        blk = blk + 1

    # Ensure the number of states selected is the same as the chosen M
    assert int(x_PE.size) == M
    assert int(y_PE.size) == M
    assert int(x_key.size) == n * n_bks
    assert int(y_key.size) == n * n_bks

    return x_key, y_key, x_PE, y_PE


def parameter_estimation(vA, x, y, t, h, v_x, v, m, s, e_pe, alt):
    """
    The two parties perform parameter estimation to estimate the values of the channel transmittance and the excess
    noise. Theoretical and practical worst-case estimators are assumed as well.
    :param vA: Alice's modulation variance.
    :param x: Alice's chosen values for parameter estimation.
    :param y: Bob's chosen values for parameter estimation.
    :param t: The actual channel transmissivity.
    :param h: The setup efficiency.
    :param v_x: The variance of the excess noise.
    :param v: The electronic noise.
    :param m: The parameter estimation instances per block.
    :param s: The variance of the Gaussian noise variable.
    :param e_pe: The probability that the estimated parameters do not belong in the confidence region.
    :param alt: The type of the parameter estimation method (True for Usenko's method, False for Leverrier's)
    :return T_h: The estimator for the channel transmissivity.
    :return x_h: The estimator for the excess noise.
    :return T_m: The worst-case estimator for the channel transmissivity.
    :return x_m: The worst-case estimator for the excess noise.
    :return T_star: The theoretical worst-case estimator for the channel transmissivity.
    :return x_star: The theoretical worst-case estimator for the excess noise.
    """

    # Alice and Bob build the maximum-likelihood estimator (MLE) of the square root transmissivity and noise variance
    t_hat = np.sum(x * y) / np.sum(x ** 2)  # Maximum-likelihood estimator of the square root transmissivity
    s_hat = (1 / m) * np.sum((y - (t_hat * x)) ** 2)  # Maximum-likelihood estimator of the noise variance

    if t_hat > 1 and math.isclose(t_hat, 1, abs_tol=0.01):
        t_hat = 0.999999
    elif t_hat <= 0 or t_hat > 1:
        raise RuntimeWarning("The square root transmissivity must be a number between 0 and 1 (current:", t_hat, ").")
    if s_hat < 1:
        raise RuntimeWarning("The noise variance must be greater than one (current:", s_hat, ").")

    # Calculate the variance of the MLE
    if alt:
        var_t = np.sqrt(((2 * (t_hat ** 2)) / m) + (s_hat / (m * (vA - 1))))  # Usenko's method
    else:
        var_t = np.sqrt(s_hat / (m * (vA - 1)))  # Leverrier's method

    # For m sufficiently large we have that, up to an error probability, the channel parameters falls in the intervals
    w = np.sqrt(2) * erfinv(1 - e_pe)
    D_t = w * var_t
    D_s = w * s_hat * np.sqrt(2) / np.sqrt(m)

    if np.sqrt(t * h) < t_hat - D_t or np.sqrt(t * h) > t_hat + D_t:
        warnings.warn("Root of τη does not fall under the confidence interval.")
    if s < s_hat - D_s or s > s_hat + D_s:
        warnings.warn("Noise variance σ_z does not fall under the confidence interval.")

    # Because the parties perfectly know the values of the detector/setup efficiency and the electronic noise, they may
    # derive the estimators for the transmissivity and excess noise
    T_hat = (t_hat ** 2) / h
    if T_hat > 1 and math.isclose(T_hat, 1, abs_tol=0.01):
        T_hat = 0.999999
    X_hat = s_hat - 1 - v

    if T_hat <= 0 or T_hat > 1:
        raise RuntimeWarning("The estimated transmissivity must be a number between 0 and 1 (current:", T_hat, ").")
    if X_hat < 0:
        raise RuntimeWarning("The estimated excess noise variance must be positive or zero (current:", X_hat, ").")

    # Then they may assume the worst case estimators
    T_m = ((t_hat - D_t) ** 2) / h
    X_m = X_hat + D_s

    if t < T_m:  # Up to an error ε
        warnings.warn("The actual transmissivity is smaller than the worst-case estimator.")
    elif v_x > X_m:
        warnings.warn("The excess noise variance is larger than the worst-case estimator.")

    # Calculate the theoretical worst-case estimators, which are derived from the actual values
    if alt:
        T_star_m = t - (np.sqrt(((4 / m) * (t ** 2)) * (2 + (s / (h * t * (vA - 1))))) * w)  # Usenko
    else:
        T_star_m = (np.sqrt(t) - w * np.sqrt(s / (m * h * (vA - 1)))) ** 2  # Leverrier
    X_star_m = v_x + w * s * np.sqrt(2 / m)

    # Obtain the excess noises from the respective variances of excess noise
    x_hat = X_hat / (h * T_hat)
    x_m = X_m / (h * T_m)
    x_star_m = X_star_m / (h * T_star_m)

    if x_hat < 0:
        raise RuntimeError("Estimated excess noise is negative.")
    if x_m < 0:
        raise RuntimeError("Worst-case scenario excess noise is negative.")
    print("Parameter estimation stage is complete.")

    return T_hat, x_hat, T_m, x_m, T_star_m, x_star_m


def code_estimations(m, x, y, t, h, v, e):
    """
    Computes the signal-to-noise ratio using the estimators from the parameter estimation stage, as well as the
    theoretical and practical correlation coefficients (using the signal-to-noise ratio and the data respectively).
    :param m: The modulation variance.
    :param x: Alice's key generation variable.
    :param y: Bob's key generation variable.
    :param t: The channel transmissivity.
    :param h: The setup efficiency.
    :param v: The electronic noise.
    :param e: The excess noise.
    :return: The estimated signal-to-noise ratio and the theoretical and practical correlation coefficients.
    """

    # Calculate the signal-to-noise ratio using the estimators
    snr = ((m - 1) * h * t) / (1 + v + (h * t * e))

    # Calculate the theoretical and practical correlation coefficients
    r = np.mean(np.multiply(x, y))
    r_th = np.sqrt(snr / (1 + snr))

    # The theoretical and practical correlation coefficients must be somewhat close
    # np.testing.assert_almost_equal(r, r_th, decimal=1)
    return snr, r, r_th
