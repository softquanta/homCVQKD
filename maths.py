# maths.py
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

import math
import numpy as np
from numba import njit


def convolution_theorem(x, y):
    conv = np.fft.ifft(np.multiply(np.fft.fft(x).astype(np.complex64),
                                   np.fft.fft(y).astype(np.complex64))).astype(np.complex64)
    return conv


def shannon_entropy(x, n_bks, n):
    _, counts_y = np.unique(x.ravel(), return_counts=True)
    p = counts_y / (n_bks * n)  # Appearance probabilities of the discretized values
    h = -np.sum(p * np.log2(p))  # Shannon entropy of the discretized values of the key generation states
    return h


def von_neumann_entropy(v):
    """
    A bosonic entropic function which calculates the Von Neumann entropy.
    :param v: The symplectic eigenvalue.
    :return: The von Neumann entropy.
    """

    return ((v + 1) / 2) * np.log2((v + 1) / 2) - ((v - 1) / 2) * np.log2((v - 1) / 2)


def symplectic_eigenvalue_calculation(V):
    """
    If the matrix V is a 4 × 4 positive-definite matrix, it can be expressed in the block form [[A C], [C^T B]]. In such
    a case, the symplectic spectrum can be calculated by the formula for ν±. Alternatively, the eigenvalues are found
    from the modulus |iΩV|.
    :param V The matrix whose symplectic eigenvalues must be obtained.
    :return The symplectic eigenvalues of V.
    """


    # Check if the given matrix is 4 x 4 and positive definite
    if V.shape == (2, 2):
        iOmega = np.dot(1j, np.array([[0, 1], [-1, 0]]))  # iΩ matrix
        v = np.linalg.eigvals((np.dot(iOmega, V)))
        v_1 = np.abs(v[0])

        assert v_1 > 1
        return v_1

    elif V.shape == (4, 4):
        if np.all(np.linalg.eigvals(V) > 0):
            A_mode = np.array([[V[0, 0], V[0, 1]], [V[1, 0], V[1, 1]]])
            B_mode = np.array([[V[2, 2], V[2, 3]], [V[3, 2], V[3, 3]]])
            C_mode = np.array([[V[0, 2], V[0, 3]], [V[1, 2], V[1, 3]]])
            Delta_V = np.linalg.det(A_mode) + np.linalg.det(B_mode) + 2 * np.linalg.det(C_mode)
            v_1 = np.sqrt((Delta_V + np.sqrt(Delta_V ** 2 - 4 * np.linalg.det(V))) / 2)
            v_2 = np.sqrt((Delta_V - np.sqrt(Delta_V ** 2 - 4 * np.linalg.det(V))) / 2)
        else:
            iOmega = np.dot(1j, np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]))  # iΩ matrix
            v = np.linalg.eigvals((np.dot(iOmega, V)))
            v_1 = np.abs(v[0])
            v_2 = np.abs(v[3])

        # Assert that the eigenvalues are positive and larger than unit
        assert v_1 > 1
        assert v_2 > 1

        return v_1, v_2


@njit(fastmath=True, cache=True)
def conditional_probability(k, i, r, a, p, d):
    """
    Calculates the conditional probability to be used for the calculation of the a priori probabilities.
    :param k: The discretized variable.
    :param i: The value of the bin.
    :param r: The correlation parameter.
    :param a: The discretization cut-off parameter.
    :param p: The number of bins exponent.
    :param d: The constant-size interval divider.
    :return: The conditional probability P(K|X).
    """

    if i == 0:
        ak = -np.inf
        bk = -a + d
    elif i == 2 ** p - 1:
        ak = -a + (2 ** p - 1) * d
        bk = np.inf
    else:
        ak = -a + i * d
        bk = -a + (i + 1) * d

    A = (ak - k * r) / np.sqrt(2 * (1 - r ** 2))
    B = (bk - k * r) / np.sqrt(2 * (1 - r ** 2))
    prob = 0.5 * (math.erf(B) - math.erf(A))

    return prob
