# utilities.py
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


def q_ary_to_binary(m, q):
    """
    Converts a q-ary sequence into a binary sequence of length q.
    :param m: The q-ary sequence.
    :param q: The Galois field exponent.
    :return: The binary representations of the q-ary sequences.
    """

    mA_bin = np.empty(len(m) * q, dtype=np.int8)  # Binary representation of Alice's q-ary message
    for i in range(len(m)):
        bitsA = np.binary_repr(m[i], width=q)
        for j in range(q):
            mA_bin[i * q + j] = bitsA[j]
    return mA_bin
