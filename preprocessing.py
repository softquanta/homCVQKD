# preprocessing.py
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
from numba import njit, prange


def normalization(x, y, same):
    """
    Alice and Bob normalize their variables by dividing them either by their respective standard deviations or Alice's
    standard deviation.
    :param x: Alice's sequence of key generation points.
    :param y: Bob's sequence of key generation points.
    :param same: True if the y sequence will be normalised by the standard deviation y, False if done so by x.
    :return: The normalized sequences of x and y.
    """

    s_x = np.std(x)
    s_y = np.std(y)
    x = x / s_x
    if same:
        y = y / s_y
    else:
        y = y / s_x

    return x, y


def rescaling(y, t, h):
    """
    As a first pre-processing step, Bob rescales his variable as y → y' := y / ηT^, so that the channel is written more
    simply.
    :param y: Bob's key generation sequence.
    :param t: The channel transmittance, as estimated by Alice and Bob during parameter estimation.
    :param h: The setup efficiency.
    """

    y_new = y / np.sqrt(h * t)
    return y_new


@njit(fastmath=True, parallel=True, cache=True)
def discretization(y, a, p, d):
    """
    Discretizes the variable in a p-ary variable with generic value {0, ..., 2 ^ p - 1)}.
    :param y: The variable to be discretized.
    :param a: The cut-off parameter.
    :param p: The discretization digits.
    :param d: The constant-size interval divider.
    :return: The integer representation of the interval, under which every value of Alice's and Bob's key falls.
    """

    k = np.empty(len(y), dtype=np.uint16)  # The discretised variable
    for i in prange(len(y)):
        if y[i] < -a + d:  # If the value falls under the leftmost bin
            k[i] = 0
        elif y[i] >= -a + d * (2 ** p - 1):  # If the value falls under the rightmost bin
            k[i] = 2 ** p - 1
        else:
            for j in prange(1, 2 ** p - 1):  # Iterate through every possible intermediate bin and stop when found
                if -a + j * d <= y[i] < -a + (j + 1) * d:
                    k[i] = j
                    break

    return k


@njit(parallel=True, fastmath=True, cache=True)
def splitting(k, d):
    """
    Splits a discretized variable in two parts, where the top variable is q-ary and the bottom variable is d-ary.
    :param k: The discretized variable.
    :param d: The weakly-correlated digits.
    :return: The top and bottom q-ary and d-ary sequences respectively.
    """

    k_top = np.empty(len(k), dtype=np.uint16)
    k_bottom = np.empty(len(k), dtype=np.uint16)
    for i in prange(len(k)):
        k_all = [(k[i] - k[i] % 2 ** d) // (2 ** d), k[i] % 2 ** d]
        k_top[i] = k_all[0]
        k_bottom[i] = k_all[1]

    return k_top, k_bottom


@njit(fastmath=True, parallel=True, cache=True)
def neighbour_concatenation(x1, x2, p):
    x = np.empty(len(x1), dtype=np.uint16)
    for i in prange(len(x1)):
        x[i] = x1[i] * (2 ** p) + x2[i]
    return x
