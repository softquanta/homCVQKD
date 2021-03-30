# galois.py
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
from numba import njit


@njit(fastmath=False, parallel=False, cache=True)
def dot_product(v, m, tA, tM, vals, rows):
    """
    Calculates the dot product of a (m x n) sparse array and a n-length vector in a fast manner using precomputed
    multiplication and addition tables of a specified Galois Field.
    :param v: The n-length vector.
    :param m: The number of rows of the array.
    :param tA: The precomputed addition table of the Galois Field.
    :param tM: The precomputed multiplication table of the Galois Field.
    :param vals: The nonzero values of the sparse array.
    :param rows: The indices of the nonzero values for every row of the array.
    :return: The size m dot product of the sparse array and the vector.
    """

    s = np.zeros(m, dtype=np.int8)  # The highest possible value of np.int8 is 127, which allows a maximum GF of (2^7)
    for i in range(0, m):
        for j in range(0, len(rows[i])):
            mul = tM[vals[(i, rows[i][j])]][v[rows[i][j]]]  # Multiplication step
            s[i] = tA[s[i]][mul]  # Addition step
    return s


@njit(cache=True)
def precomputed_addition_table(f):
    """
    Returns the lookup table for addition computations in a specified Galois Field.
    :param f: The Galois Field.
    :return: The precomputed addition table for the specified Galois Field.
    """

    if f == 8:  # GF(2^3)
        table = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                          [1, 0, 3, 2, 5, 4, 7, 6],
                          [2, 3, 0, 1, 6, 7, 4, 5],
                          [3, 2, 1, 0, 7, 6, 5, 4],
                          [4, 5, 6, 7, 0, 1, 2, 3],
                          [5, 4, 7, 6, 1, 0, 3, 2],
                          [6, 7, 4, 5, 2, 3, 0, 1],
                          [7, 6, 5, 4, 3, 2, 1, 0]])
    elif f == 16:  # GF(2^4)
        table = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                          [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14],
                          [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13],
                          [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12],
                          [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11],
                          [5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10],
                          [6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9],
                          [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8],
                          [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],
                          [9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6],
                          [10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5],
                          [11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4],
                          [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3],
                          [13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2],
                          [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1],
                          [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    else:
        raise RuntimeWarning("This Galois field is not currently supported.")
    return table


@njit(cache=True)
def precomputed_multiplication_table(f):
    """
    Returns the lookup table for multiplication computations in a specified Galois Field.
    :param f: The Galois Field.
    :return: The precomputed multiplication table for the specified Galois Field.
    """

    if f == 8:  # GF(2^3)
        table = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 2, 3, 4, 5, 6, 7],
                          [0, 2, 4, 6, 3, 1, 7, 5],
                          [0, 3, 6, 5, 7, 4, 1, 2],
                          [0, 4, 3, 7, 6, 2, 5, 1],
                          [0, 5, 1, 4, 2, 7, 3, 6],
                          [0, 6, 7, 1, 5, 3, 2, 4],
                          [0, 7, 5, 2, 1, 6, 4, 3]])

    elif f == 16:  # GF(2^4)
        table = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                          [0, 2, 4, 6, 8, 10, 12, 14, 3, 1, 7, 5, 11, 9, 15, 13],
                          [0, 3, 6, 5, 12, 15, 10, 9, 11, 8, 13, 14, 7, 4, 1, 2],
                          [0, 4, 8, 12, 3, 7, 11, 15, 6, 2, 14, 10, 5, 1, 13, 9],
                          [0, 5, 10, 15, 7, 2, 13, 8, 14, 11, 4, 1, 9, 12, 3, 6],
                          [0, 6, 12, 10, 11, 13, 7, 1, 5, 3, 9, 15, 14, 8, 2, 4],
                          [0, 7, 14, 9, 15, 8, 1, 6, 13, 10, 3, 4, 2, 5, 12, 11],
                          [0, 8, 3, 11, 6, 14, 5, 13, 12, 4, 15, 7, 10, 2, 9, 1],
                          [0, 9, 1, 8, 2, 11, 3, 10, 4, 13, 5, 12, 6, 15, 7, 14],
                          [0, 10, 7, 13, 14, 4, 9, 3, 15, 5, 8, 2, 1, 11, 6, 12],
                          [0, 11, 5, 14, 10, 1, 15, 4, 7, 12, 2, 9, 13, 6, 8, 3],
                          [0, 12, 11, 7, 5, 9, 14, 2, 10, 6, 1, 13, 15, 3, 4, 8],
                          [0, 13, 9, 4, 1, 12, 8, 5, 2, 15, 11, 6, 3, 14, 10, 7],
                          [0, 14, 15, 1, 13, 3, 2, 12, 9, 7, 6, 8, 4, 10, 11, 5],
                          [0, 15, 13, 2, 9, 6, 4, 11, 1, 14, 12, 3, 8, 7, 5, 10]])
    else:
        raise RuntimeWarning("This Galois field is not currently supported.")
    return table
