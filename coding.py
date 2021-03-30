# coding.py
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

import galois
import utilities
import numpy as np
from numba import njit, prange
from numba.typed import Dict


@njit(fastmath=True, parallel=False, cache=True)
def a_priori_probabilities(x, k_ubar, f, rho, a, p, dlt, d):
    """
    Calculates the initial likelihoods for every possible value of a Galois Field before any error-correcting iteration
    takes place.
    :param x: The discretized sequence.
    :param k_ubar: The weakly-correlated digits sequence.
    :param f: The Galois Field.
    :param rho: The correlation coefficient.
    :param a: The cut-off parameter for discretization.
    :param p: The number of bins exponent.
    :param dlt: The constant-size interval divider.
    :param d: The weakly-correlated digits.
    :return: The a priori probabilities of the block.
    """

    pr = []
    for i in prange(len(x)):
        cond_prob = np.empty(len(f), dtype=np.float64)
        for k in prange(len(f)):
            cond_prob[k] = utilities.conditional_probability(x[i], k * 2 ** d + k_ubar[i], rho, a, p, dlt)
        pr.append(cond_prob / np.sum(cond_prob))
    return pr


@njit(cache=True)
def q_ary_syndrome_calculation(c, m, gf_add, gf_mul, vals, rows):
    """
    Returns the q-ary nonzero syndrome of a codeword with a parity-check matrix. The syndrome is found using precomputed
    tables for GF(2^q) in a quick manner.
    :param c: The codeword to be encoded.
    :param m: The number of rows of the parity-check matrix.
    :param gf_add: The precomputed addition array of the given Galois field.
    :param gf_mul: The precomputed multiplication array of the given Galois field.
    :param vals: The dictionary of non-zero values of the parity-check matrix.
    :param rows: The indices of the nonzero values for every row of the parity-check matrix.
    :return: The q-ary syndrome of the codeword.
    """

    s = galois.dot_product(c, m, gf_add, gf_mul, vals, rows)
    return s


def q_ary_decode(n, m, sB, iter_max, f, p, rows, cols, cols_exc, tA, tM, ps_top, pr_top, ps_bottom, pr_bottom, vals,
                 rmn_i1, rmn_i2, rmn_i3):
    """
    Performs the sum-Product algorithm given a set of a priori probabilities to produce a q-ary codeword.
    :param n: The block length of the parity-check matrix.
    :param m: The column length of the parity-check matrix.
    :param sB: Bob's syndrome to be evaluated with Alice's syndrome.
    :param iter_max: The maximum number of iterations the belief propagation algorithm will run for.
    :param f: The Galois Field of the computations.
    :param p: The a priori probabilities for the initialization step.
    :param vals: The nonzero values of the parity-check matrix.
    :param rows: The check nodes of the parity-check matrix.
    :param cols: The variable nodes of the parity-check matrix.
    :param cols_exc: The check nodes of the parity-check matrix, excluding the current node for every node choice.
    :param tA: The Galois field precomputed addition table.
    :param tM: The Galois field precomputed multiplication table.
    :param ps_top: The first set of partial sum indices.
    :param pr_top: The first set of partial sum indices.
    :param ps_bottom: The second set of partial sum indices.
    :param pr_bottom: The second set of partial sum indices.
    :param rmn_i1: The first set of partial sum indices.
    :param rmn_i2: The second set of partial sum indices.
    :param rmn_i3: The third set of partial sum indices.
    :return: Alice's codeword and its found round. If no syndrome was matched, the last estimation is returned.
    """

    # For a high SNR, specific parameters and a small block length, there is a chance that the sum-product algorithm is
    # not needed. In such a case, the syndromes are compared before the start of the algorithm.
    if n < 1000:
        x = np.empty(shape=n, dtype=np.int16)
        for i in prange(n):
            x[i] = np.argmax(p[i])
        sA = q_ary_syndrome_calculation(x, m, tA, tM, vals, rows)
        if (sA == sB).all():
            return x, True, 0

    # Step 1: Initialization
    q, q_mn, r_mn, p_s, p_r = initialization(m, p, rows, f)

    for it in prange(1, iter_max + 1):

        # Step 2: Horizontal Step (Update r)
        r_mn = update_r_mn(m, f, q, rows, r_mn, p_s, p_r, rmn_i1, rmn_i2, rmn_i3, ps_top, pr_top, ps_bottom, pr_bottom)

        # Step 3: Vertical Step (Update q)
        q = update_q_mn(n, f, r_mn, cols, cols_exc, q_mn, p)

        # Step 4: Tentative Decoding
        x = tentative_decoding(n, f, r_mn, p, cols)

        sA = q_ary_syndrome_calculation(x, m, tA, tM, vals, rows)

        if (sA == sB).all():
            return x, True, it

        if it == iter_max:  # If this point is reached, the algorithm has failed to converge
            return x, False, it


@njit(fastmath=True, parallel=False, cache=True)
def initialization(m, p, rows, f):
    Q = Dict()
    q_mn = Dict()
    r_mn = Dict()
    p_s = Dict()
    p_r = Dict()
    for i in prange(0, m):
        for j in prange(0, len(rows[i])):
            row = rows[i][j]  # Speed upgrade
            Q[(i, row)] = p[row]
            q_mn[(i, row)] = np.zeros(f, dtype=np.float32)
            r_mn[(i, row)] = np.zeros(f, dtype=np.float32)
            p_s[(i, row)] = np.zeros(f, dtype=np.float32)
            p_r[(i, row)] = np.zeros(f, dtype=np.float32)

    return Q, q_mn, r_mn, p_s, p_r


@njit(fastmath=True, parallel=True, cache=True)
def update_r_mn(m, f, Q, rows, r_mn, p_s, p_r, r_ind1, r_ind2, r_ind3, ps_ind, pr_ind, sec_ps, sec_pr):
    for i in prange(0, m):
        row = rows[i]
        len_row = len(row)
        for a in prange(f):
            p_s[(i, row[0])][a] = Q[(i, row[0])][ps_ind[(i, 0, a)]]
            p_r[(i, row[len_row - 1])][a] = Q[(i, row[len_row - 1])][pr_ind[(i, len_row - 1, a)]]

        for j in prange(1, len_row - 1):
            k = len_row - 1 - j
            i1 = row[j]  # Do the list searches now to avoid performing them repeatedly
            i2 = row[k]
            a1 = p_s[(i, row[j - 1])]  # Do the dictionary searches now to avoid performing them repeatedly
            a2 = p_r[(i, row[k + 1])]
            q1 = Q[(i, i1)]
            q2 = Q[(i, i2)]
            for a in prange(f):
                SUMS = 0
                SUMR = 0
                for t in prange(f):
                    SUMS += a1[sec_ps[(i, j, a, t)]] * q1[t]
                    SUMR += a2[sec_pr[(i, k, a, t)]] * q2[t]
                p_s[(i, i1)][a] = SUMS
                p_r[(i, i2)][a] = SUMR

        pr_i = p_r[(i, row[1])]
        ps_i = p_s[(i, row[len_row - 2])]
        for a in prange(f):
            r_mn[(i, row[0])][a] = pr_i[r_ind1[(i, 0, a)]]
            r_mn[(i, row[len_row - 1])][a] = ps_i[r_ind2[(i, len_row - 1, a)]]
            for j in prange(1, len_row - 1):
                i1 = row[j]
                b1 = p_s[(i, row[j - 1])]
                b2 = p_r[(i, row[j + 1])]
                Pr = 0
                for s in prange(f):
                    Pr += b1[s] * b2[r_ind3[(i, j, a, s)]]
                r_mn[(i, i1)][a] = Pr

    return r_mn


@njit(fastmath=True, parallel=False, cache=True)
def update_q_mn(n, f, r, cols, cols_exc, q, p):
    a_mn = np.empty(f, dtype=np.float32)  # Vector used in the q_mn calculations
    for i in range(0, n):
        for j in range(0, len(cols[i])):
            col = cols_exc[i][j]  # Save iteration speed by looking list items up only once
            for a in prange(f):
                a_mn[a] = p[i][a]
                for m in range(0, len(cols_exc[i][j])):
                    a_mn[a] = a_mn[a] * r[(col[m], i)][a]
            q[(cols[i][j], i)] = a_mn / np.sum(a_mn)
    return q


@njit(fastmath=True, parallel=True, cache=True)
def tentative_decoding(n, f, r, p, cols):
    prob = np.empty(shape=(n, f), dtype=np.float32)
    c = np.empty(n, dtype=np.int16)
    for i in prange(0, n):
        for a in prange(f):
            R = p[i][a]
            for k in prange(0, len(cols[i])):
                R = R * r[(cols[i][k], i)][a]
            prob[i][a] = R
        c[i] = np.argmax(prob[i])
    return c


@njit(fastmath=True, parallel=False, cache=True)
def get_partial_sums_indices(m, f, H, rows, tA, tM):
    """
    Precompute the indices for the partial sum probabilities.
    :param m: The number of rows of the parity-check matrix.
    :param f: The Galois field.
    :param H: The parity-check matrix.
    :param rows: The indices of the nonzero values for every row of the parity-check matrix.
    :param tA: The Galois field precomputed addition table.
    :param tM: The Galois field precomputed multiplication table.
    :return: The indices of the partial sums for the non-binary decoding process.
    """

    ps_first = Dict()
    pr_first = Dict()
    ps_rest = Dict()
    pr_rest = Dict()

    for i in prange(0, m):
        row_i = rows[i]  # Speed upgrade
        k = len(row_i) - 1
        for a in range(0, f):
            for t in range(0, f):
                if tM[H[i][row_i[0]], t] == a:
                    ps_first[(i, 0, a)] = t
                    break
            for t in range(f):
                if tM[H[i][row_i[k]], t] == a:
                    pr_first[(i, k, a)] = t
                    break
        for j in range(1, len(rows[i]) - 1):
            k = len(row_i) - 1 - j
            row_j = row_i[j]  # Speed upgrade
            row_k = row_i[k]  # Speed upgrade
            for a in range(f):
                for t in range(f):
                    for s in range(f):
                        if tA[tM[H[i][row_j], t], s] == a:
                            ps_rest[(i, j, a, t)] = s
                            break
                    for s in range(f):
                        if tA[tM[H[i][row_k], t], s] == a:
                            pr_rest[(i, k, a, t)] = s
                            break
    return ps_first, pr_first, ps_rest, pr_rest


@njit(fastmath=True, parallel=False, cache=True)
def get_rmn_indices(m, f, H, z, rows, tA, tM):
    """
    Precompute the indices to be used during the r_mn stage.
    :param m: The number of rows of the parity-check matrix.
    :param f: The Galois field.
    :param H: The parity-check matrix.
    :param z: The syndrome sent in the clear.
    :param rows: The indices of the nonzero values for every row of the parity-check matrix.
    :param tA: The Galois field precomputed addition table.
    :param tM: The Galois field precomputed multiplication table.
    :return: The indices of the partial sums for the non-binary decoding process.
    """

    r_ind_1 = Dict()
    r_ind_2 = Dict()
    r_ind_3 = Dict()

    for i in prange(0, m):
        rows_i = rows[i]  # Speed upgrade
        for a in prange(f):
            for t in prange(f):
                if t == tA[z[i], tM[H[i, rows_i[0]], a]]:
                    r_ind_1[(i, 0, a)] = t
                    break
            for s in prange(f):
                if s == tA[z[i], tM[H[i, rows_i[len(rows_i) - 1]], a]]:
                    r_ind_2[(i, len(rows_i) - 1, a)] = s
                    break
            for j in prange(1, len(rows[i]) - 1):
                rows_ij = rows_i[j]  # Speed upgrade
                for s in prange(f):
                    for t in prange(f):
                        if tA[t, s] == tA[z[i], tM[H[i, rows_ij], a]]:
                            r_ind_3[(i, j, a, s)] = t
                            break

    return r_ind_1, r_ind_2, r_ind_3
