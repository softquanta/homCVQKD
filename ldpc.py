# ldpc.py
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

import timeit
from os import path
import warnings
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import numba as nb
from numba import njit, prange
from numba.typed import List, Dict


def generate_code(n, q, r, load):
    """
    Creates or loads a random regular low-density parity-check (LDPC) code.
    :param n: The number of columns of the regular LDPC code.
    :param q: The Galois field exponent.
    :param r: The code rate.
    :param load: Determines whether the LDPC code is loaded from disk or a new code is created.
    :return: The regular LDPC code in its dense and sparse form and the dictionary of its values.
    """

    wc = 2  # Column weight of the low-density parity-check matrix (usually 2 <= wc <= 4)
    wr = int(np.round(wc / (1 - r)))  # Row weight of the low-density parity-check matrix
    m = int(np.round(n * wc / wr))  # Number of rows of the low-density parity-check matrix
    k = n - m  # Information bits of the low-density parity-check matrix
    r_design = 1 - wc / wr  # The true rate of the code, which should be very close to the code rate from the data
    print("Code Rate R_code:", r, "Design Rate R_des:", r_design)
    np.testing.assert_almost_equal(r_design, r, decimal=1, err_msg="The error between the LDPC code rate and the "
                                                                   "code rate from the data is too high.")
    if load:
        filename = 'codes/' + str(n) + "-" + str(r_design)[0:5] + "-" + str(wc) + "-" + str(wr) + '.npz'
        # Check if the file name used matches the characteristics for the low-density parity-check code
        vals = filename.split("-")
        if int(vals[0].replace("codes/", "")) != n:
            raise RuntimeWarning("The column number specified is not the same as the column number of the loaded code.")
        elif vals[1] != str(r_design)[0:5]:
            raise RuntimeWarning("The code rate of the data is not the same as the rate of the loaded code.")
        elif int(vals[2]) != wc:
            raise RuntimeWarning("The column weight specified is not the same as the column weight of the loaded code.")
        elif int(vals[3].replace(".npz", "")) != wr:
            raise RuntimeWarning("The row weight of the data is not the same as the row weight of the loaded code.")
        else:
            try:
                code_sparse = load_npz(filename)
                print("The following LDPC parity check matrix was successfully loaded from disk:", filename)
            except (FileNotFoundError, IOError):
                raise FileNotFoundError("The file", filename, "does not exist. A simulation with the given parameters "
                                                              "must be first run in order to create the code array.")
            except ValueError:
                raise ValueError("Pickled=false error, need to fix")
            code = code_sparse.toarray()
            m = code.shape[0]
            vals = get_values(m, code_sparse)
    else:
        print("Creating a new LDPC code of size", m, "x", n, "with column weight", wc, "and row weight", wr, "...")
        code, vals = create_random_regular_code(n, m, wc, wr, q)
        code_sparse = csr_matrix(code, dtype=np.uint8)
        if path.exists('codes/' + str(n) + "-" + str(r_design)[0:5] + "-" + str(wc) + "-" + str(wr) + '.npz'):
            warnings.warn("An LDPC code with the specified specs already exists. A new one was still created.")
            save_npz('codes/' + str(n) + "-" + str(r_design)[0:5] + "-" + str(wc) + "-" + str(wr) + '-new.npz',
                     code_sparse, compressed=True)
        else:
            save_npz('codes/' + str(n) + "-" + str(r_design)[0:5] + "-" + str(wc) + "-" + str(wr) + '.npz', code_sparse,
                     compressed=True)

    return k, m, code, code_sparse, vals, r_design, wc, wr


@njit(fastmath=True, parallel=False, cache=True)
def set_ldpc_values(h, m, n, q):
    """
    Replaces the nonzero units of an array with random values from a chosen Galois field.
    :param h: The LDPC matrix.
    :param m: The number of rows of the LDPC matrix.
    :param n: The number of columns of the LDPC matrix.
    :param q: The Galois Field exponent.
    :return: The LDPC code array whose nonzero values belong to a Galois Field and the dictionary of these values.
    """

    v = Dict()
    for i in range(0, m):
        for j in range(0, n):
            if h[i][j] != 0:
                h[i][j] = np.random.randint(low=1, high=2 ** q)
                v[(i, j)] = h[i][j]

    return h, v


@njit(fastmath=True, cache=True)
def check_matrix_rank(h, m, n):
    """
    Ensures that the LDPC code array has full rank. If the array does not have full rank, its true code rate is shown.
    :param h: The LDPC matrix.
    :param m: The number of rows of the LDPC matrix.
    :param n: The number of columns of the LDPC matrix.
    """
    rank_h = np.linalg.matrix_rank(h.astype(np.float32))  # Input required to be in float format
    if m < n:
        if rank_h == h.shape[0]:
            print("The matrix has full rank.")
        else:
            print("Warning: The matrix does not have full rank. The code rate is R_code =", (n - rank_h) / n)
    else:
        if rank_h == h.shape[1]:
            print("The matrix has full rank.")
        else:
            print("Warning: The matrix does not have full rank.")


@njit(fastmath=True, parallel=False, cache=True)
def check_column_overlap(h, n):
    """
    Checks if the overlap (inner product) of two consecutive columns of an LDPC code is larger than one and reports the
    columns that have this trait.
    :param h: The LDPC matrix.
    :param n: The number of columns of the LDPC matrix.
    """
    hT_float = np.ascontiguousarray(h.T.astype(np.float32))
    for i in prange(n - 1):
        h1 = hT_float[i]
        h2 = hT_float[i + 1]
        dot = np.dot(h1, h2)
        if dot > 1.0:
            print("Warning: Inner product larger than one found between columns", i, "and", i + 1, "(", dot, ")")


@njit(fastmath=True, parallel=False, cache=True)
def check_row_weights(h, m, r):
    """
    Checks if the rows of the an LDPC code have the specified column weight and reports the rows that deviate from it.
    :param h: The LDPC matrix.
    :param m: The number of rows of the LDPC matrix.
    :param r: The specified row weight.
    """

    row_error = 0
    for i in prange(m):
        if np.count_nonzero(h[i]) != r:
            row_error = row_error + 1
            print("Row weight error in row", i, "- has", np.count_nonzero(h[i]), "bits")
    if row_error == 0:
        print("No row weight error found.")
    else:
        print("Row count with weight error:", row_error)


@njit(fastmath=True, parallel=False, cache=True)
def check_column_weights(h, n, c):
    """
    Checks if the columns of an LDPC code have the specified column weight and reports the columns that deviate from it.
    :param h: The LDPC matrix.
    :param n: The number of columns of the LDPC matrix.
    :param c: The specified column weight.
    """

    col_error = 0
    for i in prange(n):
        if np.count_nonzero(h.T[i]) != c:
            col_error = col_error + 1
            print("Column weight error in row", i, "- has", np.count_nonzero(h.T[i]), "bits")
    if col_error == 0:
        print("No column weight error found.")
    else:
        print("Column count with weight error:", col_error)


def get_values(m, h: csr_matrix):
    """
    Returns the nonzero values of an array, along with their indices. The indices are stored in Numba typed dictionaries
    so that they are able to be interpreted by Numba.
    :param m: The number of rows of the LDPC matrix.
    :param h: The sparse LDPC matrix.
    :return: The dictionary of indices and nonzero values of an array.
    """

    # If the row number is too large, extra memory will be required
    if m < 2 ** 16:
        data_type = np.uint16
        key_type = nb.uint16
    else:
        data_type = np.uint32
        key_type = nb.uint32

    r = h.tocoo().row.astype(data_type)
    c = h.tocoo().col.astype(np.uint32)

    v = Dict.empty(key_type=nb.types.Tuple((key_type, nb.uint32)), value_type=nb.types.uint8)
    for i in range(len(r)):
        v[(r[i], c[i])] = h[r[i], c[i]]

    return v


# @njit()
def get_dict_nodes(vals, rows_exc, cols_exc, c_lil, c_lil_t):

    for key in vals:
        rows_exc[key] = np.array([n for n in c_lil[key[0]] if n != key[1]], dtype=np.int32)
        cols_exc[key] = np.array([n for n in c_lil_t[key[1]] if n != key[0]], dtype=np.int32)

    return rows_exc, cols_exc


def get_nodes(n, m, h: csr_matrix, ext):
    """
    Gets the nonzero row and column indices of a sparse array. The indices are stored in Numba typed lists so that they
    are able to be interpreted by Numba.
    :param n: The number of columns of the LDPC matrix.
    :param m: The number of rows of the LDPC matrix.
    :param h: The sparse LDPC matrix.
    :param ext:
    :return: The variable and check nodes of the LDPC matrix.
    """

    rows = List()
    cols = List()
    cols_exc = List()

    # Convert the sparse matrix from a csr form to others to quickly obtain the necessary values
    c_lil = h.tolil().astype(dtype=np.uint8).rows
    c_lil_t = h.transpose().tolil().astype(dtype=np.uint8).rows

    # Get the indices of CN-to-VN messages
    if not ext:
        for r in range(m):  # For every row of the VN-to-CN messages array
            rows.append(List(c_lil[r]))  # Get the VNs connected to a certain CN
    else:
        rows_exc = List()
        for r in range(m):  # For every row of the VN-to-CN messages array
            rows.append(List(c_lil[r]))  # Get the VNs connected to a certain CN
            lst = List()
            for j in range(len(rows[r])):
                y = rows[r][:]
                y.remove(rows[r][j])
                lst.append(y)
            rows_exc.append(lst)

    # Get the indices of VN-to-CN messages and the indices of VN-to-CN messages, excluding the current VN
    for c in range(n):
        cols.append(List(c_lil_t[c]))
        lst = List()
        for j in range(len(cols[c])):
            y = cols[c][:]
            y.remove(cols[c][j])
            lst.append(y)
        cols_exc.append(lst)

    if not ext:
        return rows, cols, cols_exc
    else:
        return rows, rows_exc, cols, cols_exc


@njit(fastmath=True, parallel=False, cache=True)
def create_random_regular_code(n, m, c, r, q):
    """
    Low-density parity-check (LDPC) codes can be specified by a non-systematic sparse parity-check matrix H, having a
    uniform column weight and a uniform row weight. H is constructed at random to these constraints. A (n,c,r) LDPC code
    is specified by a parity-check matrix H having m rows and n columns, with r 1's per row and c 1's per column.
    The code formed from such a parity check matrix is known as a regular Gallagher code.
    :param n: The code block length (number of columns of H).
    :param m: The number of rows of the LDPC code.
    :param c: The column weight of H (number of non zeros per column).
    :param r: The row weight of H (number of non zeros per row).
    :param q: The Galois field exponent.
    :return: The LDPC matrix along with its values dictionary.
    """

    # Step 0: Validity checks
    if n <= r:  # n must be larger than r
        raise ValueError("The number of rows of an LDPC code must always be smaller than its number of columns.")
    if r < 2:  # r must be at least 2
        raise ValueError("The row weight of an LDPC code must be at least 2.")
    if c < 2:
        raise ValueError("The column weight of an LDPC code must be at least 2.")

    # Step 1: An all-zero matrix H of dimension (m x n) is created.
    h = np.zeros((m, n), dtype=np.uint8)

    # Step 2: For each column in H, c 1s are placed in rows chosen at random.
    for i in prange(n):
        cols = np.random.choice(m, c, replace=False)
        h.T[i][cols] = 1

    # Step 3: The software then runs through the matrix searching for a row with zero 1's or just one 1.
    for i in prange(m):
        # If a row has no 1's in it, then it is a redundant row.
        # So the software chooses 2 columns in the same row at random and places 1's in those columns.
        if np.count_nonzero(h[i]) == 0:
            a = np.random.choice(n, 2, replace=False)
            h[i][a[0]] = 1
            h[i][a[1]] = 1
        # If a row just has one 1 in a row it means that the codeword bit in that column is always zero.
        # So whenever the software finds a row with just one 1 in it, it randomly picks another column in the same row
        # and places a 1 there.
        elif np.count_nonzero(h[i]) == 1:
            h[i][np.random.randint(0, n)] = 1

    # Step 4: The software then calculates the number of 1's per row.
    # If this is not an integer, the software rounds the value to the next higher integer.
    threshold = int(np.round(r))

    # Check if the code can be regular with the given parameters (only for n <= 10 ** 3 to save time)
    if n <= 10 ** 3:
        if np.count_nonzero(h[:]) % n == 0 and np.count_nonzero(h[:]) % m == 0:
            print("The code can be regular - Total count of bits:", np.count_nonzero(h[:]))
        else:
            print("The code will be irregular - Total count of bits:", np.count_nonzero(h[:]))

    # Note down the rows, whose nonzero elements are below the threshold, to achieve faster computation in Step 5
    rows_below_threshold_list = []
    for row in range(0, m):
        if np.count_nonzero(h[row]) < threshold:
            rows_below_threshold_list.append(row)
    rows_below_threshold = np.array(rows_below_threshold_list, dtype=np.uint32)

    # Step 5: The software then runs through the matrix trying to make the number of 1's per row as uniform as possible.
    for i in range(m):
        # For any row i containing more number of ones than the value calculated in Step 4
        while np.count_nonzero(h[i]) > threshold:
            # print(i, np.count_nonzero(h[i]), rows_below_threshold.size, m)
            # The software picks a column containing a 1 at random and tries to move that 1 to a different row
            # (randomly chosen such that has it a lower number of 1's than the value in step 4) in the same
            # column. The software makes sure that the row chosen does not have a 1 in that particular column.
            non_zeros = np.nonzero(h[i])  # Available columns to choose from
            chosen_column = np.random.choice(non_zeros[0])  # Randomly choose one of the available columns
            if rows_below_threshold.size == 0:
                break
            random_row = np.random.choice(rows_below_threshold)  # Randomly choose one of the saved rows below threshold
            if np.count_nonzero(h[random_row]) <= threshold and h[random_row][chosen_column] == 0:
                h[random_row][chosen_column] = 1
                h[i][chosen_column] = 0
                # If the nonzero elements of the row are now equal to the threshold, remove the row from the list
                if np.count_nonzero(h[random_row]) == threshold:
                    index = np.where(rows_below_threshold == random_row)
                    rows_below_threshold = np.delete(rows_below_threshold, index[0][0])

    # Make the column weight uniform
    for i in range(n):
        while np.count_nonzero(h.T[i]) != c:
            non_zeros = np.nonzero(h.T[i])
            ind = np.random.choice(non_zeros[0])  # Row index to be removed
            h[ind][i] = 0

    # Quality checks (preferably performed only for smaller block lengths to avoid longer generation of the matrix)
    if n < 10 ** 4:
        check_matrix_rank(h, m, n)  # Find if the matrix has full rank
        check_column_overlap(h, n)  # Determine if the overlap between any two columns is greater than 1
        check_row_weights(h, m, r)  # Check the row weight balance
        check_column_weights(h, n, c)  # Check the column weight balance

    # If the LDPC needs to belong to a finite field, replace the 1s with q-ary symbols
    h, v = set_ldpc_values(h, m, n, q)

    return h, v
