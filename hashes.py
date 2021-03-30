# hashes.py
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
import scipy.linalg as sp
import random


def privacy_amplification(s, n, r, mode):
    """
    The sequence of the concatenated correctly decoded blocks is compressed into a final secret key of length of length
    r by applying a Toeplitz matrix.
    :param s: The sequence to be compressed.
    :param n: The privacy amplification block length.
    :param r: The length of the final key.
    :param mode: The privacy amplification method (either FFT or dot product, first is mandatory for large blocks).
    :return: The secret key.
    """

    col = np.array(np.random.choice(2, r)).astype(np.uint8)  # First column of the normal Toeplitz matrix
    row = np.array(np.random.choice(2, n)).astype(np.uint8)  # First row of the normal Toeplitz matrix

    if mode == 0:  # Toeplitz to circulant matrix with fast Fourier transforms (complexity O(nlogn))
        # The Toeplitz matrix is reformed into a circulant matrix by merging its first row and column together. Since
        # the former has dimensions ̃n×r, the length of the definition of the latter becomes ̃n + r − 1.

        # Delete the first entry from the row and reverse it for the addition into circulant matrix definition
        rrow = np.delete(row, 0)
        rrow = rrow[::-1]

        # Define the circulant matrix definition and ensure the length is n + r - 1
        c_def = np.hstack((col, rrow)).astype(np.uint8)
        assert len(c_def) == n + r - 1

        # The decoded sequence to be compressed is extended, as r−1 zeros are padded to its end
        s_ext = np.hstack((s, np.zeros(r - 1))).astype(np.uint8)

        # To efficiently calculate the key, an optimized multiplication is carried out using the fast Fourier transform.
        # Because of the convolution theorem, the * operator signifies the Hadamard product and therefore element-wise
        # multiplication can be performed.
        fft = np.fft.ifft(np.multiply(np.fft.fft(c_def), np.fft.fft(s_ext)))

        # As the key format is required to be in bits, the result of the inverse fast Fourier transform is taken mod 2.
        fft_bits = (np.round(np.real(fft)) % 2).astype(np.uint8)

        # The key is constituted by the first r bits of the resulting bit string of length ̃n + r − 1
        k = fft_bits[:r]

    else:  # Toeplitz Matrix with dot product calculation (complexity O(n^2))
        t = np.array(sp.toeplitz(col, row), dtype=np.uint8)
        k = np.dot(t, s) % 2

    return k


def universal_hashing(x, y, t):
    """
    Over the promoted binary strings, the parties compute hashes of length t bits. Bob discloses his hash to Alice,
    who compares it with hers. If the hashes are identical, the promoted binary strings are appended to the respective
    privacy amplification sequences.
    :param x: Alice's binary string.
    :param y: Bob's binary string.
    :param t: The length of the hash values.
    :return: True if the hash values are identical, False if the hash values are not identical.
    """

    Q = 32  # Bit length of the input integers
    Q_star = Q + t - 1  # Universe within which a, b and x reside
    n_apo = len(x) // Q

    # In case n_apo is not an integer, the strings are padded with zeros so that n_apo becomes an integer
    if len(x) % Q != 0:
        s = Q - (len(x) - n_apo * Q)  # Find the number of the necessary zeros to be padded
        n_apo = int(np.floor(n_apo) + 1)  # Convert n_apo into an integer
        x = np.append(x, np.zeros(shape=s, dtype=np.int8))
        y = np.append(y, np.zeros(shape=s, dtype=np.int8))
        x_Q = np.array_split(x, n_apo)
        y_Q = np.array_split(y, n_apo)
    else:
        n_apo = int(n_apo)
        x_Q = np.array_split(x, n_apo)
        y_Q = np.array_split(y, n_apo)

    # Convert every element of the arrays to string
    for i in range(n_apo):
        x_Q[i] = str(x_Q[i]).replace("[", "").replace("]", "").replace(" ", "")
        y_Q[i] = str(y_Q[i]).replace("[", "").replace("]", "").replace(" ", "")

    # Generate integers a, b, where a is non-zero odd and belongs to [1, Q_star), and b belongs to [0, Q_star)
    # These values represent the chosen universal hash function from the family and are communicated via the channel
    a = []
    ax_x = []
    ax_y = []
    for i in range(n_apo):  # Generate a different a for every d
        a_i = random.getrandbits(Q_star)
        if a_i % 2 == 0:  # a must be odd
            a_i += 1
        a.append(a_i)
    b = random.randint(0, 2 ** Q_star)

    # Perform the integer multiplications ax for every d (a: w_bar, x: w)
    for i in range(n_apo):
        ax_x_i = a[i] * int(x_Q[i], 2)
        ax_y_i = a[i] * int(y_Q[i], 2)
        ax_x.append(ax_x_i)
        ax_y.append(ax_y_i)

    h_x = sum(ax_x) + b  # Get the sum of all multiplications and add integer b afterwards
    h_y = sum(ax_y) + b  # Get the sum of all multiplications and add integer b afterwards
    # Convert to binary and obtain only the last w_bar bits, as the multiplication gives a result larger than w_bar bits
    h_x = str(np.binary_repr(h_x))[-Q_star:]
    h_y = str(np.binary_repr(h_y))[-Q_star:]

    # Modular arithmetic is replaced with bit shift
    # If there are leading zeros in a binary sequence, Python removes them when performing bit shifts
    # Since the zeros need to be kept, in order to have a fixed length output, the format function is implemented
    h_x = format(int(h_x, 2) >> (Q_star - t), '0' + str(t) + 'b')
    h_y = format(int(h_y, 2) >> (Q_star - t), '0' + str(t) + 'b')

    if h_x == h_y:
        return True
    else:
        return False
