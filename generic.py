# generic.py
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

import maths
import hashes
import warnings
import numpy as np


def composable_key_rate(n_bks, N, n, p, q, r_code, r_m_star, h_m, d_ent, p_ec, e_ent, e_sm, e_hash, e_cor, e_pe, h_k):
    """
    Calculates the composable secret key rate with finite-size effects and identifies the length of the hash function
    for the privacy amplification stage, as well as the size of the key after compression.
    :param n_bks: The total number of frames.
    :param N: The length of the entire sequence after the sacrificed signals for parameter estimation.
    :param n: The length of each frame after the sacrificed signals for parameter estimation.
    :param p: The discretization bits.
    :param q: The Galois Field exponent.
    :param r_code: The code rate.
    :param r_m_star: The theoretical worst-case key rate after parameter estimation.
    :param h_m: The worst-case Holevo bound.
    :param d_ent: The penalty of the entropy calculation.
    :param p_ec: The percentage of correctly decoded frames.
    :param e_ent: The error probability of the entropy calculation.
    :param e_sm: The smoothing parameter.
    :param e_hash: The hashing parameter.
    :param e_cor: The correctness error.
    :param e_pe: The parameter estimation error.
    :param h_k: The Shannon entropy of the discretized variable.
    :param prot: The type of the protocol.
    :return: The theoretical and practical composable key rates, the privacy amplification block length, the length of
    the final key and the epsilon security.
    """

    Delta_AEP = 4 * np.log2(2 ** (p / 2) + 2) * np.sqrt(np.log2(18 / ((p_ec ** 2) * (e_sm ** 4))))
    r_m = h_k - d_ent + r_code * q - p - h_m
    n_pa = int(p_ec * n_bks * n * p)  # The bit length of the concatenated decoded sequences
    Theta = np.log2(p_ec * (1 - ((e_sm ** 2) / 3))) + 2 * np.log2(np.sqrt(2) * e_hash)

    # The practical composable secret key rate can be compared with a corresponding theoretical rate
    r_tilde = r_m - (Delta_AEP / np.sqrt(n)) + (Theta / n)  # The composable key rate without finite-size effects
    r_tilde_star = r_m_star - (Delta_AEP / np.sqrt(n)) + (Theta / n)
    r_final = ((n * p_ec) / N) * r_tilde
    r_theo = ((n * p_ec) / N) * r_tilde_star

    r = int(np.ceil(p_ec * n_bks * n * r_tilde))  # The length of the final key
    e = e_cor + e_sm + e_hash + p_ec * (2 * e_pe + e_ent)  # The ε-security of the protocol

    # Ensure the composable key rate is positive. There may be a case of the practical key rate being positive, while
    # the theoretical key rate is negative. This is communicated to the user for future reference.
    if r_final <= 0:
        warnings.warn("The composable key rate is negative. Privacy amplification cannot be performed.")
    elif r_theo <= 0:
        warnings.warn("The theoretical composable key rate is negative. This means that future simulations under the"
                      "current parameters will highly likely return a negative composable key rate.")

    return r_final, r_theo, n_pa, r, e


def code_rate_calculation(k, n_bks, n, b, p, q, a, s, e):
    """
    Calculates the code rate using the entropy of the discretized data.
    :param k: The discretized variable.
    :param n_bks: The number of blocks.
    :param n: The block length.
    :param b: The reconciliation efficiency.
    :param p: The discretization bits.
    :param q: The Galois Field exponent.
    :param a: The cut-off parameter.
    :param s: The estimated signal-to-noise ratio.
    :param e: The entropy calculation error probability.
    :return: The theoretical and practical code rates, along with the Shannon entropy of the discretized variable and
    the penalty of its calculation.
    """

    # Calculate the Shannon entropy of the discretized variable
    h = maths.shannon_entropy(k, n_bks, n)
    d = np.log2(n * n_bks) * np.sqrt((2 * np.log(2 / e)) / (n * n_bks))
    r = ((b / 2) * np.log2(1 + s) + p - h + d) / q  # Practical code rate
    r_th = np.log2(d + a * np.sqrt((2 * (1 + s) ** b) / (np.pi * np.e))) / q  # Theoretical code rate
    assert r < 1

    return r, r_th, h, d


def frame_error_rate_calculation(n_suc, n_tot):
    """
    Calculates the frame error rate given a success rate from the decoding stage. If no frame was successfully decoded,
    the protocol is aborted.
    :param n_suc: The number of frames that was successfully decoded.
    :param n_tot: The total number of frames.
    :return The success and frame error rates.
    """

    p_EC = n_suc / n_tot
    FER = 1 - p_EC
    if FER == 1.0:
        raise RuntimeWarning("There was no frame that was successfully decoded. A secret key cannot be formed.")
    return p_EC, FER


def verification(mA, mB, b, fnd):
    """
    Verifies that the hash values of the codewords are equal, if the syndromes of the messages are equal.
    :param mA: Alice's codeword.
    :param mB: Bob's codeword.
    :param b: The bit length of the hash values.
    :param fnd: Determines whether the syndrome was matched during error correction.
    :return: True if the hash values match, False if the hash values do not match or the syndromes are not equal after
    error correction.
    """

    if fnd:  # If the syndromes are equal after error correction
        # If the hash values of the codewords are equal, verification is successful
        if hashes.universal_hashing(mA, mB, b):
            return True
        else:  # If the hash values do not match, verification has failed
            return False
    else:  # If the syndromes are not equal after error correction, the frame is automatically discarded
        return False


def precise_reconciliation_efficiency(r, i, h, q, p, d):
    """
    Identifies the reconciliation efficiency to be used with extremely high precision under a given set of parameters.
    :param r: The code rate.
    :param i: The estimated mutual information.
    :param h: The estimated entropy.
    :param q: The Galois field.
    :param p: The discretization bits.
    :param d: The penalty of the entropy estimation.
    """

    b = (h - d + r * q - p) / i
    print("Under current data, the reconciliation efficiency should ideally be:", b)
    if b >= 1:
        warnings.warn("Ideal beta is larger than 1, which implies given parameters are not correct.")

    return b
