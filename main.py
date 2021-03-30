# main.py
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

import gg02
import ldpc
import coding
import hashes
import datetime
import timeit
import galois
import preprocessing
import utilities
import os
import numpy as np
from numba import config


def validity_checks():
    """
    Ensures the input values chosen are valid and can perform a simulation.
    """

    if L <= 0:
        raise RuntimeWarning("The channel length must be a positive number.")
    if A <= 0:
        raise RuntimeWarning("The attenuation loss must be a positive number.")
    if eta <= 0 or eta > 1:
        raise RuntimeWarning("The setup efficiency must belong in the interval (0, 1].")
    if xi < 0:
        raise RuntimeWarning("The excess noise must be a positive number or zero.")
    if v_el < 0:
        raise RuntimeWarning("The electronic noise must be a positive number or zero.")
    if n_bks <= 0:
        raise RuntimeWarning("The number of blocks must be a positive number.")
    if N <= 0:
        raise RuntimeWarning("The block length must be a positive number.")
    if N % 2 != 0:
        raise RuntimeWarning("Parameter N must be an even number.")
    if M <= 0:
        raise RuntimeWarning("The total number of states for parameter estimation must be a positive number.")
    if M >= N * n_bks:
        raise RuntimeWarning("Parameter M must be smaller than the total number of states of the simulation.")
    if beta < 0 or beta >= 1:
        raise RuntimeWarning("The reconciliation efficiency must belong in the interval [0, 1).")
    if iter_max <= 0:
        raise RuntimeWarning("The maximum number of error-correcting iterations must be a positive number.")
    if p <= 0:
        raise RuntimeWarning("Parameter p must be a positive number.")
    if q <= 0:
        raise RuntimeWarning("Parameter q must be a positive number.")
    if q >= p:
        raise RuntimeWarning("Parameter p must be larger than parameter q.")
    if alpha < 3:
        raise RuntimeWarning("Parameter alpha must be at least 3.")
    if p_EC_tilde < 0 or p_EC_tilde > 1:
        raise RuntimeWarning("The decoding success rate must belong in the interval [0, 1].")


def file_logging():
    """
    Stores every significant code input and output of the simulation into a log file.
    """

    log_file = open("logs/" + start_date.strftime("%d-%b-%Y (%H.%M.%S.%f)") + ".txt", "w")
    print("Starting simulation at:", start_date, file=log_file)
    print("Is mu optimally chosen?", is_mu_optimal, "\nIs error correction performed?", is_error_corrected,
          "\nIs the code loaded from disk?", is_code_loaded, file=log_file)
    print("The input values are: L =", L, "eta =", eta, "v_el =", v_el, "att =", A, "xi =", xi, "n_bks =", n_bks,
          "N =", N, "M =", M, "\nbeta =", beta, "iter_max =", iter_max, "q =", q, "alpha =", alpha, "p =", p, "mu =",
          mu, "\ne_PE =", e_PE, "e_s =", e_s, "e_h =", e_h, "e_cor =", e_cor, file=log_file)
    print("The dependent values are: T =", T, "s_z =", s_z, "Xi =", Xi, "m =", m, "n =", n, "\nt =", t, "GF =", GF,
          "delta =", delta, "d =", d, file=log_file)
    print("The asymptotic key rate is", R_asy, "with mutual information", I_AB, file=log_file)
    print("The transmissivity estimators are: T_hat =", T_hat, "T_m =", T_m, "T_star_m =", T_star_m, file=log_file)
    print("The excess noise estimators are: xi_hat =", xi_hat, "xi_m =", xi_m, "xi_star_m =", xi_star_m, file=log_file)
    print("The modified key rate after PE using the overestimations for the Holevo bound is:", R_M, file=log_file)
    print("Accounting for the number of signals sacrificed, the key rate is:", R_m, file=log_file)
    print("The approximate code rate using only the estimated SNR is:", R_code_approx, file=log_file)
    print("Using the entropy of the data, the resulting code rate is:", R_code, file=log_file)
    print("The estimated SNR by the two parties is:", SNR_hat, "and the actual SNR is:", SNR, file=log_file)
    print("The correlation coefficient is:", rho, "and the one using the estimated SNR is", rho_th, file=log_file)
    print("The success rate of the error correction is:", p_EC, "and the FER is:", FER, file=log_file)
    print("The composable key rate under finite-size effects is:", R, file=log_file)
    print("The theoretical rate is:", R_theo, "and differs from the practical rate by:", R - R_theo, file=log_file)
    print("The length of the concatenated correctly decoded sequence to enter PA is", n_tilde, file=log_file)
    print("To match the final key rate, the bit length of the final key is:", r, file=log_file)
    print("The protocol is valid with an overall security", epsilon, file=log_file)
    if is_error_corrected:
        print("The EC stage utilized an LDPC code of size (", l, "x", n, ") and design rate:", R_des, file=log_file)
        print("Under this design rate, the code row weight is:", wr, "and the column weight is:", wc, file=log_file)
        print("Time needed for the privacy amplification stage:", hash_time, file=log_file)
        print("The average number of rounds needed to perform error-correction is:", iter_avg, file=log_file)
        print("Error-correction on individual frames is reported below:", file=log_file)
        for bk in range(n_bks):
            print("Frame:", bk, "| Found:", found[bk], "| Found round", fnd_rnd[bk], "| Hash verified:",
                  hash_verified[bk], file=log_file)
    print("Simulation finished at:", datetime.datetime.now(), file=log_file)
    log_file.close()


config.THREADING_LAYER = 'forksafe'  # Set the threading layer before any parallel target compilation
start_date = datetime.datetime.now()
# Create necessary folders if they do not exist
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('keys'):
    os.makedirs('keys')
if not os.path.exists('codes'):
    os.makedirs('codes')
if not os.path.exists('data'):
    os.makedirs('data')

is_mu_optimal = False  # Determines if the modulation variance will be optimised automatically or set manually
is_error_corrected = False  # Determines if error correction takes place or only the composable key rate is computed
is_code_loaded = True  # Determines if the parity-check matrix will be loaded from disk

# Input values definition
L = 5  # Channel length (km)
A = 0.2  # Attenuation rate (dB/km)
xi = 0.01  # Excess noise
eta = 0.8  # Detector/Setup efficiency
v_el = 0.1  # Electronic noise
beta = 0.9225  # Reconciliation parameter
n_bks = 100  # Number of blocks
N = 5000  # Block size
M = (n_bks * N) * 0.1  # Number of PE runs
p = 6  # Discretization bits
q = 4  # Most significant (top) bits
alpha = 7  # Phase-space cut-off
iter_max = 40  # Max number of EC iterations
e_PE = 2 ** -32  # Probability that the estimated parameters do not belong in the confidence region
e_s = 2 ** -32  # Smoothing parameter
e_h = 2 ** -32  # Hashing parameter
e_cor = 2 ** -32  # Correctness error (universal hash function collision probability)
p_EC_tilde = 0.99  # Fixed successful decoding rate for the calculation of the theoretical composable key rate

# Dependent values
T = 10 ** (-A * L / 10)  # Channel Losses (dB)
s_z = 1 + v_el + eta * T * xi  # Noise variance
Xi = eta * T * xi  # Excess noise variance
Chi = xi + (1 + v_el) / (T * eta)  # Equivalent noise
m = int(M / n_bks)  # PE instances per block
n = N - m  # Key generation points per block
t = int(np.ceil(-np.log2(e_cor)))  # Verification hash output length
GF = 2 ** q  # Number of the Galois Field elements
delta = alpha / (2 ** (p - 1))  # Lattice step in phase space
d = p - q  # Least significant (bottom) bits
if is_mu_optimal:  # For a fixed reconciliation parameter β, find the optimal modulation variance μ >= 1
    mu = gg02.optimal_modulation_variance(T, eta, xi, v_el, beta)
else:
    mu = 21.89226929460376
SNR = (mu - 1) / Chi  # Signal-to-noise ratio

# Ensure all given inputs are legal
validity_checks()

if __name__ == '__main__':
    # Alice prepares and transmits the coherent states. Bob receives the noisy states and measures them. After measuring,
    # they perform key sifting.
    X = np.empty(shape=[n_bks, N], dtype=np.float64)  # Alice's variable
    Y = np.empty(shape=[n_bks, N], dtype=np.float64)  # Bob's variable

    for blk in range(n_bks):
        Q_X, P_X = gg02.prepare_states(N, mu)
        Q_Y, P_Y = gg02.transmit_states(N, Q_X, P_X, T, eta, s_z)
        qu, Y[blk] = gg02.measure_states(N, Q_Y, P_Y)
        X[blk] = gg02.key_sifting(N, Q_X, P_X, qu)

    # Calculate the asymptotic key rate
    I_AB, x_Ey, R_asy = gg02.key_rate_calculation(mu, T, eta, xi, v_el, beta)

    # Determine the states for key generation and parameter estimation and perform parameter estimation
    X_key, Y_key, X_PE, Y_PE = gg02.sacrificed_states_selection(n_bks, n, m, M, X, Y)
    T_hat, xi_hat, T_m, xi_m, T_star_m, xi_star_m = gg02.parameter_estimation(mu, X_PE.ravel(), Y_PE.ravel(), T, eta, Xi,
                                                                              v_el, M, s_z, e_PE)

    # In the next step, they compute an overestimation of the Holevo bound in terms of T_m and ξ_m, so that they may write
    # the modified rate
    I_AB_hat, _, _ = gg02.key_rate_calculation(mu, T_hat, eta, xi_hat, v_el, beta)
    _, x_M, _ = gg02.key_rate_calculation(mu, T_m, eta, xi_m, v_el, beta)
    R_M = beta * I_AB_hat - x_M

    # The theoretical worst-case Holevo bound is calculated using the theoretical estimators
    _, x_M_star, _ = gg02.key_rate_calculation(mu, T_star_m, eta, xi_star_m, v_el, beta)
    R_M_star = beta * I_AB - x_M_star

    # Bob checks the threshold condition I(x : y|T^,ξ^ > χ(E : y)|TM,ξM. If it is not satisfied, the protocol is aborted.
    if I_AB_hat <= x_M:
        raise RuntimeWarning("Estimated mutual information is smaller than worst-case Holevo bound. Protocol is aborted.")
    # Accounting for the number of signals sacrificed for parameter estimation, the actual rate in terms of bits per
    # channel use is given by the rescaling
    R_m = (n / N) * R_M

    # Perform EC preprocessing, i.e., normalization, discretization and splitting of the key generation sequences
    K = np.empty(shape=[n_bks, n], dtype=np.int16)  # Bob's quantized sequence
    K_bar = np.empty(shape=[n_bks, n], dtype=np.int16)  # Bob's most significant bits to be used in encoding
    K_ubar = np.empty(shape=[n_bks, n], dtype=np.int16)  # Bob's least significant bits to be sent in the clear
    P = np.empty(shape=[n_bks, n, 2 ** q], dtype=np.float64)  # The a-priori probabilities for error correction
    gf_add = galois.precomputed_addition_table(GF)  # Galois Field lookup table for addition
    gf_mul = galois.precomputed_multiplication_table(GF)  # Galois Field lookup table for multiplication
    field_values = np.arange(2 ** q)  # All possible values that belong in the specified Galois field

    X_key, Y_key = preprocessing.normalization(X_key, Y_key)
    SNR_hat, rho, rho_th = gg02.code_estimations(mu, X_key, Y_key, T_hat, eta, v_el, xi_hat)
    for blk in range(n_bks):
        K[blk] = preprocessing.discretization(Y_key[blk], alpha, p, delta)
        K_bar[blk], K_ubar[blk] = preprocessing.splitting(K[blk], d)
        P[blk] = coding.a_priori_probabilities(X_key[blk], K_ubar[blk], field_values, rho, alpha, p, delta, d)

    # Identify the rate of the error-correcting code
    R_code, R_code_approx, H_K = gg02.code_rate_calculation(K, n_bks, n, beta, I_AB_hat, p, q, alpha, SNR_hat)

    # Proceed to error-correction, verification, frame error rate estimation and privacy amplification.
    # If specified, all the above stages are skipped, solely the composable key rate is produced and the simulation ends.
    if is_error_corrected:
        # Generate the parity-check matrix H and the indices of the variable nodes, the check nodes and the partial sums
        k, l, H, H_sparse, H_vals, R_des, wc, wr = ldpc.generate_ldpc_code(n, q, R_code, is_code_loaded)
        CN, VN, VN_exc = ldpc.get_nodes(n, l, H_sparse)
        ps_0, pr_0, ps_i, pr_i = coding.get_partial_sums_indices(l, GF, H, CN, gf_add, gf_mul)

        # Declare the values for information reconciliation, frame rate estimation, confirmation and privacy amplification
        kA_dec = np.empty(shape=[n_bks, n], dtype=np.int16)  # Alice's decoded sequence to be sent for confirmation
        K_hat_bin = np.empty(shape=[n_bks, q * n], dtype=np.int8)  # Alice's binary decoded sequence
        K_bar_bin = np.empty(shape=[n_bks, q * n], dtype=np.int8)  # Bob's binary codeword
        K_ubar_bin = np.empty(shape=[n_bks, d * n], dtype=np.int8)  # The binary weakly-correlated bits
        S = np.empty(shape=0, dtype=np.int8)
        S_hat = np.empty(shape=0, dtype=np.int8)
        found = np.empty(shape=n_bks, dtype=np.int8)  # Registers whether the frame was correctly decoded or not
        fnd_rnd = np.empty(shape=n_bks, dtype=np.int16)  # The round where every frame was correctly decoded
        hash_verified = np.zeros(shape=n_bks, dtype=np.int8)  # Registers whether the hash outputs of the keywords match
        iter_avg = 0  # Counter for the average number of error-correcting iterations needed for all decoded blocks
        hash_time = 0  # Time needed for privacy amplification

        for blk in range(0, n_bks):
            K_sd = coding.q_ary_syndrome_calculation(K_bar[blk], l, gf_add, gf_mul, H_vals, CN)
            r_mn_i1, r_mn_i2, r_mn_i3 = coding.get_rmn_indices(l, GF, H, K_sd, CN, gf_add, gf_mul)
            kA_dec[blk], found[blk], fnd_rnd[blk] = coding.q_ary_decode(n, l, K_sd, iter_max, GF, P[blk], CN, VN, VN_exc,
                                                                        gf_add, gf_mul, ps_0, pr_0, ps_i, pr_i, H_vals,
                                                                        r_mn_i1, r_mn_i2, r_mn_i3)

            # Convert q-ary and d-ary sequences from their respective field to binary to be fit for the verification stage
            K_hat_bin[blk] = utilities.q_ary_to_binary(kA_dec[blk], q)
            K_bar_bin[blk] = utilities.q_ary_to_binary(K_bar[blk], q)
            K_ubar_bin[blk] = utilities.q_ary_to_binary(K_ubar[blk], d)
            np.testing.assert_equal(len(K_hat_bin[blk]), len(K_bar_bin[blk]))

            hash_verified[blk] = gg02.verification(K_hat_bin[blk], K_bar_bin[blk], t, found[blk])
            if hash_verified[blk] == 1:
                iter_avg = iter_avg + fnd_rnd[blk]  # The found round is included in the average round calculation
                S_hat = np.append(S_hat, np.hstack((K_hat_bin[blk], K_ubar_bin[blk])))
                S = np.append(S, np.hstack((K_bar_bin[blk], K_ubar_bin[blk])))
                print("Block:", blk, "| Found successfully at round:", fnd_rnd[blk], "| Verification: Success")
            elif hash_verified[blk] == 0 and found[blk] == 1:
                print("Block:", blk, "| Found Successfully at round:", fnd_rnd[blk], "| Verification: Failure")
            else:
                print("Block:", blk, "| Decoding Failure")

        # Compute the average number of rounds needed to decode all blocks using only the successfully verified blocks
        if np.count_nonzero(hash_verified) != 0:
            iter_avg = iter_avg / np.count_nonzero(hash_verified)

        # Calculate the FER and the composable key rate
        p_EC, FER = gg02.frame_error_rate_calculation(np.count_nonzero(hash_verified), n_bks)
        R, R_theo, n_tilde, r, epsilon = gg02.composable_key_rate(n_bks, N, n, p, q, R_code, R_M_star, x_M, p_EC, e_s, e_h,
                                                                  e_cor, e_PE, H_K)

        # If the composable key rate positive, proceed to privacy amplification stage. Otherwise, the protocol is aborted.
        if R > 0:
            S_hat_bold = np.ravel(S_hat)
            S_bold = np.ravel(S)
            np.testing.assert_equal(n_tilde, len(S_hat_bold))  # Ensure the sequence to enter PA has the correct bit length
            np.testing.assert_equal(n_tilde, len(S_bold))

            # Privacy amplification is performed. The process is timed.
            start = timeit.default_timer()
            K_bold = hashes.privacy_amplification(S_hat_bold, len(S_hat_bold), r, 0)
            stop = timeit.default_timer()
            hash_time = stop - start

            # Output the final key to a file
            final_key_file = open("keys/" + datetime.datetime.now().strftime("%d-%b-%Y (%H.%M.%S.%f)") + ".txt", "w")
            print(K_bold, file=final_key_file)
            final_key_file.close()

    else:
        p_EC, FER = gg02.frame_error_rate_calculation(n_bks * p_EC_tilde, n_bks)
        R, R_theo, n_tilde, r, epsilon = gg02.composable_key_rate(n_bks, N, n, p, q, R_code, R_M_star, x_M, p_EC, e_s, e_h,
                                                                  e_cor, e_PE, H_K)

    # Before runtime is over, store the input and output values in a log file
    file_logging()
