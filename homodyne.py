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
import generic
import datetime
import timeit
import galois
import preprocessing
import utilities
import warnings
import psutil
import numpy as np
import numba as nb
from numba import config
from numba.typed import Dict
from numba import types
from dataclasses import dataclass


@dataclass
class Results:
    # Dependent values
    s_z: float  # Noise variance
    Chi: float  # Equivalent noise
    Xi: float  # Excess noise variance
    d: int  # Least significant (bottom) bits
    m: int  # Parameter estimation instances per block
    t: int  # Verification hash output length
    n: int  # Key generation points per block
    GF: int  # Number of the Galois Field elements
    delta: int  # Lattice step in phase space

    # Asymptotic limit values
    I_XY: float  # Mutual information
    SNR: float  # Signal-to-noise ratio
    R_asy: float  # Asymptotic key rate

    # Parameter estimation results
    T_hat: float  # Estimated transmissivity
    xi_hat: float  # Estimated excess noise
    T_m: float  # Worst-case scenario for the estimated transmissivity
    xi_m: float  # Worst-case scenario for the estimated excess noise
    T_star_m: float  # Worst-case scenario for the transmissivity using
    xi_star_m: float  # Worst-case scenario for the transmissivity using
    x_M: float  # Worst-case scenario for the Holevo bound
    R_M: float  # Key rate after parameter estimation
    R_m: float  # Key rate after parameter estimation accounting for the sacrificed number of states
    R_M_star: float  # Theoretical key rate after parameter estimation

    # Signal-to-noise ratio and code rate values
    I_XY_hat: float  # Estimated mutual information
    SNR_hat: float  # Estimated signal-to-noise ratio
    rho: float  # Correlation coefficient
    rho_th: float  # Correlation coefficient using the estimated signal-to-noise ratio
    H_K: float  # Entropy of the discretized
    d_ent: float  # Penalty of the entropy calculation
    R_code: float  # Code rate
    R_code_approx: float  # Code rate theoretical approximation

    # LDPC design parameters
    l: int  # Number of rows of the LDPC matrix
    k: int  # Number of information bits of the LDPC matrix
    wc: int  # Number of non-zero entries per column
    wr: int  # Number of non-zero entries per row
    R_des: float  # The design rate of the LDPC code
    beta_true: float  # The reconciliation efficiency that matches the design rate with extreme precision

    # Performance and memory values
    ec_time: float  # Duration of the error-correction stage
    pa_time: float  # Duration of the privacy amplification stage
    sim_time: float  # Duration of the entire simulation
    time_avg: float  # Average duration needed to decode a block (in seconds)
    iter_avg: float  # Average number of rounds needed to perform error correction
    peak_mem_ec: float  # Peak memory consumption at the end of the error-correction stage
    peak_mem_end: float  # Peak memory consumption at the end of the simulation

    # Block decoding analytics
    found: np.ndarray  # Boolean array of whether the frame was successfully decoded or not
    fnd_rnd: np.ndarray  # Error-correction syndrome matching rounds
    hash_verified: np.ndarray  # Boolean array of whether the keyword hash outputs match or not
    dec_time: np.ndarray  # Duration of decoding for each block

    # Composable key rate values
    r: int  # Final key length
    R: float  # Composable key rate
    R_theo: float  # Composable key rate from theoretical computations
    p_EC: float  # Error-correction success probability
    FER: float  # Frame error rate
    n_tilde: int  # Total bit string length after error correction
    epsilon: float  # Epsilon security parameter

    def __init__(self, s_z, Chi, Xi, d, m, n, t, delta, GF, SNR, I_XY, R_asy, T_hat, xi_hat, T_m, xi_m, T_star_m, xi_star_m,
                 x_M, R_m, R_M, R_M_star, I_XY_hat, SNR_hat, rho, rho_th, H_K, d_ent, R_code, R_code_approx, l, k, wr, wc, R_des, beta_true, peak_mem_ec,
                 peak_mem_end, ec_time, pa_time, sim_time, time_avg, iter_avg, found, fnd_rnd, hash_verified, dec_time, p_EC, FER, r, R, R_theo,
                 n_tilde, epsilon):

        self.s_z = s_z
        self.Chi = Chi
        self.Xi = Xi
        self.d = d
        self.m = m
        self.n = n
        self.t = t
        self.GF = GF
        self.delta = delta

        self.SNR = SNR
        self.I_XY = I_XY
        self.R_asy = R_asy
        self.T_hat = T_hat
        self.xi_hat = xi_hat
        self.T_m = T_m
        self.xi_m = xi_m
        self.T_star_m = T_star_m
        self.xi_star_m = xi_star_m
        self.x_M = x_M
        self.R_M = R_M
        self.R_m = R_m
        self.R_M_star = R_M_star

        self.I_XY_hat = I_XY_hat
        self.SNR_hat = SNR_hat
        self.rho = rho
        self.rho_th = rho_th
        self.H_K = H_K
        self.d_ent = d_ent
        self.R_code = R_code
        self.R_code_approx = R_code_approx

        self.l = l
        self.k = k
        self.wr = wr
        self.wc = wc
        self.R_des = R_des
        self.beta_true = beta_true

        self.ec_time = ec_time
        self.pa_time = pa_time
        self.sim_time = sim_time
        self.time_avg = time_avg
        self.iter_avg = iter_avg
        self.peak_mem_ec = peak_mem_ec
        self.peak_mem_end = peak_mem_end

        self.found = found
        self.fnd_rnd = fnd_rnd
        self.hash_verified = hash_verified
        self.dec_time = dec_time

        self.r = r
        self.R = R
        self.R_theo = R_theo
        self.n_tilde = n_tilde
        self.p_EC = p_EC
        self.FER = FER
        self.epsilon = epsilon


config.THREADING_LAYER = 'forksafe'  # Set the threading layer before any parallel target compilation
start_date = datetime.datetime.now()
start_timer = timeit.default_timer()
proc = psutil.Process()
sim_id = np.random.randint(10 ** 6, size=1, dtype=np.int32)  # Session identifier for saving files onto the disk

e_PE = 2 ** -32  # Probability that the estimated parameters do not belong in the confidence region
e_ent = 2 ** -32  # Error probability of entropy calculation
e_s = 2 ** -32  # Smoothing parameter
e_h = 2 ** -32  # Hashing parameter
e_cor = 2 ** -32  # Correctness error (universal hash function collision probability)


class Protocol:

    def __init__(self, **kwargs):
        protocol_arg = kwargs.get('protocol', None)
        is_error_corrected_arg = kwargs.get('ec', None)
        is_pe_data_saved_arg = kwargs.get('save_pe', None)
        is_pe_data_loaded_arg = kwargs.get('load_pe', None)
        is_pe_alternative_arg = kwargs.get('pe_alt', None)
        is_code_loaded_arg = kwargs.get('load_code', None)
        is_mu_optimal_arg = kwargs.get('optimal_mu', None)
        is_data_saved_arg = kwargs.get('save_data', None)
        is_data_loaded_arg = kwargs.get('load_data', None)

        x_data_input = kwargs.get('X_data', None)
        y_data_input = kwargs.get('Y_data', None)
        A_input = kwargs.get('A', None)
        L_input = kwargs.get('L', None)
        T_input = kwargs.get('T', None)
        eta_input = kwargs.get('eta', None)
        xi_input = kwargs.get('xi', None)
        v_el_input = kwargs.get('v_el', None)
        N_input = kwargs.get('N', None)
        n_bks_input = kwargs.get('n_bks', None)
        mu_input = kwargs.get('mu', None)
        M_input = kwargs.get('M', None)
        beta_input = kwargs.get('beta', None)
        iter_input = kwargs.get('iter_max', None)
        self.b_id = kwargs.get('b_id', None)
        p_input = kwargs.get('p', None)
        q_input = kwargs.get('q', None)
        alpha_input = kwargs.get('alpha', None)
        p_EC_tilde_input = kwargs.get('p_EC_tilde', None)

        if x_data_input is not None:
            self.X_data = x_data_input
        if y_data_input is not None:
            self.Y_data = y_data_input
        if A_input is not None:
            self.A = A_input
        if L_input is not None:
            self.L = L_input
        if eta_input is not None:
            self.eta = eta_input
        if xi_input is not None:
            self.xi = xi_input
        if v_el_input is not None:
            self.v_el = v_el_input
        if N_input is not None:
            self.N = N_input
        if n_bks_input is not None:
            self.n_bks = n_bks_input
        if beta_input is not None:
            self.beta = beta_input
        if iter_input is not None:
            self.iter_max = iter_input
        if p_input is not None:
            self.p = p_input
        if q_input is not None:
            self.q = q_input
        if alpha_input is not None:
            self.alpha = alpha_input
        if p_EC_tilde_input is not None:
            self.p_EC_tilde = p_EC_tilde_input
        else:
            self.p_EC_tilde = 1

        if protocol_arg is not None:
            self.protocol = protocol_arg
        if is_error_corrected_arg is not None:
            self.is_error_corrected = is_error_corrected_arg
        else:
            self.is_error_corrected = False
        if is_pe_data_saved_arg is not None:
            self.is_pe_data_saved = is_pe_data_saved_arg
        else:
            self.is_pe_data_saved = False
        if is_pe_data_loaded_arg is not None:
            self.is_pe_data_loaded = is_pe_data_loaded_arg
        else:
            self.is_pe_data_loaded = False
        if is_pe_alternative_arg is not None:
            self.is_pe_alternative = is_pe_alternative_arg
        else:
            self.is_pe_alternative = True
        if is_code_loaded_arg is not None:
            self.is_code_loaded = is_code_loaded_arg
        else:
            self.is_code_loaded = False
        if is_mu_optimal_arg is not None:
            self.is_mu_optimal = is_mu_optimal_arg
        else:
            self.is_mu_optimal = False
        if is_data_saved_arg is not None:
            self.is_data_saved = is_data_saved_arg
        else:
            self.is_data_saved = False
        if is_data_loaded_arg is not None:
            self.is_data_loaded = is_data_loaded_arg
        else:
            self.is_data_loaded = False

        # Dependent values
        if T_input is not None:
            self.T = T_input
        else:
            self.T = 10 ** (-self.A * self.L / 10)  # Channel Losses (dB)
        if M_input is not None:
            self.M = int(np.round(self.n_bks * self.N * M_input))  # Number of PE runs
        else:
            self.M = int(np.round(self.n_bks * self.N * 0.1))

        if mu_input is not None:
            self.mu = mu_input
        else:
            if self.is_mu_optimal:  # For a fixed reconciliation parameter β, find the optimal modulation variance μ >= 1
                self.mu = gg02.optimal_modulation_variance(self.T, self.eta, self.xi, self.v_el, self.beta)
            else:
                Chi = self.xi + (1 + self.v_el) / (self.T * self.eta)
                self.mu = 6 * Chi + 1

    def validity_checks(self):
        """
        Ensures the input values chosen are valid and can perform a simulation.
        """

        if self.L <= 0:
            raise ValueError("The channel length must be a positive number.")
        if self.A <= 0:
            raise ValueError("The attenuation loss must be a positive number.")
        if self.mu < 1:
            raise ValueError("The modulation variance must be at least 1.")
        if self.eta <= 0 or self.eta > 1:
            raise ValueError("The setup efficiency must belong in the interval (0, 1].")
        if self.xi < 0:
            raise ValueError("The excess noise must be a positive number or zero.")
        if self.v_el < 0:
            raise ValueError("The electronic noise must be a positive number or zero.")
        if self.n_bks <= 0:
            raise ValueError("The number of blocks must be a positive number.")
        if self.N <= 0:
            raise ValueError("The block length must be a positive number.")
        if self.N % 2 != 0:
            raise ValueError("Parameter N must be an even number.")
        if self.M <= 0:
            raise ValueError("The total number of states for parameter estimation must be a positive number.")
        if self.M >= self.N * self.n_bks:
            raise ValueError("Parameter M must be smaller than the total number of states of the simulation.")
        if self.beta < 0 or self.beta >= 1:
            raise ValueError("The reconciliation efficiency must belong in the interval [0, 1).")
        if self.iter_max <= 0:
            raise ValueError("The maximum number of error-correcting iterations must be a positive number.")
        if self.p <= 0:
            raise ValueError("Parameter p must be a positive number.")
        if self.q <= 0:
            raise ValueError("Parameter q must be a positive number.")
        if self.q >= self.p:
            raise ValueError("Parameter p must be larger than parameter q.")
        if self.q > 8:
            raise ValueError("The maximum value currently supported for q is 8. Higher values may be supported soon.")
        if self.alpha < 3:
            raise ValueError("Parameter alpha must be at least 3.")
        if self.p_EC_tilde < 0 or self.p_EC_tilde > 1:
            raise ValueError("The decoding success rate must belong in the interval [0, 1].")

    def file_logging(self, res):
        """
        Stores every significant code input and output of the simulation into a log file.
        """

        log_file = open("logs/" + start_date.strftime("%d-%b-%Y (%H.%M.%S.%f)") + ".txt", "w")
        print("Logs for simulation with sim id:", sim_id, file=log_file)
        print("Started simulation at:", start_date, file=log_file)
        print("The chosen protocol utilises homodyne detection.", file=log_file)
        print("Is mu optimally chosen?", self.is_mu_optimal,
              "\nIs error correction performed?", self.is_error_corrected, "\nIs the code loaded from disk?", self.is_code_loaded,
              "\nParameter Estimation method (False for Leverrier, True for Usenko):", self.is_pe_alternative, file=log_file)
        print("The input values are: L =", self.L, "eta =", self.eta, "v_el =", self.v_el, "att =", self.A, "xi =", self.xi, "n_bks =", self.n_bks,
              "N =", self.N, "M =", self.M, "\nbeta =", self.beta, "iter_max =", self.iter_max, "q =", self.q, "alpha =", self.alpha, "p =", self.p, "mu =",
              self.mu, "\ne_PE =", e_PE, "e_s =", e_s, "e_h =", e_h, "e_cor =", e_cor, file=log_file)
        print("The dependent values are: T =", self.T, "s_z =", res.s_z, "Xi =", res.Xi, "m =", res.m, "n =", res.n, "\nt =", res.t, "GF =", res.GF,
              "delta =", res.delta, "d =", res.d, file=log_file)
        print("The asymptotic key rate is", res.R_asy, file=log_file)
        print("The transmissivity estimators are: T_hat =", res.T_hat, "T_m =", res.T_m, "T_star_m =", res.T_star_m, file=log_file)
        print("The excess noise estimators are: xi_hat =", res.xi_hat, "xi_m =", res.xi_m, "xi_star_m =", res.xi_star_m, file=log_file)
        print("The overestimation of the Holevo bound is:", res.x_M, file=log_file)
        print("The modified key rate after PE using the overestimations for the Holevo bound is:", res.R_M, file=log_file)
        print("The theoretical modified key rate is:", res.R_M_star, file=log_file)
        print("Accounting for the number of signals sacrificed, the key rate is:", res.R_m, file=log_file)
        print("The approximate code rate using only the estimated SNR is:", res.R_code_approx, file=log_file)
        print("The entropy estimator of the discretized variable is:", res.H_K, file=log_file)
        print("The penalty of the entropy calculation is:", res.d_ent, file=log_file)
        print("Using the entropy of the data, the resulting code rate is:", res.R_code, file=log_file)
        print("The estimated mutual information between the two parties is:", res.I_XY_hat, "and the actual one is", res.I_XY, file=log_file)
        print("The estimated SNR by the two parties is:", res.SNR_hat, "and the actual SNR is:", res.SNR, file=log_file)
        print("The correlation coefficient is:", res.rho, "and the one using the estimated SNR is", res.rho_th, file=log_file)
        print("The success rate of the error correction is:", res.p_EC, file=log_file)
        print("The frame error rate of the error correction is:", res.FER, file=log_file)
        print("The composable key rate under finite-size effects is:", res.R, file=log_file)
        print("The theoretical rate is:", res.R_theo, "and differs from the practical rate by:", res.R - res.R_theo, file=log_file)
        print("The length of the concatenated correctly decoded sequence to enter PA is:", res.n_tilde, file=log_file)
        print("To match the final key rate, the bit length of the final key is:", res.r, file=log_file)
        print("The protocol is valid with an overall security", res.epsilon, file=log_file)
        if self.is_error_corrected:
            print("The EC stage utilized an LDPC code of size (", res.l, "x", res.n, ") and design rate:", res.R_des, file=log_file)
            print("The number of information bits of the LDPC code is:", res.k, file=log_file)
            print("Under this design rate, the code row weight is:", res.wr, "and the column weight is:", res.wc, file=log_file)
            print("For this set of data, the ideal the reconciliation efficiency is:", res.beta_true, file=log_file)
            print("Duration of the error correction stage:", str(datetime.timedelta(seconds=res.ec_time)), file=log_file)
            print("Duration of the privacy amplification stage:", str(datetime.timedelta(seconds=res.pa_time)), file=log_file)
            print("Ratio of the duration of EC to the entire simulation:", res.ec_time / res.sim_time, file=log_file)
            print("Peak memory consumption at the end of error correction:", res.peak_mem_ec, file=log_file)
            print("The average number of rounds needed to perform error correction is:", res.iter_avg, file=log_file)
            print("The average duration needed to decode a block (in seconds) is:", res.time_avg, file=log_file)
            print("Error-correction analysis on individual frames is reported below:", file=log_file)
            for bk in range(self.n_bks):
                print("Frame:", bk + 1, "| Found:", res.found[bk], "| Found round", res.fnd_rnd[bk], "| Hash verified:",
                      res.hash_verified[bk], "| Total Processing time:", res.dec_time[bk], "| Average Processing Time per Iteration:",
                      res.dec_time[bk] / res.fnd_rnd[bk], file=log_file)
        print("The peak memory usage of the simulation in MB was:", res.peak_mem_end, file=log_file)
        print("Duration of the entire simulation:", str(datetime.timedelta(seconds=res.sim_time)), file=log_file)
        print("Simulation finished at:", datetime.datetime.now(), file=log_file)
        log_file.close()

    def processing(self):

        # Calculate the dependent values
        Xi = self.eta * self.T * self.xi  # Excess noise variance
        s_z = Xi + self.v_el + 1
        Chi = self.xi + (1 + self.v_el) / (self.T * self.eta)
        d = self.p - self.q
        m = int(self.M / self.n_bks)  # PE instances per block
        n = self.N - m  # Key generation points per block
        t = int(np.ceil(-np.log2(e_cor)))  # Verification hash output length
        GF = 2 ** self.q  # Number of the Galois Field elements
        delta = self.alpha / (2 ** (self.p - 1))  # Lattice step in phase space
        SNR = (self.mu - 1) / Chi  # Signal-to-noise ratio

        # Specify the file names for the data
        data_filename = "data/N_" + str(self.N) + "_n_bks_" + str(self.n_bks) + "_SNR_" + str(SNR) + "_p_" + str(self.p)  + ".npz"
        pe_data_filename = "data/PE_N_" + str(self.N) + "_n_bks_" + str(self.n_bks) + "_SNR_" + str(SNR) + "_p_" + str(self.p) + ".npz"

        # Alice prepares and transmits the coherent states. Bob receives the noisy states and measures them. After measuring,
        # they perform key sifting.
        if self.is_data_loaded:
            data = np.load(data_filename, allow_pickle=True)
            X = data["x"]
            Y = data["y"]
            # Ensure the length of the loaded arrays matches the chosen block length
            np.testing.assert_equal(len(X[0]), self.N)
            np.testing.assert_equal(len(Y[0]), self.N)
            np.testing.assert_equal(len(X), self.n_bks)
            np.testing.assert_equal(len(Y), self.n_bks)
        else:
            X = np.empty(shape=[self.n_bks, self.N], dtype=np.float64)  # Alice's variable
            Y = np.empty(shape=[self.n_bks, self.N], dtype=np.float64)  # Bob's variable
            for blk in range(self.n_bks):
                Q_X, P_X = gg02.prepare_states(self.N, self.mu)
                Q_Y, P_Y = gg02.transmit_states(self.N, Q_X, P_X, self.T, self.eta, s_z)
                qu, Y[blk] = gg02.measure_states(self.N, Q_Y, P_Y)
                X[blk] = gg02.key_sifting(self.N, Q_X, P_X, qu)
            if self.is_data_saved:
                np.savez_compressed(data_filename, x=X, y=Y)

        # Calculate the asymptotic key rate
        I_XY, x_Ey, R_asy = gg02.key_rate_calculation(self.mu, self.T, self.eta, self.xi, self.v_el, self.beta)

        # Determine the states for key generation and parameter estimation and perform parameter estimation
        if self.is_pe_data_loaded:
            data = np.load(pe_data_filename, allow_pickle=True)
            X_key = data["xkey"]
            Y_key = data["ykey"]
            X_PE = data["xpe"]
            Y_PE = data["ype"]
            # Ensure the length of the loaded arrays matches the parameters set
            np.testing.assert_equal(len(X_key), self.n_bks)
            np.testing.assert_equal(len(Y_key), self.n_bks)
            np.testing.assert_equal(len(X_PE), self.n_bks)
            np.testing.assert_equal(len(Y_PE), self.n_bks)
            np.testing.assert_equal(len(X_key[0]), n)
            np.testing.assert_equal(len(Y_key[0]), n)
            np.testing.assert_equal(len(X_PE[0]), m)
            np.testing.assert_equal(len(Y_PE[0]), m)
        else:
            X_key, Y_key, X_PE, Y_PE = gg02.sacrificed_states_selection(self.n_bks, n, m, self.M, X, Y)
            if self.is_pe_data_saved:
                np.savez_compressed(pe_data_filename, xkey=X_key, ykey=Y_key, xpe=X_PE, ype=Y_PE)
        T_hat, xi_hat, T_m, xi_m, T_star_m, xi_star_m = gg02.parameter_estimation(self.mu, X_PE.ravel(), Y_PE.ravel(), self.T,
                                                                                  self.eta, Xi, self.v_el, self.M, s_z, e_PE,
                                                                                  self.is_pe_alternative)

        # In the next step, they compute an overestimation of the Holevo bound in terms of T_m and ξ_m, so that they may write
        # the modified rate
        I_XY_hat, _, _ = gg02.key_rate_calculation(self.mu, T_hat, self.eta, xi_hat, self.v_el, self.beta)
        _, x_M, _ = gg02.key_rate_calculation(self.mu, T_m, self.eta, xi_m, self.v_el, self.beta)
        R_M = self.beta * I_XY_hat - x_M
        # The theoretical worst-case Holevo bound is calculated using the theoretical estimators
        _, x_M_star, _ = gg02.key_rate_calculation(self.mu, T_star_m, self.eta, xi_star_m, self.v_el, self.beta)
        R_M_star = self.beta * I_XY - x_M_star

        # Bob checks the threshold condition I(x : y|T^,ξ^ > χ(E : y)|TM,ξM. If it is not satisfied, the protocol is aborted.
        if I_XY_hat <= x_M:
            print("Estimated mutual information:", I_XY_hat, "Worst-case Holevo bound:", x_M)
            raise RuntimeWarning("Estimated mutual information is smaller than worst-case Holevo bound. Protocol is aborted.")
        # Accounting for the number of signals sacrificed for parameter estimation, the actual rate in terms of bits per
        # channel use is given by the rescaling
        R_m = (n / self.N) * R_M

        # Perform EC preprocessing, i.e., normalization, discretization and splitting of the key generation sequences
        K = np.empty(shape=[self.n_bks, n], dtype=np.uint16)  # Bob's quantized sequence
        K_bar = np.empty(shape=[self.n_bks, n], dtype=np.uint16)  # Bob's most significant bits to be used in encoding
        K_ubar = np.empty(shape=[self.n_bks, n], dtype=np.uint16)  # Bob's least significant bits to be sent in the clear
        P = np.empty(shape=[self.n_bks, n, 2 ** self.q], dtype=np.float32)  # The a-priori probabilities for error correction
        field_values = np.arange(2 ** self.q)  # All possible values that belong in the specified Galois field

        X_key, Y_key = preprocessing.normalization(X_key, Y_key, True)
        SNR_hat, rho, rho_th = gg02.code_estimations(self.mu, X_key, Y_key, T_hat, self.eta, self.v_el, xi_hat)
        for blk in range(self.n_bks):
            K[blk] = preprocessing.discretization(Y_key[blk], self.alpha, self.p, delta)
            K_bar[blk], K_ubar[blk] = preprocessing.splitting(K[blk], d)
            P[blk] = coding.a_priori_probabilities(X_key[blk], K_ubar[blk], field_values, rho, self.alpha, self.p, delta, d)

        # Identify the rate of the error-correcting code
        R_code, R_code_approx, H_K, d_ent = generic.code_rate_calculation(K, self.n_bks, n, self.beta, self.p,
                                                                          self.q, self.alpha, SNR_hat, e_ent)

        if self.b_id is not None:
            return R_code, I_XY_hat, H_K, self.q, self.p, d_ent

        del X, Y, X_key, Y_key, X_PE, Y_PE, K  # Free memory by deleting unneeded variables

        # Proceed to error-correction, verification, frame error rate estimation and privacy amplification.
        # If specified, all the above stages are skipped, solely the composable key rate is produced and the simulation ends.
        if self.is_error_corrected:
            # Generate the parity-check matrix H, as well as the indices of the variable nodes and the check nodes
            k, l, H, H_sparse, H_vals, R_des, wc, wr = ldpc.generate_code(n, self.q, R_code, self.is_code_loaded)
            beta_true = generic.precise_reconciliation_efficiency(R_des, I_XY_hat, H_K, self.q, self.p, d_ent)
            start = timeit.default_timer()
            CN, VN, VN_exc = ldpc.get_nodes(n, l, H_sparse, False)
            stop = timeit.default_timer()

            # Conserve memory by using a smaller integer capacity if the number of rows of the parity check matrix allows it
            if l < 2 ** 16:
                key_type = nb.uint16
            else:
                print("Because the row number of the LDPC code is very large, extra memory will be required.")
                key_type = nb.uint32
            ps_first = Dict.empty(key_type=nb.types.Tuple((key_type, nb.uint8, nb.uint8)), value_type=types.uint8)
            pr_first = Dict.empty(key_type=nb.types.Tuple((key_type, nb.uint8, nb.uint8)), value_type=types.uint8)
            ps_rest = Dict.empty(key_type=nb.types.Tuple((key_type, nb.uint8, nb.uint8, nb.uint8)), value_type=types.uint8)
            pr_rest = Dict.empty(key_type=nb.types.Tuple((key_type, nb.uint8, nb.uint8, nb.uint8)), value_type=types.uint8)
            r_ind_1 = Dict.empty(key_type=nb.types.Tuple((key_type, nb.uint8, nb.uint8)), value_type=types.uint8)
            r_ind_2 = Dict.empty(key_type=nb.types.Tuple((key_type, nb.uint8, nb.uint8)), value_type=types.uint8)
            r_ind_3 = Dict.empty(key_type=nb.types.Tuple((key_type, nb.uint8, nb.uint8, nb.uint8)), value_type=types.uint8)
            gf_add = galois.precomputed_addition_table(GF)  # Galois Field lookup table for addition
            gf_mul = galois.precomputed_multiplication_table(GF)  # Galois Field lookup table for multiplication
            ps_0, pr_0, ps_i, pr_i = coding.get_partial_sums_indices(l, GF, H, CN, gf_add, gf_mul, ps_first, pr_first,
                                                                     ps_rest, pr_rest)

            # Declare the values for information reconciliation, frame rate estimation, confirmation and privacy amplification
            kA_dec = np.empty(shape=[self.n_bks, n], dtype=np.uint16)  # Alice's decoded sequence to be sent for confirmation
            K_hat_bin = np.empty(shape=[self.n_bks, self.q * n], dtype=np.uint8)  # Alice's binary decoded sequence
            K_bar_bin = np.empty(shape=[self.n_bks, self.q * n], dtype=np.uint8)  # Bob's binary codeword
            K_ubar_bin = np.empty(shape=[self.n_bks, d * n], dtype=np.uint8)  # The binary weakly-correlated bits
            S = np.empty(shape=0, dtype=np.uint8)
            S_hat = np.empty(shape=0, dtype=np.uint8)
            found = np.empty(shape=self.n_bks, dtype=np.uint8)  # Registers whether the frame was correctly decoded or not
            fnd_rnd = np.empty(shape=self.n_bks, dtype=np.uint16)  # The round where every frame was correctly decoded
            hash_verified = np.zeros(shape=self.n_bks, dtype=np.int8)  # Registers whether the hash outputs of the keywords match
            dec_time = np.empty(shape=self.n_bks, dtype=np.float32)  # The duration of postprocessing on every frame
            iter_avg = 0  # Counter for the average number of error-correcting iterations needed for all decoded blocks
            time_avg = 0  # Counter for the average time needed to successfully decode and verify a block
            ec_time = 0  # Duration of error correction
            pa_time = 0  # Duration of privacy amplification

            ec_start = timeit.default_timer()  # Timer marking the start of the error-correction stage
            for blk in range(0, self.n_bks):
                iter_start = timeit.default_timer()
                K_sd = coding.q_ary_syndrome_calculation(K_bar[blk], l, gf_add, gf_mul, H_vals, CN)
                r_mn_i1, r_mn_i2, r_mn_i3 = coding.get_rmn_indices(l, GF, H, K_sd, CN, gf_add, gf_mul, r_ind_1, r_ind_2, r_ind_3)
                kA_dec[blk], found[blk], fnd_rnd[blk] = coding.q_ary_decode(n, l, K_sd, self.iter_max, GF, P[blk], CN, VN, VN_exc,
                                                                            gf_add, gf_mul, ps_0, pr_0, ps_i, pr_i, H_vals,
                                                                            r_mn_i1, r_mn_i2, r_mn_i3)

                # Convert q-ary and d-ary sequences from their respective field to binary to be fit for the verification stage
                K_hat_bin[blk] = utilities.q_ary_to_binary(kA_dec[blk], self.q)
                K_bar_bin[blk] = utilities.q_ary_to_binary(K_bar[blk], self.q)
                K_ubar_bin[blk] = utilities.q_ary_to_binary(K_ubar[blk], d)
                np.testing.assert_equal(len(K_hat_bin[blk]), len(K_bar_bin[blk]))

                hash_verified[blk] = generic.verification(K_hat_bin[blk], K_bar_bin[blk], t, found[blk])
                iter_stop = timeit.default_timer()
                dec_time[blk] = iter_stop - iter_start  # Time spent on decoding the block and verifying

                if hash_verified[blk] == 1:
                    iter_avg = iter_avg + fnd_rnd[blk]
                    time_avg = time_avg + dec_time[blk]
                    S_hat = np.append(S_hat, np.hstack((K_hat_bin[blk], K_ubar_bin[blk])))
                    S = np.append(S, np.hstack((K_bar_bin[blk], K_ubar_bin[blk])))
                    print("Block:", blk + 1, "| Found successfully at round:", fnd_rnd[blk], "| Verification: Success", "| Processing Time:", dec_time[blk],
                          "Average Processing Time per Iteration:", dec_time[blk] / fnd_rnd[blk])
                elif hash_verified[blk] == 0 and found[blk] == 1:
                    print("Block:", blk + 1, "| Found Successfully at round:", fnd_rnd[blk], "| Verification: Failure", "| Processing Time:", dec_time[blk],
                          "Average Processing Time per Iteration:", dec_time[blk] / fnd_rnd[blk])
                else:
                    print("Block:", blk + 1, "| Decoding Failure", "| Total Processing Time:", dec_time[blk],
                          "Average Processing Time per Iteration:", dec_time[blk] / fnd_rnd[blk])
            ec_stop = timeit.default_timer()
            ec_time = ec_stop - ec_start

            # Compute the average number of rounds needed to decode all blocks using only the successfully verified blocks
            if np.count_nonzero(hash_verified) != 0:
                iter_avg = iter_avg / np.count_nonzero(hash_verified)
                time_avg = time_avg / np.count_nonzero(hash_verified)

            # Calculate the FER and the composable key rate
            p_EC, FER = generic.frame_error_rate_calculation(np.count_nonzero(hash_verified), self.n_bks)
            R, R_theo, n_tilde, r, epsilon = generic.composable_key_rate(self.n_bks, self.N, n, self.p, self.q, R_code,
                                                                      R_M_star, x_M, d_ent, p_EC, e_ent, e_s, e_h, e_cor, e_PE, H_K)
            print("Composable key rate:", R, "Theoretical rate:", R_theo)

            # Free unnecessary variables from the memory before proceeding to privacy amplification
            del kA_dec, K_ubar, K_hat_bin, K_bar_bin, K_ubar_bin, ps_0, pr_0, ps_i, pr_i, CN, VN, VN_exc, H, H_sparse, K_sd, r_mn_i1, r_mn_i2, r_mn_i3

            # Measure the memory after error correction and verification and before privacy amplification
            peak_mem_ec = utilities.peak_memory_measurement(proc)

            # If the composable key rate positive, proceed to privacy amplification stage. Otherwise, the protocol is aborted.
            if R > 0:
                S_hat_bold = np.ravel(S_hat)
                S_bold = np.ravel(S)
                # Ensure the sequence to enter PA has the correct bit length
                if n_tilde != len(S_hat_bold):
                    raise RuntimeWarning("The length of the PA sequence is not correct. n_tilde:", n_tilde, "S_hat_bold:", S_hat_bold)
                elif n_tilde != len(S_bold):
                    raise RuntimeWarning("The length of the PA sequence is not correct. n_tilde:", n_tilde, "S_hat_bold:", S_bold)

                # Privacy amplification is performed. The process is timed.
                start = timeit.default_timer()
                K_bold = hashes.privacy_amplification(S_hat_bold, n_tilde, r, 0)
                stop = timeit.default_timer()
                pa_time = stop - start

                # Output the final key to a file
                try:
                    # np.savetxt("keys/" + "binary_of_" + str(sim_id), S_hat_bold, fmt='%d')
                    np.savetxt("keys/" + datetime.datetime.now().strftime("%d-%b-%Y (%H.%M.%S.%f)"), K_bold, fmt='%d')
                except PermissionError:
                    warnings.warn("The key could not be saved to the disk. Permissions are not set for this action.")
                except OSError:
                    warnings.warn("The key could not be saved to the disk. A possible cause is no hard disk space.")

            # Measure the peak memory consumption at the end of the simulation
            peak_mem_end = utilities.peak_memory_measurement(proc)

            # Measure the duration of the entire simulation
            end_timer = timeit.default_timer()
            sim_time = end_timer - start_timer

            return Results(s_z, Chi, Xi, d, m, n, t, delta, GF, SNR, I_XY, R_asy, T_hat, xi_hat, T_m, xi_m,
                           T_star_m, xi_star_m, x_M, R_m, R_M, R_M_star, I_XY_hat, SNR_hat, rho, rho_th, H_K, d_ent, R_code,
                           R_code_approx, l, k, wr, wc, R_des, beta_true, peak_mem_ec, peak_mem_end, ec_time, pa_time,
                           sim_time, time_avg, iter_avg, found, fnd_rnd, hash_verified, dec_time, p_EC, FER, r, R,
                           R_theo, n_tilde, epsilon)

        else:
            p_EC, FER = generic.frame_error_rate_calculation(self.n_bks * self.p_EC_tilde, self.n_bks)
            R, R_theo, n_tilde, r, epsilon = generic.composable_key_rate(self.n_bks, self.N, n, self.p, self.q, R_code,
                                                                      R_M_star, x_M, d_ent, p_EC, e_ent, e_s, e_h, e_cor, e_PE, H_K)
            print("R_M_star", R_M_star, "Composable key rate:", R, "Theoretical:", R_theo, "\n")
            return Results(s_z, Chi, Xi, d, m, n, t, delta, GF, SNR, I_XY, R_asy, T_hat, xi_hat, T_m, xi_m, T_star_m,
                           xi_star_m, x_M, R_m, R_M, R_M_star, I_XY_hat, SNR_hat, rho, rho_th, H_K, d_ent, R_code, R_code_approx, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p_EC, FER, r, R, R_theo, n_tilde, epsilon)
