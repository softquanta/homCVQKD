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

import homodyne
import generic
import plots
import numpy as np
import os

# Create necessary folders if they do not exist
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('keys'):
    os.makedirs('keys')
if not os.path.exists('codes'):
    os.makedirs('codes')
if not os.path.exists('data'):
    os.makedirs('data')

config = "normal"

# The normal mode for a simulation.
if config == "normal":
    vals = homodyne.Protocol(n_bks=10, N=20000, M=0.1, mu=None, A=0.2, L=5, T=None, eta=0.8, v_el=0.1, xi=0.01,
                             beta=0.92, iter_max=100, p=7, q=4, alpha=7, p_EC_tilde=1, protocol='homodyne',
                             ec=True, pe_alt=True, optimal_mu=False, load_code=True, save_data=False,
                             load_data=False, save_pe=False, load_pe=False)
    homodyne.Protocol.validity_checks(vals)
    results = homodyne.Protocol.processing(vals)
    homodyne.Protocol.file_logging(vals, results)

# Used to identify an accurate average estimate for the reconciliation efficiency, given a set of simulation parameters,
# over a sufficiently large number of runs (different data are being generated for every run).
elif config == "find_ideal_beta":

    # Input
    runs = 100
    desired_code_rate = 1 - 2 / 13
    print("Desired rate:", desired_code_rate)
    betas_file = "logs/L_7.npz"

    betas = np.zeros(shape=runs, dtype=np.float64)
    for j in range(runs):
        print("Run", j + 1, "starting...")
        vals = homodyne.Protocol(n_bks=100, N=200000, M=0.1, mu=20, A=0.2, L=7, T=None, eta=0.8, v_el=0.1, xi=0.01,
                                 beta=0.922, iter_max=200, p=7, q=4, alpha=7, protocol='homodyne', ec=False,
                                 pe_alt=True, load_code=False, save_data=False, load_data=False, save_pe=False,
                                 load_pe=False, optimal_mu=False, b_id=1)
        homodyne.Protocol.validity_checks(vals)
        r, i, h, q, p, d_ent, prot = homodyne.Protocol.processing(vals)
        b = generic.precise_reconciliation_efficiency(desired_code_rate, i, h, q, p, d_ent)
        if b > 1:
            raise RuntimeError("The specified desired code rate returns a reconciliation efficiency above 1.")
        else:
            betas[j] = b
        print("Run", j + 1, "Beta:", b, "Average beta until now:", np.average(betas[np.argwhere(betas)]), "\n")
    print("The reconciliation efficiency to use given these certain parameters is:", np.average(betas))

    np.savez_compressed(betas_file, beta=betas)

# Used to find the optimal beta for a particular simulation. The data are saved to the disk so that the simulation can
# be run afterwards in normal mode by using the accurately computed reconciliation efficiency.
elif config == "optimize_beta":
    rates = list()
    betas = dict()
    wc = 2
    vals = homodyne.Protocol(n_bks=5, N=360000, M=0.1, mu=None, A=0.2, L=5, T=None, eta=0.8, v_el=0.1, xi=0.01,
                             beta=0.92, iter_max=200, p=8, q=4, alpha=7, protocol='homodyne', ec=False,
                             pe_alt=True, load_code=False, save_data=False, load_data=False, save_pe=False,
                             load_pe=False, optimal_mu=False, b_id=1)
    homodyne.Protocol.validity_checks(vals)
    r, i, h, q, p, d_ent, prot = homodyne.Protocol.processing(vals)
    for j in range(50):
        wr = wc + 1 + j  # For a positive code rate, the minimum possible row weight is the column weight plus one
        rate = 1 - wc / wr
        rates.append(rate)
        b = generic.precise_reconciliation_efficiency(rates[j], i, h, q, p, d_ent)
        if b >= 1:  # Once the value of beta goes above one, the maximum possible row weight is found and the loop stops
            break
        elif b < 0:  # Negative betas are not added to the list of possible choices
            continue
        else:
            betas[wr] = (b, rates[j])
    for k, d in betas.items():
        print(str(k) + ":", str(d))

# Used to maximize the key rate with regards to the percentage of sacrificed states that go to parameter estimation.
elif config == "optimize_pe":
    m = np.arange(0.01, 1, 0.01)  # Examine every M from 1% to 99% with step 1%
    m = np.around(m, decimals=2)  # Eliminate numerical inaccuracies
    r = np.zeros_like(m, dtype=np.float64)  # Storage for composable key rates

    for i in range(len(m)):
        vals = homodyne.Protocol(n_bks=100, N=200000, M=m[i], mu=None, A=0.2, L=5, T=None, eta=0.8, v_el=0.1, xi=0.1,
                                 beta=0.92, iter_max=100, p=7, q=4, alpha=7, p_EC_tilde=1, protocol='homodyne',
                                 ec=False, pe_alt=True, optimal_mu=False, load_code=True, save_data=False,
                                 load_data=False, save_pe=False, load_pe=False)
        homodyne.Protocol.validity_checks(vals)
        results = homodyne.Protocol.processing(vals)
        r[i] = results.R_theo
    m_max, r_max = plots.rate_pe_runs_optimization(m, r)
    print("The optimal percentage for M is", m_max, "which yields a key rate of", r_max)

else:
    raise RuntimeError("This configuration does not exist. Please choose from an existing configuration.")
