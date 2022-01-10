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

import numpy as np
import os
from numba import njit
from sys import platform
if platform == "linux" or platform == "linux2":
    import resource


def percent_to_decibel_conversion(x):
    return 10 * np.log10(x)


def decibel_to_percent_conversion(x):
    return 10 ** (x / 10)


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


def peak_memory_measurement(proc):
    """"
    Measures the peak memory consumption of the software in MiB the until a certain runtime point.
    :param proc: The current process.
    """

    if os.name == "nt":  # Works only in Windows systems
        mem = proc.memory_full_info().peak_wset / 2 ** 20  # Original measurement in bytes
    else:
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Original measurement in KiB
    return mem



