# plots.py
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

import matplotlib.pyplot as plt
import numpy as np


def rate_pe_runs_optimization(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel("Number of PE Runs $M$")
    plt.ylabel("Rate $R$")
    ax.plot(x, y, marker='o')

    # Identify the M, for which the key rate R is maximized
    x_max = x[np.argmax(y)]
    y_max = y.max()
    ax.annotate('Max', xy=(x_max, y_max), xytext=(x_max, y_max), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.grid()
    plt.show()
    return x_max, y_max
