#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Matteo Larcher
"""
"""
Test FMU
"""

#%% import libraries
from FMU_wrap import *
import os
import numpy as np
from fmpy.util import plot_result
import matplotlib.pyplot as plt
import math
import ctypes

#%%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   ____             __ _
#  / ___|___  _ __  / _(_) __ _ ___
# | |   / _ \| '_ \| |_| |/ _` / __|
# | |__| (_) | | | |  _| | (_| \__ \
#  \____\___/|_| |_|_| |_|\__, |___/
#                         |___/
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

start_time = 0.0
stop_time = 10.0
step_size = 0.1
show_plot = True

#%%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  _                    _   _____ __  __ _   _
# | |    ___   __ _  __| | |  ___|  \/  | | | |
# | |   / _ \ / _` |/ _` | | |_  | |\/| | | | |
# | |__| (_) | (_| | (_| | |  _| | |  | | |_| |
# |_____\___/ \__,_|\__,_| |_|   |_|  |_|\___/
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

dirname = os.path.dirname(__file__)
fmu_path = os.path.join(dirname, "fmu_model", "xy_model_om_dd_par.fmu")

# instantiate the class
fmu_model = FMU2_model(
    fmu_path,
    start_time=start_time,
    start_values={"x": 0.0, "y": 0.0},
    parameters={"const.k": 1.0, "custom_parameter1.p": 1.0},
    learnable_parameters={"custom_parameter1.p"},
    instance_name="xy_1"
)

#%%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  ____  _                 _       _   _
# / ___|(_)_ __ ___  _   _| | __ _| |_(_) ___  _ __
# \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \
#  ___) | | | | | | | |_| | | (_| | |_| | (_) | | | |
# |____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# result vector
res = []
# current time
time = fmu_model.time
# state vector
state = []

while time < stop_time:
    # set the inputs
    fmu_model.set_inputs({"x": math.sin(time), "y": math.cos(time)})

    # get the inputs
    inputs = fmu_model.get_inputs()

    # simulate
    fmu_model.do_step(step_size)

    # get the outputs
    outputs = fmu_model.get_outputs()

    # get directional derivative input/output
    directional_derivative_io = fmu_model.get_directional_derivative_io()

    # get directional derivative parameter
    directional_derivative_lp = fmu_model.get_directional_derivative_lp()

    # gather FMU state
    state.append(fmu_model.get_FMU_state())

    # get current time
    time = fmu_model.time

    # store the results
    res.append((time, *inputs, *outputs, *directional_derivative_io[0], *directional_derivative_io[1], *directional_derivative_lp[0], *directional_derivative_lp[1]))

# convert the results to a structured NumPy array
res = np.array(
    res,
    dtype=np.dtype(
        [
            ("time", np.float64),
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("z_int_p", np.float64),
            ("dz_dx", np.float64),
            ("dz_dy", np.float64),
            ("dz_int_dx", np.float64),
            ("dz_int_dy", np.float64),
            ("dz_dp", np.float64),
            ("dz_int_dp", np.float64),
        ]
    ),
)

#%% plot the results
if show_plot:
    plot_result(res)
    plt.show()

#%% restore te FMU state at t=5.0 and perform a step
fmu_model.set_FMU_state(state[49])
fmu_model.do_step(step_size)

#%% get the FMU outputs
outputs = fmu_model.get_outputs()
print(f"FMU outputs at t={fmu_model.time:.3f}: {outputs}")

#%% extract the value of c_void_p from state[0] and cast to int
print(int(state[0][0].value))
# cast back to c_void_p
print(ctypes.c_void_p(int(state[0][0].value)))

#%% terminate the FMU
fmu_model.terminate()

print("ALL DONE!")



# %%
