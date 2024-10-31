#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Matteo Larcher
"""
"""
Test FMU
"""

#%% import libraries
from xy_FMU import xy_FMU_class
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
fmu_model = xy_FMU_class(
    fmu_path,
    start_time=start_time,
    start_values=[1.0,  # x
                  2.0], # y
    parameters=[2.0],  # k
    learnable_parameters=[5.0],  # p
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
# time vector
t_vect = np.arange(start_time, stop_time, step_size)
# state vector
state = []

for time in t_vect:
    # set the inputs
    fmu_model.set_inputs(
        x=math.sin(time),
        y=math.cos(time),
    )

    # get the inputs
    inputs = fmu_model.get_inputs()

    # simulate
    fmu_model.do_step(time, step_size)

    # get the outputs
    outputs = fmu_model.get_outputs()

    # get directional derivative input/output
    directional_derivative_io = fmu_model.get_directional_derivative()

    # get directional derivative parameter
    directional_derivative_p = fmu_model.get_directional_derivative_p()

    # gather FMU state
    state.append(fmu_model.get_FMU_state())

    # store the results
    res.append((time, *inputs, *outputs, *directional_derivative_io, *directional_derivative_p))

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
            ("dz_int_p_dp", np.float64),
        ]
    ),
)

#%% plot the results
if show_plot:
    plot_result(res)
    plt.show()

#%% restore te FMU state at t=5.0 and perform a step
fmu_model.set_FMU_state(state[50])
fmu_model.do_step(5.0, step_size)

#%% get the FMU outputs
outputs = fmu_model.get_outputs()
print("FMU outputs at t=5.0:", outputs)

#%% extract the value of c_void_p from state[0] and cast to int
print(int(state[0].value))
# cast back to c_void_p
print(ctypes.c_void_p(int(state[0].value)))

#%% terminate the FMU
fmu_model.terminate()

print("ALL DONE!")


