#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Matteo Larcher
"""

"""
Class to implement the FMU model
"""

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil


class xy_FMU_class(object):
    """class to implement the FMU model"""

    #
    def __init__(
        self,
        fmu_path,
        start_time=0.0,
        start_values=[0.0, 0.0],
        parameters=[1.0],
        learnable_parameters=[1.0],
        instance_name="xy_1",
    ):
        """class constructor
        fmu_path: path to the FMU file
        """

        # load the FMU
        model_description = read_model_description(fmu_path)
        # print("Model description:")
        # print(model_description)

        # collect the value references
        self.vrs = {}
        for variable in model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference

        # get the value references for the variables we want to get/set
        # Inputs:
        self.x = self.vrs["x"]
        self.y = self.vrs["y"]

        # Parameters:
        self.k = self.vrs["const.k"]

        # Learnable parameters:
        self.p = self.vrs["custom_parameter1.p"]

        # Outputs:
        self.z = self.vrs["z"]
        self.z_int = self.vrs["z_int"]

        # extract the FMU
        unzipdir = extract(fmu_path)

        self.fmu = FMU2Slave(
            guid=model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=model_description.coSimulation.modelIdentifier,
            instanceName=instance_name,
        )

        # initialize the FMU
        self.fmu.instantiate()

        # setup the experiment
        self.fmu.setupExperiment(startTime=start_time)

        # set the start values
        self.fmu.setReal([self.x, self.y, self.k, self.p], start_values + parameters + learnable_parameters)

        # enter initialization mode
        self.fmu.enterInitializationMode()

        # exit initialization mode
        self.fmu.exitInitializationMode()

        # store the FMU state
        self.fmu_initial_state = self.fmu.getFMUstate()

        # clean up
        shutil.rmtree(unzipdir, ignore_errors=True)

    #
    def set_inputs(
        self,
        x=None,
        y=None,
    ):
        """set the inputs"""

        self.fmu.setReal(
            [self.x, self.y],
            [
                x if x is not None else self.fmu.getReal([self.x])[0],
                y if y is not None else self.fmu.getReal([self.y])[0],
            ],
        )

    #
    def get_inputs(self):
        """get the inputs"""

        return self.fmu.getReal([
            self.x,
            self.y,
        ])

    #
    def get_outputs(self):
        """get the outputs"""

        return self.fmu.getReal([self.z, self.z_int])

    #
    def get_directional_derivative(self):
        """get the directional derivative"""

        dz_dx = self.fmu.getDirectionalDerivative([self.z], [self.x], [1.0])[0]
        dz_dy = self.fmu.getDirectionalDerivative([self.z], [self.y], [1.0])[0]

        return [dz_dx, dz_dy]

    #
    def get_directional_derivative_p(self):
        """get the directional derivative"""

        dz_int_dp = self.fmu.getDirectionalDerivative([self.z_int], [self.p], [1.0])[0]

        return [dz_int_dp]

    #
    def get_FMU_state(self):
        """get the FMU state"""

        return self.fmu.getFMUstate()

    #
    def set_FMU_state(self, state):
        """set the FMU state"""

        self.fmu.setFMUstate(state)

    #
    def free_FMU_state(self, state):
        """free the FMU state"""

        self.fmu.freeFMUstate(state)

    #
    def serialize_FMU_state(self, state):
        """serialize the FMU state"""

        return self.fmu.serializeFMUstate(state)

    #
    def deserialize_FMU_state(self, state):
        """deserialize the FMU state"""

        return self.fmu.deSerializeFMUstate(state)

    #
    def reset_FMU(self):
        """reset the FMU"""

        self.set_FMU_state(self.fmu_initial_state)

    #
    def do_step(self, current_time, step_size):
        """do a step"""

        self.fmu.doStep(currentCommunicationPoint=current_time,
                        communicationStepSize=step_size)

    #
    def terminate(self):
        """terminate the FMU"""

        self.fmu.terminate()
        self.fmu.freeInstance()


# EOF: xy_FMU.py
