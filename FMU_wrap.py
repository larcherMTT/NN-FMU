"""
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                     *
 * FMU wrapper class                                                   *
 *                                                                     *
 *  @authors: Matteo Larcher                                           *
 *                                                                     *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
"""

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil


class FMU2_model(object):
    """class to implement the FMU model"""

    #
    def __init__(
        self,
        fmu_path: str = None,
        start_time: float = 0.0,
        start_values: dict = None,
        parameters: dict = None,
        learnable_parameters: list | set = None,
        instance_name: str = "instance1",
    ):
        """class constructor
        fmu_path: path to the FMU file
        start_time: start time
        start_values: dictionary with the start values
        parameters: dictionary with the parameters values
        learnable_parameters: list of learnable parameters
        """

        # load the FMU
        self.model_description = read_model_description(fmu_path)

        # collect the value references
        self.vrs = {}
        for variable in self.model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference

        # check if the inputs dicts comply with the FMU variables
        if start_values is not None:
            for key in start_values.keys():
                if key not in self.vrs.keys():
                    raise ValueError(f"Variable {key} not found in the FMU")
        if parameters is not None:
            for key in parameters.keys():
                if key not in self.vrs.keys():
                    raise ValueError(f"Variable {key} not found in the FMU")
        if learnable_parameters is not None:
            for key in learnable_parameters:
                if key not in self.vrs.keys():
                    raise ValueError(f"Variable {key} not found in the FMU")

        # extract the FMU
        unzipdir = extract(fmu_path)

        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName=instance_name,
        )

        # FMU time
        self.time = start_time

        # FMU outputs
        self.out = {}
        for output in self.model_description.outputs:
            self.out[output.variable.name] = output.variable.valueReference

        # FMU inputs
        self.inp = {}
        for input in self.model_description.modelVariables:
            if input.causality == "input":
                self.inp[input.name] = input.valueReference

        # FMU learnable parameters
        self.learnable_parameters = {}
        for p in learnable_parameters:
            self.learnable_parameters[p] = self.vrs[p]

        # initialize the FMU
        self.fmu.instantiate()

        # setup the experiment
        self.fmu.setupExperiment(startTime=start_time)

        # set the parameters and learnable parameters
        self.fmu.setReal([self.vrs[key] for key in parameters.keys()], list(parameters.values()))

        # enter initialization mode
        self.fmu.enterInitializationMode()

        # set the start values
        self.fmu.setReal([self.vrs[key] for key in start_values.keys()], list(start_values.values()))

        # exit initialization mode
        self.fmu.exitInitializationMode()

        # store the FMU state
        self.fmu_initial_state = [self.fmu.getFMUstate(), self.time]

        # clean up
        shutil.rmtree(unzipdir, ignore_errors=True)

    #
    def set_inputs(self, inputs: dict = None):
        """set the inputs"""

        self.fmu.setReal([self.inp[key] for key in inputs.keys()], list(inputs.values()))

        #
    def set_known(self, knw: dict = None):
        """set the known inputs and parameters"""

        self.fmu.setReal([self.vrs[key] for key in knw.keys()], list(knw.values()))

    #
    def get_inputs(self):
        """get the inputs"""

        return self.fmu.getReal(list(self.inp.values()))

    #
    def get_outputs(self):
        """get the outputs"""

        return self.fmu.getReal(list(self.out.values()))

    #
    def get_outputs_names(self):
        """get the outputs names"""

        return self.out.keys()

    #
    def get_directional_derivative_io(self):
        """get the directional derivative of outputs w.r.t. inputs"""

        der = [] # FIXME: consider preallocating the list for performance
        # construct the jacobian matrix (column-wise)
        for inp in self.inp.keys():
            der.append(self.fmu.getDirectionalDerivative(list(self.out.values()), [self.inp[inp]], [1.0]))

        return list(map(list, zip(*der))) # FIXME: transpose the matrix during the construction (see getAdjointDerivative fmi3)

    #
    def get_directional_derivative_lp(self):
        """get the directional derivative of outputs w.r.t. learnable parameters"""

        der = [] # FIXME: consider preallocating the list for performance
        # construct the jacobian matrix (column-wise)
        for p in self.learnable_parameters.keys():
            der.append(self.fmu.getDirectionalDerivative(list(self.out.values()), [self.learnable_parameters[p]], [1.0]))

        return list(map(list, zip(*der))) # FIXME: transpose the matrix during the construction (see getAdjointDerivative fmi3)

    #
    def get_FMU_state(self):
        """get the FMU state"""

        return [self.fmu.getFMUstate(), self.time]

    #
    def get_model_description(self):
        """get the model description"""

        return self.model_description

    #
    def get_FMU_time(self):
        """get the FMU time"""

        return self.time

    #
    def set_FMU_state(self, state):
        """set the FMU state"""

        self.fmu.setFMUstate(state[0])
        self.time = state[1]

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
    def do_step(self, step_size):
        """do a step"""

        self.time += step_size

        self.fmu.doStep(currentCommunicationPoint=self.time,
                        communicationStepSize=step_size)

    #
    def terminate(self):
        """terminate the FMU"""

        self.fmu.terminate()
        self.fmu.freeInstance()


# EOF: FMU_wrap.py
