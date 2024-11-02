"""
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                     *
 * FMU custom layers                                                   *
 *                                                                     *
 *  @authors: Matteo Larcher                                           *
 *                                                                     *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# import libraries
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import tensorflow as tf
import ctypes

# custom libraries
from FMU_wrap import *


#  _____ __  __ _   _  ____     _ _
# |  ___|  \/  | | | |/ ___|___| | |
# | |_  | |\/| | | | | |   / _ \ | |
# |  _| | |  | | |_| | |__|  __/ | |
# |_|   |_|  |_|\___/ \____\___|_|_|

@tf.keras.utils.register_keras_serializable(package="Custom", name="FMUCell")
class FMUCell(tf.keras.layers.Layer):

    def __init__(
        self,
        fmu_path: str = None,
        start_time: float = 0.0,
        start_values: dict = None,
        parameters: dict = None,
        learnable_parameters: list | set = None,
        step_size: float = 0.1,
        do_step_in_gradient: bool = False,
        **kwargs
    ):
        super(FMUCell, self).__init__(**kwargs)

        # instantiate the class
        self.fmu_model = FMU2_model(
            fmu_path,
            start_time=start_time,
            start_values=start_values,
            parameters=parameters,
            learnable_parameters=learnable_parameters,
            instance_name=self.name,
        )

        self.do_step_in_gradient = do_step_in_gradient
        self.start_time = start_time
        self.dt = step_size
        self.state_size = 2 # state: [pointer value to the FMU state, FMU time]
        self.output_size = len(self.fmu_model.get_outputs())

    def build(self, input_shape):
        self.input_size = input_shape[-1]
        self.built = True

    def call(self, input_tensor, state):
        output = self.fmu_op(input_tensor, state)
        fmu_state = tf.py_function(
            func=lambda: tf.constant(
                self.fmu_model.get_FMU_state_value(),
                dtype=tf.keras.backend.floatx(),
            ),
            inp=[],
            Tout=tf.keras.backend.floatx(),
        )
        return output, [tf.reshape(fmu_state, [1, self.state_size])]

    @tf.custom_gradient
    def fmu_op(self, inputs, state):

        def fmu_step(inputs):
            self.fmu_model.set_inputs(tf.unstack(tf.reshape(inputs, [-1])))
            self.fmu_model.do_step(self.dt)
            return tf.convert_to_tensor(
                self.fmu_model.get_outputs(), dtype=tf.keras.backend.floatx()
            )

        outputs = tf.py_function(fmu_step, inp=[inputs], Tout=tf.keras.backend.floatx())

        def custom_grad(upstream):

            def grad_step(upstream, state, inputs):
                # set the FMU state
                self.fmu_model.set_FMU_state_value(state[0,0], set_time=False)
                if self.do_step_in_gradient:
                    # do step to compute the directional derivative at the effective current state
                    self.fmu_model.set_inputs(tf.unstack(tf.reshape(inputs, [-1])))
                    self.fmu_model.do_step(self.dt)

                # Compute the Jacobian with directional derivative
                jacobian = tf.convert_to_tensor(
                    self.fmu_model.get_directional_derivative_io(),
                    dtype=tf.keras.backend.floatx(),
                )

                # Sum over the rows of the Jacobian matrix to get the gradient w.r.t. each input
                grad = tf.reduce_sum(tf.transpose(upstream) * jacobian, axis=0)
                return grad

            grad = tf.py_function(
                grad_step, inp=[upstream, state, inputs], Tout=tf.keras.backend.floatx()
            )
            return tf.reshape(grad, [1, self.input_size]), None # TODO: add learnable parameters

        return (
            tf.reshape(outputs, [1, self.output_size]),
            custom_grad,
        )

    def reset_states(self):
        # reset FMU states
        self.fmu_model.reset_FMU()

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "state_size": self.state_size,
            }
        )
        return config


#  _____ __  __ _   _ _
# |  ___|  \/  | | | | |    __ _ _   _  ___ _ __
# | |_  | |\/| | | | | |   / _` | | | |/ _ \ '__|
# |  _| | |  | | |_| | |__| (_| | |_| |  __/ |
# |_|   |_|  |_|\___/|_____\__,_|\__, |\___|_|
#                                |___/

@tf.keras.utils.register_keras_serializable(package="Custom", name="FMULayer")
class FMULayer(tf.keras.layers.Layer):

    def __init__(
        self,
        fmu_path: str = None,
        start_time: float = 0.0,
        start_values: dict = None,
        parameters: dict = None,
        learnable_parameters: list | set = None,
        step_size: float = 0.1,
        do_step_in_gradient: bool = False,
        **kwargs
    ):
        super(FMULayer, self).__init__()

        # check if the backend is float64 (needed for FMU state pointer)
        if tf.keras.backend.floatx() != "float64":
            raise ValueError("The FMU layer only supports float64") # FIXME: add support for float32

        self.units = 1
        self.cell = FMUCell(
            fmu_path,
            start_time,
            start_values,
            parameters,
            learnable_parameters,
            step_size,
            do_step_in_gradient=do_step_in_gradient,
        )

        # set initial states
        self.initial_state = tf.constant(
            [self.cell.fmu_model.get_FMU_state_value()],
            dtype=tf.keras.backend.floatx(),
        )

        # Create the RNN layer with the custom cell
        self.rnn_layer = tf.keras.layers.RNN(self.cell, **kwargs)

    def build(self, input_shape):
        self.cell.build(input_shape)
        self.built = True

    def call(self, inputs):
        self.cell.reset_states()
        return self.rnn_layer(inputs, self.initial_state)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.cell.fmu_model.get_outputs()))
